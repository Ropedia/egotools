#!/usr/bin/env python3
"""Build an experimental temporal split by excluding reserved QA intervals.

This protocol is not source-video-disjoint and is not the paper training split.
All paths are explicit. Video/caption filenames and training video IDs must use
the same canonical IDs as the benchmark manifest. Requires ffmpeg/ffprobe and
constant-frame-rate video for frame-indexed annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

MIN_TRAIN_SEGMENT_SECONDS = 1.0
EPS = 1e-3


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def parse_ratio(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    return float(Fraction(value))


def ffprobe_video(path: Path) -> dict[str, float]:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate,duration,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    data = json.loads(out)
    stream = data["streams"][0]
    duration = float(stream.get("duration") or 0.0)
    fps = parse_ratio(stream.get("avg_frame_rate")) or parse_ratio(stream.get("r_frame_rate"))
    nb_frames = stream.get("nb_frames")
    if (not fps or fps <= 0) and nb_frames and duration > 0:
        fps = float(nb_frames) / duration
    if not fps or fps <= 0 or not math.isfinite(fps) or not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Cannot determine positive duration and frame rate for {path}")
    return {"duration": duration, "fps": fps}


def load_benchmark_records(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()] if path.suffix == ".jsonl" else json.load(fh)
    records = {}
    for row in rows:
        qa_id = row.get("qa_id") or row.get("annotation_id")
        if qa_id and row.get("clip_start_seconds") is not None and row.get("clip_end_seconds") is not None:
            records[str(qa_id)] = row
    return records


def load_eval_intervals(eval_dir: Path, benchmark_full: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    records = load_benchmark_records(benchmark_full)
    manifest = eval_dir / "manifest.tsv"
    intervals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_timestamps = []
    with manifest.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            qa_id = row["qa_id"]
            rec = records.get(qa_id)
            if not rec:
                missing_timestamps.append(qa_id)
                continue
            video_id = str(row.get("canonical_video_id") or "").strip()
            if not video_id:
                raise ValueError(f"Missing canonical_video_id for QA {qa_id}")
            start = float(rec["clip_start_seconds"])
            end = float(rec["clip_end_seconds"])
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise ValueError(f"Invalid evaluation interval for QA {qa_id}: {start}, {end}")
            intervals[video_id].append(
                {
                    "start": start,
                    "end": end,
                    "qa_id": qa_id,
                    "clip_video": row.get("clip_video"),
                    "video": row.get("video"),
                }
            )
    if missing_timestamps:
        raise ValueError(f"Missing evaluation timestamps for {len(missing_timestamps)} QA IDs")
    report = {
        "manifest_rows": sum(len(v) for v in intervals.values()) + len(missing_timestamps),
        "timestamped_rows": sum(len(v) for v in intervals.values()),
        "missing_timestamp_rows": len(missing_timestamps),
        "missing_timestamp_qa_ids": missing_timestamps,
        "eval_video_ids": len(intervals),
    }
    return intervals, report


def merge_intervals(intervals: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    points = []
    for item in intervals:
        start = max(0.0, min(duration, float(item["start"])))
        end = max(0.0, min(duration, float(item["end"])))
        if end - start > EPS:
            points.append({"start": start, "end": end, "qa_ids": [item["qa_id"]]})
    points.sort(key=lambda x: (x["start"], x["end"]))
    merged: list[dict[str, Any]] = []
    for item in points:
        if not merged or item["start"] > merged[-1]["end"] + EPS:
            merged.append(item)
        else:
            merged[-1]["end"] = max(merged[-1]["end"], item["end"])
            merged[-1]["qa_ids"].extend(item["qa_ids"])
    return merged


def complement_intervals(eval_intervals: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    train = []
    cursor = 0.0
    for item in eval_intervals:
        if item["start"] - cursor >= MIN_TRAIN_SEGMENT_SECONDS:
            train.append({"start": cursor, "end": item["start"]})
        cursor = max(cursor, item["end"])
    if duration - cursor >= MIN_TRAIN_SEGMENT_SECONDS:
        train.append({"start": cursor, "end": duration})
    return train


def symlink_or_replace(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(str(src.resolve()), dst)


def copy_json(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def create_train_segment(src: Path, dst: Path, start: float, end: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Decode and re-encode for accurate interval boundaries. Stream-copy seeks
    # can retain frames before the requested start at the previous keyframe.
    run([
        "ffmpeg", "-v", "error", "-y", "-i", str(src),
        "-ss", f"{start:.6f}", "-t", f"{end - start:.6f}",
        "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", str(dst),
    ])


def overlaps(start: float, end: float, intervals: list[dict[str, Any]]) -> bool:
    return any(start < item["end"] - EPS and end > item["start"] + EPS for item in intervals)


def find_train_segment(start: float, end: float, segments: list[dict[str, Any]]) -> dict[str, Any] | None:
    for seg in segments:
        if start >= seg["start"] - EPS and end <= seg["end"] + EPS:
            return seg
    return None


def row_video_id(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    for value in (row.get("canonical_video_id"), metadata.get("canonical_video_id"),
                  row.get("video_id"), metadata.get("video_id")):
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def row_sample_type(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and metadata.get("sample_type"):
        return str(metadata["sample_type"])
    return str(row.get("sample_type") or "missing")


def row_frame_range(row: dict[str, Any]) -> tuple[int | None, int | None]:
    metadata = row.get("metadata")
    start = row.get("start_frame")
    end = row.get("end_frame")
    if start is None and isinstance(metadata, dict):
        start = metadata.get("start_frame")
    if end is None and isinstance(metadata, dict):
        end = metadata.get("end_frame")
    if start is None or end is None:
        return None, None
    return int(start), int(end)


def set_video_path(row: dict[str, Any], rel_path: str) -> None:
    if "videos" in row and isinstance(row["videos"], list) and row["videos"]:
        row["videos"] = [rel_path]
    if "_video_rel" in row:
        row["_video_rel"] = rel_path


def shift_frames(row: dict[str, Any], frame_offset: int) -> None:
    start, end = row_frame_range(row)
    if start is None or end is None:
        return
    new_start = max(1, start - frame_offset)
    new_end = max(new_start, end - frame_offset)
    if "start_frame" in row:
        row["start_frame"] = new_start
    if "end_frame" in row:
        row["end_frame"] = new_end
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        if "start_frame" in metadata:
            metadata["start_frame"] = new_start
        if "end_frame" in metadata:
            metadata["end_frame"] = new_end


def annotate_trim(row: dict[str, Any], original_video_id: str, segment: dict[str, Any] | None) -> None:
    metadata = row.get("metadata")
    target = metadata if isinstance(metadata, dict) else row
    target["_original_video_id"] = original_video_id
    if segment is not None:
        target["_train_segment_id"] = segment["segment_id"]
        target["_train_segment_start_seconds"] = segment["start"]
        target["_train_segment_end_seconds"] = segment["end"]


def filter_jsonl(
    src: Path,
    dst: Path,
    eval_by_video: dict[str, list[dict[str, Any]]],
    train_segments_by_video: dict[str, list[dict[str, Any]]],
    video_info: dict[str, dict[str, float]],
) -> dict[str, Any]:
    total = kept = removed = no_segment = missing_source_video = 0
    sample_types = Counter()
    removed_types = Counter()
    kept_video_ids = set()
    removed_video_ids = set()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as in_fh, dst.open("w", encoding="utf-8") as out_fh:
        for line in in_fh:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            vid = row_video_id(row)
            sample_type = row_sample_type(row)
            sample_types[sample_type] += 1
            if not vid:
                missing_source_video += 1
                removed += 1
                removed_types[sample_type] += 1
                continue
            if vid not in video_info:
                missing_source_video += 1
                removed += 1
                removed_types[sample_type] += 1
                removed_video_ids.add(vid)
                continue
            start_frame, end_frame = row_frame_range(row)
            fps = video_info.get(vid, {}).get("fps", 20.0)
            start_s = ((start_frame or 1) - 1) / fps
            end_s = (end_frame or start_frame or 1) / fps
            eval_intervals = eval_by_video.get(vid, [])
            if eval_intervals and (start_frame is None or end_frame is None or start_frame < 1 or end_frame < start_frame):
                no_segment += 1
                removed += 1
                removed_types[sample_type] += 1
                continue
            if eval_intervals and overlaps(start_s, end_s, eval_intervals):
                removed += 1
                removed_types[sample_type] += 1
                removed_video_ids.add(vid)
                continue
            segment = None
            rel_path = f"videos/train/{vid}.mp4"
            if eval_intervals:
                segment = find_train_segment(start_s, end_s, train_segments_by_video[vid])
                if segment is None:
                    no_segment += 1
                    removed += 1
                    removed_types[sample_type] += 1
                    removed_video_ids.add(vid)
                    continue
                rel_path = f"videos/train/{segment['filename']}"
                frame_offset = int(round(segment["start"] * fps))
                shift_frames(row, frame_offset)
            set_video_path(row, rel_path)
            annotate_trim(row, vid, segment)
            kept_video_ids.add(vid)
            kept += 1
            if "_index" in row:
                row["_index"] = kept - 1
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "input": str(src),
        "output": str(dst),
        "total_rows": total,
        "kept_rows": kept,
        "removed_rows": removed,
        "removed_rows_without_train_segment": no_segment,
        "removed_rows_missing_source_video": missing_source_video,
        "kept_video_ids": len(kept_video_ids),
        "removed_video_ids": len(removed_video_ids),
        "sample_type_counts": dict(sample_types),
        "removed_sample_type_counts": dict(removed_types),
    }


def frame_number(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        return int(digits) if digits else None
    return None


def clean_caption_json(src: Path, dst: Path, eval_intervals: list[dict[str, Any]], fps: float) -> dict[str, Any]:
    data = json.load(src.open("r", encoding="utf-8"))
    segments = data.get("segments")
    if not isinstance(segments, list):
        raise ValueError(f"{src}: temporal filtering requires a segments list")
    kept_segments = []
    removed = 0
    for seg in segments:
        start = frame_number(seg.get("start_frame"))
        end = frame_number(seg.get("end_frame"))
        if start is None or end is None or start < 1 or end < start:
            removed += 1
            continue
        start_s = (start - 1) / fps
        end_s = end / fps
        if overlaps(start_s, end_s, eval_intervals):
            removed += 1
        else:
            kept_segments.append(seg)
    data["segments"] = kept_segments
    data["_eval_overlap_removed"] = {
        "removed_segments": removed,
        "source_segments": len(segments),
        "policy": "Removed caption segments whose frame range overlaps benchmark eval intervals.",
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"input_segments": len(segments), "kept_segments": len(kept_segments), "removed_segments": removed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--benchmark-full", type=Path, required=True)
    parser.add_argument("--captions-dir", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, action="append", required=True,
                        help="Training JSONL; repeat for distinct input filenames")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    out = args.output_root
    if len({path.name for path in args.input_jsonl}) != len(args.input_jsonl):
        raise ValueError("Input JSONL filenames must be distinct")
    if any((out / path.name).resolve() == path.resolve() for path in args.input_jsonl):
        raise ValueError("Output JSONL must differ from input")
    # Old full-video files in train/ could expose newly excluded intervals.
    if (out / "videos").exists() or (out / "captions").exists():
        raise ValueError("Use a fresh output directory for temporal media")
    train_video_dir = out / "videos" / "train"
    train_caption_dir = out / "captions" / "train"
    generated_out = out

    eval_intervals_raw, eval_report = load_eval_intervals(args.eval_dir, args.benchmark_full)

    video_info: dict[str, dict[str, float]] = {}
    eval_by_video: dict[str, list[dict[str, Any]]] = {}
    train_segments_by_video: dict[str, list[dict[str, Any]]] = {}
    video_report = {
        "train_symlinked_full_videos": 0,
        "train_trimmed_source_videos": 0,
        "train_segments_created": 0,
    }

    for src in sorted(args.videos_dir.glob("*.mp4")):
        vid = src.stem
        info = ffprobe_video(src)
        video_info[vid] = info
        raw_intervals = eval_intervals_raw.get(vid, [])
        if not raw_intervals:
            symlink_or_replace(src, train_video_dir / src.name)
            video_report["train_symlinked_full_videos"] += 1
            continue
        if any(item["end"] > info["duration"] + EPS for item in raw_intervals):
            raise ValueError(f"Evaluation interval exceeds source duration for {vid}")
        merged = merge_intervals(raw_intervals, info["duration"])
        train_segments = complement_intervals(merged, info["duration"])
        eval_by_video[vid] = merged
        train_segments_by_video[vid] = []
        video_report["train_trimmed_source_videos"] += 1
        for idx, seg in enumerate(train_segments):
            filename = f"{vid}__trainseg{idx:03d}_{int(round(seg['start'] * 1000)):010d}_{int(round(seg['end'] * 1000)):010d}.mp4"
            dst = train_video_dir / filename
            create_train_segment(src, dst, seg["start"], seg["end"])
            item = dict(seg)
            item["segment_id"] = f"{vid}__trainseg{idx:03d}"
            item["filename"] = filename
            train_segments_by_video[vid].append(item)
            video_report["train_segments_created"] += 1

    caption_report = {
        "train_caption_files": 0,
        "skipped_missing_source_video": 0,
        "removed_segments": 0,
    }
    for src in sorted(args.captions_dir.glob("*.json")):
        vid = src.stem
        if vid not in video_info:
            caption_report["skipped_missing_source_video"] += 1
            continue
        if vid in eval_by_video:
            stats = clean_caption_json(src, train_caption_dir / src.name, eval_by_video[vid], video_info[vid]["fps"])
            caption_report["removed_segments"] += stats["removed_segments"]
        else:
            copy_json(src, train_caption_dir / src.name)
        caption_report["train_caption_files"] += 1

    jsonl_reports = []
    for src in args.input_jsonl:
        jsonl_reports.append(
            filter_jsonl(src, generated_out / src.name, eval_by_video, train_segments_by_video, video_info)
        )

    report = {
        "policy": (
            "Experimental split excluding benchmark eval time spans; "
            "train videos are full-video symlinks for non-overlap sources and trimmed subclips for overlap sources."
        ),
        "eval": eval_report,
        "videos": video_report,
        "captions": caption_report,
        "jsonl": jsonl_reports,
        "eval_intervals_by_video": eval_by_video,
        "train_segments_by_video": train_segments_by_video,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "temporal_exclusion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote experimental temporal training split to {out}")
    print(
        json.dumps(
            {
                "eval": eval_report,
                "videos": video_report,
                "captions": caption_report,
                "jsonl": [
                    {
                        "file": Path(item["output"]).name,
                        "total_rows": item["total_rows"],
                        "kept_rows": item["kept_rows"],
                        "removed_rows": item["removed_rows"],
                        "removed_rows_without_train_segment": item["removed_rows_without_train_segment"],
                        "removed_rows_missing_source_video": item["removed_rows_missing_source_video"],
                    }
                    for item in jsonl_reports
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
