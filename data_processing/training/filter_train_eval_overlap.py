#!/usr/bin/env python3
"""Exclude entire benchmark source videos from training JSONL and optional assets.

The benchmark TSV must contain canonical_video_id. Training records must use
the same ID in canonical_video_id or metadata.canonical_video_id. Legacy
video_id fields require an explicit --source-id-map. Filenames and clip IDs are
never used to infer the source identity. Missing IDs are excluded and counted.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def load_eval_keys(eval_dir: Path) -> tuple[set[str], dict[str, Any]]:
    manifest = eval_dir / "manifest.tsv"
    keys: set[str] = set()
    rows = 0
    with manifest.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if "canonical_video_id" not in (reader.fieldnames or []):
            raise ValueError(f"{manifest}: canonical_video_id column is required")
        for rows, row in enumerate(reader, 1):
            key = str(row.get("canonical_video_id") or "").strip()
            if not key:
                raise ValueError(f"{manifest}: row {rows} has no canonical_video_id")
            keys.add(key)
    return keys, {"manifest_rows": rows, "eval_canonical_video_ids": len(keys)}


def row_video_id(row: dict[str, Any], source_id_map: dict[str, str] | None = None) -> str | None:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    values = [row.get("canonical_video_id"), metadata.get("canonical_video_id")]
    canonical = {str(value).strip() for value in values if value is not None and str(value).strip()}
    # A legacy video_id can be a recording or segment name in another namespace.
    # Comparing it directly with benchmark canonical IDs silently retains overlap.
    # Resolve it only through an explicit source-ID mapping supplied by the caller.
    if source_id_map:
        for value in (row.get("video_id"), metadata.get("video_id")):
            if value is not None and str(value).strip() in source_id_map:
                canonical.add(source_id_map[str(value).strip()])
    if len(canonical) > 1:
        raise ValueError("Record has conflicting canonical_video_id values or source-ID mapping")
    return canonical.pop() if canonical else None


def load_source_id_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or any(
        not isinstance(key, str) or not key.strip()
        or not isinstance(value, str) or not value.strip()
        for key, value in mapping.items()
    ):
        raise ValueError("--source-id-map must be a JSON object of source ID to canonical video ID strings")
    return {key.strip(): value.strip() for key, value in mapping.items()}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            yield row


def filter_jsonl(src: Path, dst: Path, eval_keys: set[str], reindex: bool = True,
                 source_id_map: dict[str, str] | None = None) -> dict[str, Any]:
    total = kept = missing = 0
    removed_ids: Counter[str] = Counter()
    kept_ids: set[str] = set()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fh:
        for row in iter_jsonl(src):
            total += 1
            vid = row_video_id(row, source_id_map)
            if not vid:
                missing += 1
                continue
            if vid in eval_keys:
                removed_ids[vid] += 1
                continue
            kept_ids.add(vid)
            if reindex and "_index" in row:
                row["_index"] = kept
            kept += 1
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "input": str(src), "output": str(dst), "total_rows": total,
        "kept_rows": kept, "removed_rows": total - kept,
        "overlap_rows": sum(removed_ids.values()), "missing_video_id_rows": missing,
        "kept_video_ids": len(kept_ids), "kept_canonical_video_ids": sorted(kept_ids),
        "removed_video_ids": sorted(removed_ids),
    }


def filter_assets(src_dir: Path, dst_dir: Path, kept_ids: set[str], *, captions: bool) -> dict[str, int]:
    """Copy assets named by canonical ID; preserve files only for kept sources."""
    total = kept = 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_dir.iterdir()):
        if captions and src.suffix == ".json":
            vid = src.stem
        elif not captions and src.suffix == ".mp4":
            vid = src.stem
        elif not captions and src.is_dir() and src.name.endswith("_clips"):
            vid = src.name.removesuffix("_clips")
        else:
            continue
        total += 1
        if vid not in kept_ids:
            continue
        if src.is_dir():
            shutil.copytree(src, dst_dir / src.name, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst_dir / src.name)
        kept += 1
    return {"total_entries": total, "kept_entries": kept, "removed_entries": total - kept}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True, help="Directory containing manifest.tsv")
    parser.add_argument("--input-jsonl", type=Path, required=True, action="append",
                        help="Training JSONL; repeat for multiple inputs with distinct filenames")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--captions-dir", type=Path, help="Optional caption files named <canonical_video_id>.json")
    parser.add_argument("--videos-dir", type=Path, help="Optional media named <canonical_video_id>.mp4 or <ID>_clips/")
    parser.add_argument("--no-reindex", action="store_true")
    parser.add_argument("--source-id-map", type=Path,
                        help="Optional JSON object mapping legacy video_id values to canonical_video_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = args.input_jsonl
    if len({src.name for src in inputs}) != len(inputs):
        raise ValueError("Input JSONL filenames must be distinct")
    if any((args.output_root / src.name).resolve() == src.resolve() for src in inputs):
        raise ValueError("Output JSONL must differ from input")
    # Reusing an asset directory can retain files excluded on a later run.
    # Write optional media into a fresh directory to keep exclusion complete.
    for source, subdir in ((args.captions_dir, "captions"), (args.videos_dir, "videos")):
        destination = args.output_root / subdir
        if source is not None and destination.exists():
            raise ValueError(f"Use a fresh asset output directory: {destination}")
    eval_keys, eval_info = load_eval_keys(args.eval_dir)
    source_id_map = load_source_id_map(args.source_id_map)
    reports = [filter_jsonl(src, args.output_root / src.name, eval_keys, not args.no_reindex, source_id_map)
               for src in inputs]
    report: dict[str, Any] = {"policy": "Exclude complete canonical benchmark source videos and records with unknown source IDs.",
                              "eval": eval_info, "source_id_map_entries": len(source_id_map), "jsonl": reports}
    kept_ids = {value for item in reports for value in item["kept_canonical_video_ids"]}
    if args.captions_dir:
        report["captions"] = filter_assets(args.captions_dir, args.output_root / "captions", kept_ids, captions=True)
    if args.videos_dir:
        report["videos"] = filter_assets(args.videos_dir, args.output_root / "videos", kept_ids, captions=False)
    report_path = args.output_root / "filter_eval_overlap_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
