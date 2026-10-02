"""Exercise the public data-processing entrypoints with synthetic source data."""

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from egotools.data.benchmark import build_benchmark
from egotools.data.splits import build_train_temporal_split as temporal

ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = "egotools.data.benchmark."


@pytest.fixture(autouse=True)
def working_directory_outside_checkout(tmp_path, monkeypatch):
    """Exercise installed modules without relying on the repository as cwd."""
    monkeypatch.chdir(tmp_path)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def cli(module, *args):
    result = subprocess.run([sys.executable, "-m", module, *map(str, args)],
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_offline_qa_to_reviewed_benchmark_and_source_exclusion(tmp_path):
    source = tmp_path / "sources" / "source1"
    source.mkdir(parents=True)
    (source / "full.mp4").write_bytes(b"synthetic-full-video")
    (source / "clip.mp4").write_bytes(b"synthetic-short-clip")
    (source / "annotation_manifest.json").write_text(json.dumps({
        "canonical_video_id": "video1", "source_video_path": "full.mp4",
        "annotations": [{"annotation_id": "a1", "clip_path": "clip.mp4"}],
    }))
    raw = tmp_path / "input" / "qa.jsonl"
    write_jsonl(raw, [{
        "qa_id": "source1#a1", "annotation_id": "a1", "source_file": "source1.json",
        "canonical_video_id": "video1", "question": "Which tool is being used?",
        "answer": "hammer", "distractors": ["pliers", "wrench", "chisel", "sander", "mallet", "shears", "trowel"],
        "annotator_id": "private-author", "editor": "private-editor",
    }])
    normalized = tmp_path / "normalized"
    cli(MODULE_ROOT + "normalize_to_8choice", "--input-jsonl", raw, "--output-dir", normalized,
        "--mode", "checks_only")
    rows = read_jsonl(normalized / "normalized.jsonl")
    assert len(rows) == 1
    assert rows[0]["normalized_options"][rows[0]["correct_index"]] == "hammer"
    assert len(rows[0]["normalized_options"]) == 8
    assert "annotator_id" not in rows[0]
    normalization_report = json.loads((normalized / "normalization_report.json").read_text())
    assert not (normalized / "run_manifest.json").exists()
    cli(MODULE_ROOT + "build_benchmark", "--normalized-run", normalized, "--output-root", tmp_path / "builds",
        "--version", "review", "--date", "20260101", "--workspace-sources-root", tmp_path / "sources",
        "--export-review-records")
    review_package = tmp_path / "builds" / "review_20260101"
    assert "private-" not in (review_package / "review_records.jsonl").read_text()
    assert json.loads((review_package / "build_report.json").read_text())["source_git_sha"] == normalization_report["git_sha"]
    edits = tmp_path / "edits.jsonl"
    write_jsonl(edits, [
        {"qa_id": "source1#a1", "saved_at_utc": "2026-01-01T00:00:00Z", "edited_question": "Old question", "review_status": "pending"},
        {"qa_id": "source1#a1", "saved_at_utc": "2026-01-02T00:00:00Z", "edited_question": "Which tool is visible?", "review_status": "clean", "editor": "private-editor"},
    ])
    reviewed = tmp_path / "reviewed.jsonl"
    cli(MODULE_ROOT + "extract_final_reviewed_qa", "--review-records", review_package / "review_records.jsonl",
        "--edits", edits, "--package-dir", review_package, "--output", reviewed,
        "--report", tmp_path / "reports" / "review.json")
    reviewed_row = read_jsonl(reviewed)[0]
    assert reviewed_row["question"] == "Which tool is visible?"
    assert "latest_edit_editor" not in reviewed_row
    tracked = tmp_path / "tracked.jsonl"
    cli(MODULE_ROOT + "assign_benchmark_research_tracks", "--input", reviewed, "--output", tracked,
        "--report", tmp_path / "reports" / "tracks.json", "--exclude-dropped")
    # Reusing a review build must not mix newer questions with old review records.
    original_package = {p.relative_to(review_package): p.read_bytes()
                        for p in review_package.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        build_benchmark.build(normalized_run=tracked.parent, normalized_jsonl=tracked,
                              output_root=tmp_path / "builds", version="review", date="20260101",
                              link_mode="copy", dry_run=False, limit=None,
                              workspace_sources_root=tmp_path / "sources", repo_root=ROOT)
    assert {p.relative_to(review_package): p.read_bytes()
            for p in review_package.rglob("*") if p.is_file()} == original_package
    cli(MODULE_ROOT + "build_benchmark", "--input-jsonl", tracked, "--output-root", tmp_path / "builds",
        "--version", "final", "--date", "20260101", "--workspace-sources-root", tmp_path / "sources")
    final = tmp_path / "builds" / "final_20260101"
    with (final / "manifest.tsv").open() as fh:
        manifest = list(csv.DictReader(fh, delimiter="\t"))
    assert len(manifest) == 1
    assert manifest[0][manifest[0]["answer"]] == "hammer"
    assert manifest[0]["research_track_id"]
    assert (final / manifest[0]["video"]).read_bytes() == b"synthetic-full-video"
    assert "private-" not in (final / "manifest.tsv").read_text()
    assert not (final / "review_records.jsonl").exists()
    assert not (final / "manifest.jsonl").exists()
    assert not (final / "build_manifest.json").exists()
    training = tmp_path / "train.jsonl"
    write_jsonl(training, [
        {"canonical_video_id": "video1", "_index": 4},
        {"metadata": {"canonical_video_id": "video2"}, "_index": 8},
        {"videos": ["unresolvable-clip.mp4"], "_index": 11},
    ])
    cli("egotools.data.splits.filter_train_eval_overlap", "--eval-dir", final,
        "--input-jsonl", training, "--output-root", tmp_path / "train-clean")
    kept = read_jsonl(tmp_path / "train-clean" / "train.jsonl")
    assert kept == [{"metadata": {"canonical_video_id": "video2"}, "_index": 0}]
    report = json.loads((tmp_path / "train-clean" / "filter_eval_overlap_report.json").read_text())
    assert report["jsonl"][0]["missing_video_id_rows"] == 1


def test_normalization_dry_run_does_not_create_output(tmp_path):
    raw = tmp_path / "input" / "qa.jsonl"
    write_jsonl(raw, [{"qa_id": "q1", "canonical_video_id": "v1", "question": "Which tool?",
                       "answer": "hammer", "distractors": ["pliers"]}])
    output = tmp_path / "absent"
    cli(MODULE_ROOT + "normalize_to_8choice", "--input-jsonl", raw, "--output-dir", output, "--dry-run")
    assert not output.exists()


def test_builder_refreshes_same_size_copy_and_drops_failed_full_media(tmp_path, monkeypatch):
    src, dst = tmp_path / "source.mp4", tmp_path / "dest.mp4"
    src.write_bytes(b"fresh")
    dst.write_bytes(b"stale")
    build_benchmark._materialize_file(src, dst, link_mode="copy", dry_run=False)
    assert dst.read_bytes() == b"fresh"
    source = tmp_path / "sources" / "source1"
    source.mkdir(parents=True)
    (source / "annotation_manifest.json").write_text(json.dumps({
        "canonical_video_id": "v1", "source_video_path": str(src), "annotations": [],
    }))
    normalized = tmp_path / "normalized"
    write_jsonl(normalized / "normalized.jsonl", [{
        "qa_id": "q1", "annotation_id": "a1", "canonical_video_id": "v1", "source_file": "source1.json",
        "question": "Which tool?", "normalized_options": list("abcdefgh"), "correct_index": 0,
    }])
    def fail_copy(*args, **kwargs):
        raise OSError("synthetic disk write failure")
    monkeypatch.setattr(build_benchmark, "_materialize_file", fail_copy)
    result = build_benchmark.build(normalized_run=normalized, output_root=tmp_path / "out",
                                   version="test", date="20260101", link_mode="copy", dry_run=False,
                                   limit=None, workspace_sources_root=tmp_path / "sources", repo_root=ROOT)
    assert result["counts"]["kept"] == 0
    assert result["counts"]["skipped_missing_video"] == 1


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg required")
def test_temporal_cli_excludes_intervals_and_trims_on_frame_boundaries(tmp_path):
    videos, captions, benchmark = (tmp_path / name for name in ("videos", "captions", "benchmark"))
    for directory in (videos, captions, benchmark):
        directory.mkdir()
    # A long GOP makes a stream-copy trim retain unwanted earlier frames.
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=red:s=64x64:r=20:d=2",
                    "-f", "lavfi", "-i", "color=green:s=64x64:r=20:d=2", "-f", "lavfi", "-i", "color=blue:s=64x64:r=20:d=2",
                    "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]", "-map", "[v]",
                    "-c:v", "libx264", "-g", "200", "-sc_threshold", "0", str(videos / "v1.mp4")],
                   check=True, capture_output=True, timeout=30)
    (captions / "v1.json").write_text(json.dumps({"segments": [
        {"start_frame": 1, "end_frame": 20}, {"start_frame": 51, "end_frame": 70},
    ]}))
    (benchmark / "manifest.tsv").write_text("qa_id\tcanonical_video_id\nq1\tv1\n")
    timestamps = tmp_path / "intervals.jsonl"
    write_jsonl(timestamps, [{"qa_id": "q1", "clip_start_seconds": 2, "clip_end_seconds": 4}])
    training = tmp_path / "train.jsonl"
    write_jsonl(training, [
        {"canonical_video_id": "v1", "start_frame": 81, "end_frame": 100, "videos": ["v1.mp4"]},
        {"canonical_video_id": "v1", "start_frame": 51, "end_frame": 70, "videos": ["v1.mp4"]},
        {"canonical_video_id": "v1", "videos": ["v1.mp4"]},
    ])
    out = tmp_path / "split"
    cli("egotools.data.splits.build_train_temporal_split", "--eval-dir", benchmark,
        "--benchmark-full", timestamps, "--captions-dir", captions, "--videos-dir", videos,
        "--input-jsonl", training, "--output-root", out)
    kept = read_jsonl(out / "train.jsonl")
    assert len(kept) == 1
    assert (kept[0]["start_frame"], kept[0]["end_frame"]) == (1, 20)
    clip = out / kept[0]["videos"][0]
    assert temporal.ffprobe_video(clip)["duration"] == pytest.approx(2.0, abs=0.05)
    pixels = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(clip), "-frames:v", "1",
                                      "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], timeout=15)
    assert sum(pixels[2::3]) / len(pixels[2::3]) > 200
    assert sum(pixels[1::3]) / len(pixels[1::3]) < 20
    assert len(json.loads((out / "captions" / "train" / "v1.json").read_text())["segments"]) == 1


def test_missing_evaluation_timestamps_are_not_silently_retained(tmp_path):
    (tmp_path / "manifest.tsv").write_text("qa_id\tcanonical_video_id\nmissing\tv1\n")
    (tmp_path / "intervals.json").write_text("[]")
    with pytest.raises(ValueError, match="Missing evaluation timestamps"):
        temporal.load_eval_intervals(tmp_path, tmp_path / "intervals.json")


def test_cached_review_must_match_current_option_texts():
    from egotools.data.benchmark.qa_common import QARecord, latest_matching_eval

    record = QARecord(qa_id="q1", source_file="source.json", source_id="source",
                      canonical_video_id="v1", display_name="", annotation_id="a1",
                      question="Which tool?", answer="hammer", distractors=["pliers"],
                      annotator_id=None, raw={}, evals=[{
                          "question": "Which tool?", "answer": "hammer", "num_choices": 2,
                          "evaluated_at": "2026-01-01T00:00:00Z",
                          "options": [{"text": "hammer"}, {"text": "chisel"}],
                      }])
    assert latest_matching_eval(record) is None
    record.evals[0]["options"][1]["text"] = "pliers"
    assert latest_matching_eval(record) == record.evals[0]


def test_distributed_sft_legacy_ids_require_explicit_source_mapping(tmp_path):
    from egotools.data.splits.filter_train_eval_overlap import filter_assets, filter_jsonl

    source = tmp_path / "public-shaped.jsonl"
    rows = [
        {"messages": [{"role": "user", "content": "<video> Which tool?"}],
         "videos": ["videos/train/legacy-source__clip.mp4"], "start_frame": 1, "end_frame": 20,
         "metadata": {"video_id": "legacy-source", "sample_type": "mcq"}},
        {"messages": [], "videos": ["videos/train/other.mp4"],
         "metadata": {"video_id": "legacy-other", "canonical_video_id": "other-source"}},
    ]
    write_jsonl(source, rows)
    report = filter_jsonl(source, tmp_path / "canonical-only.jsonl", {"benchmark-source"})
    assert report["missing_video_id_rows"] == 1
    assert report["kept_rows"] == 1
    assert read_jsonl(tmp_path / "canonical-only.jsonl") == [rows[1]]
    report = filter_jsonl(source, tmp_path / "mapped.jsonl", {"benchmark-source"},
                          source_id_map={"legacy-source": "benchmark-source"})
    assert report["missing_video_id_rows"] == 0
    assert report["overlap_rows"] == 1
    assert report["kept_rows"] == 1

    assets = tmp_path / "assets"
    assets.mkdir()
    for name in ("legacy-source.mp4", "benchmark-source.mp4", "other-source.mp4"):
        (assets / name).write_bytes(b"synthetic-media")
    filter_assets(assets, tmp_path / "retained-assets", {"other-source"}, captions=False)
    assert [path.name for path in (tmp_path / "retained-assets").iterdir()] == ["other-source.mp4"]
