#!/usr/bin/env python3
"""Extract final reviewed benchmark QA records from visualizer edits.

Review edits are read from an append-only JSONL file supplied with --edits.
This script takes the newest edit per qa_id, overlays it on the benchmark
release manifest, and writes one final record per reviewed QA.

Use --keep-unreviewed-normalized to also include manifest QAs that never
received a manual edit; these remain review_status=normalized and are tagged
as unreviewed normalized records.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"Expected object in {path}:{line_no}")
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def latest_edits_by_qa(edits: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edit in edits:
        qid = str(edit.get("qa_id") or edit.get("annotation_id") or "").strip()
        if qid:
            grouped[qid].append(edit)

    def sort_key(edit: dict[str, Any]) -> tuple[str, int]:
        return (str(edit.get("saved_at_utc") or ""), int(edit.get("_line_no") or 0))

    return {qid: sorted(items, key=sort_key)[-1] for qid, items in grouped.items()}


def normalize_review_status(raw: Any) -> str:
    status = str(raw or "").strip().lower()
    if status == "dropped":
        return "drop"
    if status in {"clean", "drop", "pending"}:
        return status
    return "pending"


def clean_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def apply_final_edit(
    base: dict[str, Any],
    edit: dict[str, Any],
    package_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    qid = str(base.get("qa_id") or base.get("annotation_id") or edit.get("qa_id") or "").strip()

    options = clean_text_list(edit.get("edited_options")) or clean_text_list(base.get("normalized_options"))
    correct_index = edit.get("edited_correct_index")
    if not isinstance(correct_index, int):
        correct_index = base.get("correct_index")

    if not isinstance(correct_index, int) or correct_index < 0 or correct_index >= len(options):
        warnings.append("invalid_correct_index")
        correct_index = None

    edited_answer = str(edit.get("edited_answer") or "").strip() if edit.get("edited_answer") is not None else ""
    if correct_index is not None and options:
        answer = options[correct_index]
        if edited_answer and edited_answer != answer:
            warnings.append("edited_answer_mismatch_selected_option")
    else:
        answer = edited_answer or str(base.get("answer") or "").strip()

    question = (
        str(edit.get("edited_question") or "").strip()
        if edit.get("edited_question") is not None
        else str(base.get("question") or "").strip()
    )
    review_status = normalize_review_status(edit.get("review_status"))

    video_rel = str(base.get("_video_rel") or "").strip()
    clip_rel = str(base.get("_clip_rel") or "").strip()
    video_path = package_dir / video_rel if video_rel else None
    clip_path = package_dir / clip_rel if clip_rel else None

    out = {
        "qa_id": qid,
        "annotation_id": str(base.get("annotation_id") or qid),
        "source_file": base.get("source_file") or edit.get("source_file") or "",
        "source_id": base.get("source_id") or "",
        "canonical_video_id": base.get("canonical_video_id") or "",
        "qtype": base.get("qtype") or "",
        "question": question,
        "answer": answer,
        "normalized_options": options,
        "correct_index": correct_index,
        "review_status": review_status,
        "manual_reviewed": True,
        "is_unreviewed_normalized": False,
        "keep_unreviewed_normalized": False,
        "keep_for_benchmark": review_status != "drop",
        "is_dropped": review_status == "drop",
        "drop_reason": str(edit.get("drop_reason") or "").strip(),
        "latest_edit_saved_at_utc": edit.get("saved_at_utc") or "",
        "latest_edit_line_no": edit.get("_line_no"),
        "benchmark_version": edit.get("benchmark_version") or "",
        "video_rel": video_rel,
        "clip_rel": clip_rel,
        "video_exists": bool(video_path and video_path.is_file()),
        "clip_exists": bool(clip_path and clip_path.is_file()),
        "original_manifest_index": base.get("_index"),
        "original_question": base.get("question") or "",
        "original_answer": base.get("answer") or "",
        "original_normalized_options": base.get("normalized_options") or [],
        "original_correct_index": base.get("correct_index"),
        "edit_warning_flags": warnings,
    }
    return out, warnings


def build_unreviewed_normalized_record(
    base: dict[str, Any],
    package_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    qid = str(base.get("qa_id") or base.get("annotation_id") or "").strip()
    options = clean_text_list(base.get("normalized_options"))
    correct_index = base.get("correct_index")
    if not isinstance(correct_index, int) or correct_index < 0 or correct_index >= len(options):
        warnings.append("invalid_correct_index")
        correct_index = None

    answer = options[correct_index] if correct_index is not None and options else str(base.get("answer") or "").strip()
    video_rel = str(base.get("_video_rel") or "").strip()
    clip_rel = str(base.get("_clip_rel") or "").strip()
    video_path = package_dir / video_rel if video_rel else None
    clip_path = package_dir / clip_rel if clip_rel else None

    out = {
        "qa_id": qid,
        "annotation_id": str(base.get("annotation_id") or qid),
        "source_file": base.get("source_file") or "",
        "source_id": base.get("source_id") or "",
        "canonical_video_id": base.get("canonical_video_id") or "",
        "qtype": base.get("qtype") or "",
        "question": str(base.get("question") or "").strip(),
        "answer": answer,
        "normalized_options": options,
        "correct_index": correct_index,
        "review_status": "normalized",
        "manual_reviewed": False,
        "is_unreviewed_normalized": True,
        "keep_unreviewed_normalized": True,
        "keep_for_benchmark": True,
        "is_dropped": False,
        "drop_reason": "",
        "latest_edit_saved_at_utc": "",
        "latest_edit_line_no": None,
        "benchmark_version": "",
        "video_rel": video_rel,
        "clip_rel": clip_rel,
        "video_exists": bool(video_path and video_path.is_file()),
        "clip_exists": bool(clip_path and clip_path.is_file()),
        "original_manifest_index": base.get("_index"),
        "original_question": base.get("question") or "",
        "original_answer": base.get("answer") or "",
        "original_normalized_options": base.get("normalized_options") or [],
        "original_correct_index": base.get("correct_index"),
        "edit_warning_flags": warnings,
    }
    return out, warnings


def validate_records(
    records: list[dict[str, Any]],
    manifest_by_qa: dict[str, dict[str, Any]],
    *,
    require_clip: bool,
) -> dict[str, Any]:
    qa_counts = Counter(r["qa_id"] for r in records)
    video_pairs = Counter((r.get("qa_id"), r.get("video_rel"), r.get("clip_rel")) for r in records)
    video_rel_counts = Counter(r.get("video_rel") for r in records)

    invalid_options = []
    missing_video = []
    missing_clip = []
    empty_clip_rel = []
    missing_manifest = []
    answer_mismatch = []
    for rec in records:
        qid = rec["qa_id"]
        opts = rec.get("normalized_options") or []
        ci = rec.get("correct_index")
        if qid not in manifest_by_qa:
            missing_manifest.append(qid)
        if len(opts) != 8 or not isinstance(ci, int) or ci < 0 or ci >= len(opts):
            invalid_options.append(qid)
        elif str(opts[ci]).strip() != str(rec.get("answer") or "").strip():
            answer_mismatch.append(qid)
        if not rec.get("video_exists"):
            missing_video.append(qid)
        if rec.get("clip_rel") and not rec.get("clip_exists"):
            missing_clip.append(qid)
        elif not rec.get("clip_rel"):
            empty_clip_rel.append(qid)
            if require_clip:
                missing_clip.append(qid)

    duplicated_qa_ids = sorted(qid for qid, count in qa_counts.items() if count != 1)
    duplicated_mapping_rows = sorted(
        [qid for (qid, _video, _clip), count in video_pairs.items() if count != 1]
    )
    return {
        "record_count": len(records),
        "status_counts": dict(Counter(r.get("review_status") for r in records)),
        "manual_review_counts": dict(Counter(bool(r.get("manual_reviewed")) for r in records)),
        "unreviewed_normalized_count": sum(1 for r in records if r.get("is_unreviewed_normalized")),
        "keep_for_benchmark_count": sum(1 for r in records if r.get("keep_for_benchmark")),
        "dropped_count": sum(1 for r in records if r.get("is_dropped")),
        "clean_count": sum(1 for r in records if r.get("review_status") == "clean"),
        "unique_qa_ids": len(qa_counts),
        "duplicated_qa_ids": duplicated_qa_ids,
        "missing_manifest_qa_ids": sorted(missing_manifest),
        "invalid_option_qa_ids": sorted(invalid_options),
        "answer_mismatch_qa_ids": sorted(answer_mismatch),
        "missing_video_qa_ids": sorted(missing_video),
        "missing_clip_qa_ids": sorted(missing_clip),
        "empty_clip_rel_qa_ids": sorted(empty_clip_rel),
        "non_unique_qa_video_clip_rows": duplicated_mapping_rows,
        "unique_video_count": len([v for v in video_rel_counts if v]),
        "max_qas_per_video": max(video_rel_counts.values(), default=0),
        "video_rel_counts_top10": video_rel_counts.most_common(10),
        "passed": not (
            duplicated_qa_ids
            or missing_manifest
            or invalid_options
            or answer_mismatch
            or missing_video
            or (require_clip and missing_clip)
            or duplicated_mapping_rows
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--edits", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--require-clip",
        action="store_true",
        help="Fail validation if a reviewed QA has no packaged QA clip. Full video is always required.",
    )
    parser.add_argument(
        "--keep-unreviewed-normalized",
        action="store_true",
        help="Also output manifest QAs with no manual edit, tagged as review_status=normalized.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_rows = read_jsonl(args.manifest)
    edit_rows = read_jsonl(args.edits)

    manifest_by_qa = {
        str(row.get("qa_id") or row.get("annotation_id") or "").strip(): row
        for row in manifest_rows
        if str(row.get("qa_id") or row.get("annotation_id") or "").strip()
    }
    latest_edits = latest_edits_by_qa(edit_rows)

    records: list[dict[str, Any]] = []
    all_warnings: dict[str, list[str]] = {}
    for qid, edit in sorted(latest_edits.items(), key=lambda kv: int(kv[1].get("_line_no") or 0)):
        base = manifest_by_qa.get(qid)
        if base is None:
            base = {"qa_id": qid, "annotation_id": qid}
        record, warnings = apply_final_edit(base, edit, args.package_dir)
        records.append(record)
        if warnings:
            all_warnings[qid] = warnings

    if args.keep_unreviewed_normalized:
        for qid, base in manifest_by_qa.items():
            if qid in latest_edits:
                continue
            record, warnings = build_unreviewed_normalized_record(base, args.package_dir)
            records.append(record)
            if warnings:
                all_warnings[qid] = warnings

    records.sort(key=lambda r: (r.get("original_manifest_index") is None, r.get("original_manifest_index") or 0, r["qa_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "edits": str(args.edits),
        "output": str(args.output),
        "edit_rows": len(edit_rows),
        "unique_reviewed_qa_ids": len(latest_edits),
        "keep_unreviewed_normalized": bool(args.keep_unreviewed_normalized),
        "unreviewed_normalized_added": max(0, len(records) - len(latest_edits)),
        "manifest_rows": len(manifest_rows),
        "warning_flags_by_qa": all_warnings,
        "validation": validate_records(records, manifest_by_qa, require_clip=args.require_clip),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")

    validation = report["validation"]
    print(f"Wrote {len(records)} final reviewed QA records to {args.output}")
    print(f"Wrote validation report to {args.report}")
    print(f"Status counts: {validation['status_counts']}")
    print(f"QA-video validation passed: {validation['passed']}")
    if not validation["passed"]:
        print("Validation failures:")
        for key in (
            "duplicated_qa_ids",
            "missing_manifest_qa_ids",
            "invalid_option_qa_ids",
            "answer_mismatch_qa_ids",
            "missing_video_qa_ids",
            "missing_clip_qa_ids",
            "non_unique_qa_video_clip_rows",
        ):
            values = validation[key]
            if values:
                print(f"  {key}: {len(values)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
