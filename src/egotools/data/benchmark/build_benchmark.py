#!/usr/bin/env python
"""Build a portable benchmark from normalized QA rows and source manifests.

Writes relative videos/<canonical_video_id>.mp4 and clips/<qa_id>.mp4 paths,
manifest.tsv and local build reports. --export-review-records also writes
review_records.jsonl for the optional manual-review workflow.
Use --link-mode copy for a self-contained package. Hardlinks require the same
filesystem; symlink outputs depend on source media remaining available.

--dry-run reports the media estimate without writing. The existing 600 GiB
copy budget is retained to prevent an accidental large media materialization;
--max-copy-gib makes that limit explicit and adjustable for the target disk.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from egotools.data.benchmark._resolvers import VideoRefs, resolve_video_refs

# --------------------------------------------------------------------------- #
# Columns consumed by evaluation and track reporting
# --------------------------------------------------------------------------- #

TSV_COLUMNS = [
    "index",
    "video",
    "question",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "answer",
    "qtype",
    "qa_id",
    "canonical_video_id",
    "clip_video",
    "research_track_id",
    "research_subtrack_id",
    "tool_capability_axis",
]
OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]

# Retain the existing copy-size safeguard; the CLI exposes the disk budget.
DISK_BUDGET_BYTES = 600 * 1024**3


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #


def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _read_jsonl(p: Path) -> Iterable[dict]:
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _materialize_file(
    src: Path, dst: Path, *, link_mode: str, dry_run: bool
) -> tuple[bool, int]:
    """Copy/link ``src`` → ``dst`` according to ``link_mode``.

    Returns ``(did_work, bytes_transferred)``. ``did_work`` is False when the
    destination already refers to the source file or
    when ``dry_run`` is set.
    """
    if not src.exists():
        raise FileNotFoundError(src)
    src_size = src.stat().st_size

    if dst.exists() or dst.is_symlink():
        # Same-sized files may have different content. Native file identity
        # can skip existing links; ordinary copies are refreshed on a re-run.
        if dst.exists() and src.samefile(dst):
            return False, 0
        if not dry_run:
            dst.unlink()

    if dry_run:
        return False, src_size

    dst.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "copy":
        shutil.copy2(src, dst)
    elif link_mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
    elif link_mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"unknown link_mode: {link_mode}")
    return True, src_size


# --------------------------------------------------------------------------- #
# Row pipeline
# --------------------------------------------------------------------------- #


def _build_tsv_row(idx: int, row: dict, *, video_rel: str, clip_rel: str) -> dict:
    options = list(row.get("normalized_options") or [])
    correct = int(row.get("correct_index", 0))
    answer_letter = chr(ord("A") + correct) if 0 <= correct < len(options) else ""
    out = {c: "" for c in TSV_COLUMNS}
    out["index"] = idx
    out["video"] = video_rel
    out["question"] = row.get("question", "")
    for i, letter in enumerate(OPTION_LETTERS):
        out[letter] = options[i] if i < len(options) else ""
    out["answer"] = answer_letter
    out["qtype"] = row.get("qtype", "")
    out["qa_id"] = row.get("qa_id", "")
    out["canonical_video_id"] = row.get("canonical_video_id", "")
    out["clip_video"] = clip_rel
    for column in ("research_track_id", "research_subtrack_id", "tool_capability_axis"):
        out[column] = row.get(column, "")
    return out


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #


def build(
    *,
    normalized_run: Path,
    output_root: Path,
    version: str,
    date: str,
    link_mode: str,
    dry_run: bool,
    limit: int | None,
    workspace_sources_root: Path,
    repo_root: Path,
    disk_budget_bytes: int = DISK_BUDGET_BYTES,
    normalized_jsonl: Path | None = None,
    export_review_records: bool = False,
) -> dict:
    """Run the build end-to-end and return the build report."""
    started_at = _utcnow_iso()

    norm_jsonl = normalized_jsonl or normalized_run / "normalized.jsonl"
    if not norm_jsonl.exists():
        raise FileNotFoundError(f"missing normalized.jsonl under {normalized_run}")

    # Provenance — capture upstream git_sha when available.
    normalization_report = normalized_run / "normalization_report.json"
    source_git_sha = ""
    if normalization_report.exists():
        try:
            with normalization_report.open() as f:
                source_git_sha = json.load(f).get("git_sha", "")
        except (json.JSONDecodeError, OSError):
            source_git_sha = ""

    for label, value in (("version", version), ("date", date)):
        if not value or value in {".", ".."} or any(c in value for c in "/\\"):
            raise ValueError(f"{label} must be a single filename component")
    out_dir = output_root / f"{version}_{date}"
    videos_dir = out_dir / "videos"
    clips_dir = out_dir / "clips"
    rows = list(_read_jsonl(norm_jsonl))
    if limit is not None:
        rows = rows[:limit]
    n_input = len(rows)
    rows = [row for row in rows if not row.get("is_dropped") and row.get("keep_for_benchmark", True)]
    # IDs become output filenames, so validate them before creating assets.
    seen_ids: set[str] = set()
    for row in rows:
        for field in ("qa_id", "canonical_video_id"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip() or value in {".", ".."} or any(c in value for c in "/\\"):
                raise ValueError(f"{field} must be a nonempty filename component")
        if row["qa_id"] in seen_ids:
            raise ValueError(f"Duplicate qa_id: {row['qa_id']}")
        seen_ids.add(row["qa_id"])
        options, correct = row.get("normalized_options"), row.get("correct_index")
        if (not isinstance(options, list) or len(options) != 8
                or any(not isinstance(option, str) or not option.strip() for option in options)
                or not isinstance(correct, int) or isinstance(correct, bool) or not 0 <= correct < 8
                or not str(row.get("question") or "").strip()):
            raise ValueError(f"Invalid eight-choice question: {row['qa_id']}")

    # ---- Resolve all rows up front so we can run a pre-flight size estimate.
    resolved: list[tuple[dict, VideoRefs]] = []
    for r in rows:
        refs = resolve_video_refs(
            workspace_sources_root=workspace_sources_root,
            source_file=r.get("source_file", ""),
            annotation_id=r.get("annotation_id", ""),
            canonical_video_id_from_row=r.get("canonical_video_id", ""),
        )
        resolved.append((r, refs))

    # Bytes estimate — unique videos + per-row clips (only those resolved).
    unique_videos: dict[str, Path] = {}
    clip_paths: list[Path] = []
    for _, refs in resolved:
        if refs.source_video_path is not None:
            previous = unique_videos.setdefault(refs.canonical_video_id, refs.source_video_path)
            if previous.resolve() != refs.source_video_path.resolve():
                raise ValueError(f"Multiple media paths for canonical_video_id {refs.canonical_video_id}")
        if refs.clip_path is not None:
            clip_paths.append(refs.clip_path)
    est_video_bytes = sum(p.stat().st_size for p in unique_videos.values())
    est_clip_bytes = sum(p.stat().st_size for p in clip_paths)
    est_total = est_video_bytes + est_clip_bytes

    print(
        f"[estimate] unique_videos={len(unique_videos)} "
        f"video_bytes={est_video_bytes:,} ({est_video_bytes/1e9:.2f} GB) | "
        f"clips={len(clip_paths)} clip_bytes={est_clip_bytes:,} "
        f"({est_clip_bytes/1e9:.2f} GB) | total={est_total:,} "
        f"({est_total/1e9:.2f} GB)",
        flush=True,
    )

    if est_total > disk_budget_bytes and not dry_run:
        raise SystemExit(
            f"ABORT: estimated copy size {est_total/1e9:.1f} GB exceeds budget "
            f"{disk_budget_bytes/1e9:.0f} GB. Re-run with --dry-run to inspect, "
            "or set --max-copy-gib for the available storage."
        )

    if dry_run:
        print("[dry-run] no files will be written", flush=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=False)

    # ---- Per-canonical_video_id copy pass (deduped).
    counts = {
        "input": n_input,
        "kept": 0,
        "skipped_missing_id": 0,
        "skipped_dropped": n_input - len(rows),
        "skipped_missing_video": 0,
        "skipped_missing_clip": 0,
    }
    skipped_rows: list[dict] = []
    tsv_rows: list[dict] = []
    review_records: list[dict] = []
    bytes_copied = 0

    # video copy: dedupe by canonical_video_id
    copied_videos: set[str] = set()
    for _, refs in resolved:
        if refs.source_video_path is None:
            continue
        cid = refs.canonical_video_id
        if cid in copied_videos:
            continue
        dst = videos_dir / f"{cid}.mp4"
        try:
            _, n_bytes = _materialize_file(
                refs.source_video_path, dst,
                link_mode=link_mode, dry_run=dry_run,
            )
            if not dry_run:
                bytes_copied += n_bytes
            copied_videos.add(cid)
        except OSError as e:
            # Materialization itself failed — surface in skipped, drop the
            # canonical so all rows referencing it get marked.
            print(f"[warn] failed to copy {refs.source_video_path} → {dst}: {e}",
                  file=sys.stderr, flush=True)

    # ---- Per-row pass — clips + manifest emission.
    next_idx = 0
    for r, refs in resolved:
        qa_id = r.get("qa_id", "")
        cid = refs.canonical_video_id

        if refs.reason == "empty_canonical_video_id":
            counts["skipped_missing_id"] += 1
            skipped_rows.append({
                "qa_id": qa_id,
                "source_file": r.get("source_file", ""),
                "annotation_id": r.get("annotation_id", ""),
                "canonical_video_id": "",
                "reason": "empty_canonical_video_id",
            })
            continue

        if refs.source_video_path is None or cid not in copied_videos:
            counts["skipped_missing_video"] += 1
            skipped_rows.append({
                "qa_id": qa_id,
                "source_file": r.get("source_file", ""),
                "annotation_id": r.get("annotation_id", ""),
                "canonical_video_id": cid,
                "reason": refs.reason or "source_video_materialization_failed",
            })
            continue

        # Row is kept. Try the clip; missing clip is non-fatal.
        clip_rel = ""
        if refs.clip_path is not None:
            clip_dst = clips_dir / f"{qa_id}.mp4"
            try:
                _, n_bytes = _materialize_file(
                    refs.clip_path, clip_dst,
                    link_mode=link_mode, dry_run=dry_run,
                )
                if not dry_run:
                    bytes_copied += n_bytes
                clip_rel = f"clips/{qa_id}.mp4"
            except OSError as e:
                counts["skipped_missing_clip"] += 1
                skipped_rows.append({
                    "qa_id": qa_id,
                    "source_file": r.get("source_file", ""),
                    "annotation_id": r.get("annotation_id", ""),
                    "canonical_video_id": cid,
                    "reason": f"clip_copy_failed:{e}",
                })
        else:
            counts["skipped_missing_clip"] += 1
            skipped_rows.append({
                "qa_id": qa_id,
                "source_file": r.get("source_file", ""),
                "annotation_id": r.get("annotation_id", ""),
                "canonical_video_id": cid,
                "reason": refs.reason or "clip_unresolved",
            })

        video_rel = f"videos/{cid}.mp4"
        tsv_rows.append(
            _build_tsv_row(next_idx, r, video_rel=video_rel, clip_rel=clip_rel)
        )
        if export_review_records:
            # Keep identities, original snapshots, and raw annotations private.
            public_fields = (
                "qa_id", "annotation_id", "source_file", "source_id", "canonical_video_id",
                "question", "answer", "normalized_options", "correct_index", "qtype",
                "research_track_id", "research_track_name", "research_subtrack_id",
                "research_subtrack_name", "tool_capability_axis", "review_status",
                "clip_start_seconds", "clip_end_seconds",
            )
            review_records.append({key: r[key] for key in public_fields if key in r} | {
                "_index": next_idx, "_video_rel": video_rel, "_clip_rel": clip_rel,
            })
        counts["kept"] += 1
        next_idx += 1

    # ---- Emit outputs.
    manifest_tsv = out_dir / "manifest.tsv"
    skipped_csv = out_dir / "skipped_build.csv"

    if not dry_run:
        with manifest_tsv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=TSV_COLUMNS, delimiter="\t")
            writer.writeheader()
            writer.writerows(tsv_rows)

        if export_review_records:
            with (out_dir / "review_records.jsonl").open("w", encoding="utf-8") as f:
                for rec in review_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        with skipped_csv.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "qa_id",
                    "source_file",
                    "annotation_id",
                    "canonical_video_id",
                    "reason",
                ],
            )
            w.writeheader()
            for r in skipped_rows:
                w.writerow(r)

    finished_at = _utcnow_iso()
    build_report = {
        "input_run_dir": str(normalized_run),
        "output_dir": str(out_dir),
        "version": version,
        "date": date,
        "source_git_sha": source_git_sha,
        "built_git_sha": _git_sha(repo_root),
        "counts": counts,
        "link_mode": link_mode,
        "dry_run": dry_run,
        "limit": limit,
        "estimated_bytes": est_total,
        "bytes_copied": bytes_copied,
        "build_started_at_utc": started_at,
        "build_finished_at_utc": finished_at,
    }
    if not dry_run:
        with (out_dir / "build_report.json").open("w") as f:
            json.dump(build_report, f, indent=2, sort_keys=True)

    print("[counts]", json.dumps(counts), flush=True)
    print(
        f"[done] kept={counts['kept']} bytes_copied={bytes_copied:,} "
        f"({bytes_copied/1e9:.2f} GB)",
        flush=True,
    )
    return build_report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a versioned egotools benchmark snapshot from a "
        "normalized run directory.",
    )
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--normalized-run", type=Path,
                        help="Run directory containing normalized.jsonl")
    inputs.add_argument("--input-jsonl", type=Path,
                        help="Normalized, reviewed, or track-assigned QA JSONL")
    p.add_argument(
        "--output-root", type=Path, required=True,
        help="Root dir under which <version>_<date>/ is created.",
    )
    p.add_argument("--version", required=True, help="e.g. v1")
    p.add_argument("--date", required=True, help="Build date in YYYYMMDD format")
    p.add_argument("--export-review-records", action="store_true",
                   help="Also write review_records.jsonl for manual review; not needed for evaluation.")
    p.add_argument(
        "--link-mode", choices=("copy", "symlink", "hardlink"), default="copy",
        help="How to materialize videos/clips (default: copy → self-contained).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Estimate copy size, log resolution outcomes, write nothing.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N normalized rows (development aid).",
    )
    p.add_argument(
        "--workspace-sources-root",
        type=Path,
        required=True,
        help="Root containing one subdir per source_id with annotation_manifest.json.",
    )
    p.add_argument(
        "--repo-root", type=Path, default=Path("."),
        help="Repo root for git_sha capture (default: cwd).",
    )
    p.add_argument("--max-copy-gib", type=float, default=600,
                   help="Maximum estimated package size in GiB (default: 600)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    build(
        normalized_run=(args.normalized_run or args.input_jsonl.parent).resolve(),
        normalized_jsonl=args.input_jsonl.resolve() if args.input_jsonl else None,
        output_root=args.output_root.resolve(),
        version=args.version,
        date=args.date,
        link_mode=args.link_mode,
        dry_run=args.dry_run,
        limit=args.limit,
        workspace_sources_root=args.workspace_sources_root.resolve(),
        repo_root=args.repo_root.resolve(),
        disk_budget_bytes=int(args.max_copy_gib * 1024**3),
        export_review_records=args.export_review_records,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
