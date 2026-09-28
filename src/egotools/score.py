"""Score EgoTools A-H predictions with deterministic exact-match parsing."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .answers import extract_letter
from .manifest import OPTION_LETTERS, read_manifest


def _identity_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _read_predictions(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    columns, rows = read_manifest(path)
    if "prediction" not in columns:
        raise ValueError("predictions file is missing the 'prediction' column")
    return columns, rows


def _manifest_indexes(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_qa: dict[str, dict[str, Any]] = {}
    by_index: dict[str, dict[str, Any]] = {}
    for row in rows:
        qa_id = _identity_text(row.get("qa_id"))
        index = _identity_text(row.get("index"))
        if qa_id:
            if qa_id in by_qa:
                raise ValueError(f"manifest contains duplicate qa_id: {qa_id}")
            by_qa[qa_id] = row
        if index:
            if index in by_index:
                raise ValueError(f"manifest contains duplicate index: {index}")
            by_index[index] = row
    return by_qa, by_index


def _join_manifest(
    predictions: list[dict[str, Any]], manifest_rows: list[dict[str, Any]], *, allow_partial: bool = False
) -> list[dict[str, Any]]:
    by_qa, by_index = _manifest_indexes(manifest_rows)
    joined: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_number, prediction in enumerate(predictions, 1):
        qa_id = _identity_text(prediction.get("qa_id"))
        index = _identity_text(prediction.get("index"))
        source = by_qa.get(qa_id) if qa_id else by_index.get(index)
        if source is None:
            raise ValueError(f"prediction row {row_number} does not match the manifest")
        if qa_id and index and by_index.get(index) is not source:
            raise ValueError(f"prediction row {row_number} has conflicting qa_id and index")
        identity = str(source.get("qa_id") or source.get("index"))
        if identity in seen:
            raise ValueError(f"duplicate prediction for {identity}")
        seen.add(identity)
        merged = dict(source)
        merged["prediction"] = prediction["prediction"]
        joined.append(merged)
    if len(joined) != len(manifest_rows) and not allow_partial:
        raise ValueError(
            f"found predictions for {len(joined)} of {len(manifest_rows)} manifest rows; "
            "use --allow-partial to explicitly score an incomplete run"
        )
    return joined


def _metric(correct: int, total: int) -> dict[str, Any]:
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }


def _group_metrics(rows: list[dict[str, Any]], column: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        value = str(row.get(column, "") or "_unknown").strip() or "_unknown"
        groups[value].append(int(row["correct"]))
    return {name: _metric(sum(values), len(values)) for name, values in sorted(groups.items())}


def score_predictions(
    predictions_path: Path,
    *,
    manifest_path: Path | None = None,
    allow_partial: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score predictions and return metrics plus annotated rows."""

    _, prediction_rows = _read_predictions(predictions_path)
    if manifest_path is not None:
        _, manifest_rows = read_manifest(manifest_path)
        rows = _join_manifest(prediction_rows, manifest_rows, allow_partial=allow_partial)
    else:
        rows = prediction_rows
        _manifest_indexes(rows)
        if any("answer" not in row for row in rows):
            raise ValueError("--manifest is required when predictions do not contain answers")

    for row in rows:
        predicted_letter = extract_letter(row.get("prediction"), options=row)
        answer = str(row.get("answer", "")).strip().upper()
        if answer not in OPTION_LETTERS:
            raise ValueError(f"invalid gold answer for qa_id={row.get('qa_id')!r}: {answer!r}")
        row["predicted_letter"] = predicted_letter
        row["is_valid"] = int(bool(predicted_letter))
        row["correct"] = int(predicted_letter == answer)

    total = len(rows)
    correct = sum(int(row["correct"]) for row in rows)
    valid = sum(int(row["is_valid"]) for row in rows)
    metrics: dict[str, Any] = {
        "overall": _metric(correct, total),
        "valid_predictions": {
            "valid": valid,
            "invalid": total - valid,
            "total": total,
            "rate": valid / total if total else 0.0,
        },
    }
    if manifest_path is not None:
        metrics["coverage"] = {
            "predicted": total,
            "expected": len(manifest_rows),
            "complete": total == len(manifest_rows),
        }

    for column, output_name in (
        ("research_track_id", "per_track"),
        ("research_track", "per_track"),
        ("track", "per_track"),
    ):
        if any(str(row.get(column, "")).strip() for row in rows):
            metrics[output_name] = _group_metrics(rows, column)
            break
    if any(str(row.get("qtype", "")).strip() for row in rows):
        metrics["per_qtype"] = _group_metrics(rows, "qtype")

    return metrics, rows


def _write_scored_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "index",
        "qa_id",
        "qtype",
        "research_track_id",
        "answer",
        "prediction",
        "predicted_letter",
        "is_valid",
        "correct",
    ]
    present = {key for row in rows for key in row}
    fieldnames = [key for key in preferred if key in present]
    fieldnames.extend(sorted(present.difference(fieldnames)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="Score incomplete predictions and report coverage.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/score"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    metrics, rows = score_predictions(args.predictions, manifest_path=args.manifest, allow_partial=args.allow_partial)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    scored_path = args.output_dir / "predictions_scored.tsv"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_scored_tsv(scored_path, rows)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"wrote {metrics_path}")
    print(f"wrote {scored_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
