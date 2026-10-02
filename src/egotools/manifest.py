"""Validate an EgoTools benchmark manifest without loading media files."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

OPTION_LETTERS = tuple("ABCDEFGH")
REQUIRED_COLUMNS = (
    "index",
    "video",
    "question",
    *OPTION_LETTERS,
    "answer",
    "qa_id",
    "canonical_video_id",
)


def _read_delimited(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"empty manifest: {path}")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"duplicate manifest columns: {path}")
        rows = list(reader)
        if not rows:
            raise ValueError(f"empty manifest: {path}")
        if any(None in row for row in rows):
            raise ValueError(f"manifest row has more fields than its header: {path}")
        return list(reader.fieldnames), rows


def _read_jsonl(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    columns: list[str] = []
    seen_columns: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            for key in value:
                if key not in seen_columns:
                    seen_columns.add(key)
                    columns.append(key)
            rows.append(value)
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return columns, rows


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Read TSV/CSV/JSONL and normalize media paths to stripped POSIX paths."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        columns, rows = _read_jsonl(path)
    elif path.suffix.lower() in {".tsv", ".csv"}:
        columns, rows = _read_delimited(path)
    else:
        raise ValueError(f"unsupported manifest format: {path.suffix}")
    for row in rows:
        for column in ("video", "clip_video"):
            if isinstance(row.get(column), str):
                row[column] = row[column].strip().replace("\\", "/")
    return columns, rows


def _relative_asset_path(value: Any) -> bool:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        return False
    path = PurePosixPath(text)
    return bool(path.parts) and not path.is_absolute() and not PureWindowsPath(text).drive and ".." not in path.parts and "://" not in text


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _first_duplicate(values: Iterable[Any]) -> Any | None:
    seen = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def validate_manifest(
    path: Path,
    *,
    expected_rows: int | None = None,
    asset_root: Path | None = None,
    asset_mode: str = "none",
) -> dict[str, Any]:
    """Validate schema, row identity, answers, options, and optional assets."""

    if asset_mode not in {"none", "full", "clip", "both"}:
        raise ValueError("asset_mode must be one of: none, full, clip, both")

    columns, rows = read_manifest(path)
    errors: list[str] = []
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing_columns:
        errors.append("missing columns: " + ", ".join(missing_columns))

    if expected_rows is not None and len(rows) != expected_rows:
        errors.append(f"expected {expected_rows} rows, found {len(rows)}")

    indices = [_text(row.get("index")) for row in rows]
    # pandas infers an all-numeric TSV column, so e.g. 01 and 1 share an index.
    for parse in (int, float):
        try:
            indices = [parse(value) for value in indices]
            break
        except ValueError:
            pass
    duplicate_index = _first_duplicate(indices)
    if duplicate_index is not None:
        errors.append(f"duplicate index: {duplicate_index!r}; use egotools-prepare-benchmark --reindex")
    duplicate_qa = _first_duplicate(_text(row.get("qa_id")) for row in rows)
    if duplicate_qa is not None:
        errors.append(f"duplicate qa_id: {duplicate_qa!r}")

    answer_counts = {letter: 0 for letter in OPTION_LETTERS}
    missing_assets: list[str] = []
    checked_assets = 0
    root = asset_root.expanduser().resolve() if asset_root is not None else None
    selected_asset_columns = {
        "none": (),
        "full": ("video",),
        "clip": ("clip_video",),
        "both": ("video", "clip_video"),
    }[asset_mode]

    for row_number, row in enumerate(rows, 1):
        qa_id = _text(row.get("qa_id"))
        label = qa_id or f"row {row_number}"
        if not qa_id:
            errors.append(f"row {row_number}: empty qa_id")
        for column in ("index", "canonical_video_id"):
            if not _text(row.get(column)):
                errors.append(f"{label}: empty {column}")
        if not _text(row.get("question")):
            errors.append(f"{label}: empty question")
        options = [letter for letter in OPTION_LETTERS if _text(row.get(letter))]
        if len(options) < 2:
            errors.append(f"{label}: at least two non-empty options are required")
        answer = _text(row.get("answer")).upper()
        if answer not in OPTION_LETTERS:
            errors.append(f"{label}: invalid answer {answer!r}")
        else:
            answer_counts[answer] += 1
            if answer not in options:
                errors.append(f"{label}: gold answer {answer} references an empty option")

        video = row.get("video", "")
        if not _relative_asset_path(video):
            errors.append(f"{label}: video must be a non-empty relative path")
        clip = row.get("clip_video", "")
        if str(clip or "").strip() and not _relative_asset_path(clip):
            errors.append(f"{label}: clip_video must be a relative path when present")

        if selected_asset_columns:
            if root is None:
                errors.append("asset_root is required when asset_mode is not 'none'")
                selected_asset_columns = ()
                continue
            for column in selected_asset_columns:
                relative = str(row.get(column, "") or "").strip()
                if not relative and column == "clip_video":
                    relative = _text(row.get("video"))
                if not relative:
                    missing_assets.append(f"{label}:{column}:<empty>")
                    continue
                checked_assets += 1
                if not (root / relative).is_file():
                    missing_assets.append(relative)

        if len(errors) >= 100:
            break

    if missing_assets:
        errors.append(
            f"missing {len(missing_assets)} checked asset(s); first entries: "
            + ", ".join(missing_assets[:10])
        )

    report = {
        "manifest": str(path.expanduser().resolve()),
        "rows": len(rows),
        "columns": columns,
        "answer_counts": answer_counts,
        "asset_mode": asset_mode,
        "checked_assets": checked_assets,
        "valid": not errors,
        "errors": errors,
    }
    return report


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--asset-mode", choices=("none", "full", "clip", "both"), default="none")
    parser.add_argument("--json-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    report = validate_manifest(
        args.manifest,
        expected_rows=args.expected_rows,
        asset_root=args.asset_root,
        asset_mode=args.asset_mode,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
