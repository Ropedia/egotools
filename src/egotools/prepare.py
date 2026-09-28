"""Copy a benchmark manifest to TSV with unique sequential row indices.

The historical 902-question bundle contains repeated integer indices. Question
IDs are the durable identity; reindexing preserves them and all question data.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .manifest import read_manifest


def prepare_manifest(source: Path, output: Path) -> int:
    columns, rows = read_manifest(source)
    seen: set[str] = set()
    for row_number, row in enumerate(rows):
        qa_id = str(row.get("qa_id") or "").strip()
        if not qa_id or qa_id in seen:
            raise ValueError(f"row {row_number + 1}: missing or duplicate qa_id {qa_id!r}")
        seen.add(qa_id)
        row["index"] = row_number
    if "index" not in columns:
        columns.insert(0, "index")
    if source.expanduser().resolve() == output.expanduser().resolve():
        raise ValueError("choose a separate output path to retain the original manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Destination TSV; media paths are preserved.")
    args = parser.parse_args(argv)
    count = prepare_manifest(args.manifest, args.output)
    print(f"Wrote {count} rows to {args.output}; qa_id values and media paths preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
