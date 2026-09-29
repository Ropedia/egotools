#!/usr/bin/env python3
"""Remove provenance columns before ms-swift loads an EgoTools SFT JSONL.

The public staging file mixes different nested metadata schemas. Hugging Face
Datasets tries to load those structures into Arrow before ms-swift can remove
unused columns, and fails when a later batch introduces different fields.
Keep the original file for provenance; this copy contains only SFT model inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

# These are the SFT input fields consumed by the pinned ms-swift preprocessor.
# Metadata, _index, start_frame and end_frame are not consumed by that loader.
MODEL_INPUT_COLUMNS = ("messages", "images", "videos", "audios", "tools", "objects")


def prepare_sft(source: Path, output: Path) -> dict[str, Any]:
    """Stream a model-input copy, preserving record order and input values.

    Fail on a malformed row without replacing an existing output or publishing
    a partial file. No media is downloaded, resolved, decoded, or rewritten.
    """
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if source == output:
        raise ValueError("choose a separate output path; retain the original JSONL for provenance")
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    removed: Counter[str] = Counter()
    rows = 0
    temporary: Path | None = None
    try:
        with source.open(encoding="utf-8") as reader, tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False
        ) as writer:
            temporary = Path(writer.name)
            for line_number, line in enumerate(reader, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected a JSON object")
                    messages = row.get("messages")
                    if not isinstance(messages, list) or not messages:
                        raise ValueError("messages must be a non-empty list")
                    for message in messages:
                        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
                            raise ValueError("each message must contain a string role")
                        if not isinstance(message.get("content"), str):
                            raise ValueError("each message must contain string content")
                    cleaned = {key: row[key] for key in MODEL_INPUT_COLUMNS if key in row}
                    serialized = json.dumps(cleaned, ensure_ascii=False, allow_nan=False)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"line {line_number}: {exc}") from exc
                writer.write(serialized + "\n")
                removed.update(set(row).difference(MODEL_INPUT_COLUMNS))
                rows += 1
            if rows == 0:
                raise ValueError("input has no SFT records")
        # A failed conversion must not leave a usable-looking partial dataset.
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "input": str(source),
        "output": str(output),
        "rows": rows,
        "removed_columns": dict(sorted(removed.items())),
        "media_paths": "unchanged; resolve from the training working directory",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original SFT JSONL, retained unchanged.")
    parser.add_argument("--output", type=Path, required=True, help="Destination model-input JSONL.")
    args = parser.parse_args(argv)
    print(json.dumps(prepare_sft(args.source, args.output), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
