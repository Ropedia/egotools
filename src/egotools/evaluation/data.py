"""Dataset paths and MCQ prompts for EgoTools evaluation."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from egotools.manifest import OPTION_LETTERS


def resolve_dataset_root(dataset_root: str | Path | None, version: str = "custom") -> Path:
    """Resolve an explicit root or a parent containing a named version."""
    value = dataset_root or os.environ.get("EGOTOOLS_BENCH_ROOT")
    if not value:
        raise ValueError("set --dataset-root or EGOTOOLS_BENCH_ROOT to a directory containing manifest.tsv")
    root = Path(value).expanduser().resolve()
    if not (root / "manifest.tsv").is_file() and (root / version / "manifest.tsv").is_file():
        root /= version
    if not (root / "manifest.tsv").is_file():
        raise FileNotFoundError(
            f"manifest.tsv not found under {root}; use egotools-prepare-benchmark to prepare your download"
        )
    return root


_MCQ_PROMPT_TEMPLATE = (
    "You are watching an egocentric (first-person) video of a user using "
    "tools or performing manual tasks.\n"
    "Answer the following multiple-choice question about the video.\n"
    "Respond with ONLY the single uppercase letter of the best option "
    "(one of: {letters}). Do not include any explanation.\n\n"
    "Question: {question}\n"
    "{options}\n"
    "Answer: "
)


def _is_empty_option(value: Any) -> bool:
    """Recognize empty option cells, including NaN from tabular inputs."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return True
    return str(value).strip().lower() in ("", "nan", "none")


def build_mcq_prompt_text(row: dict) -> tuple[str, list[str]]:
    """Build an MCQ prompt and retain the letters of nonempty A-H options."""
    kept_letters: list[str] = []
    option_lines: list[str] = []
    for letter in OPTION_LETTERS:
        value = row.get(letter)
        if _is_empty_option(value):
            continue
        kept_letters.append(letter)
        option_lines.append(f"{letter}. {str(value).strip()}")

    if not kept_letters:
        kept_letters = ["A"]
        option_lines = ["A. (no options provided)"]

    return _MCQ_PROMPT_TEMPLATE.format(
        letters=", ".join(kept_letters),
        question=str(row.get("question", "")).strip(),
        options="\n".join(option_lines),
    ), kept_letters
