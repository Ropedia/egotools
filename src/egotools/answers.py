"""Answer extraction used by the EgoTools development evaluation adapter.

Keep the letter/prefix and option-text fallback behavior shared between the
standalone scorer and VLMEvalKit. This parser is heuristic, not an LLM judge.
"""

from __future__ import annotations

import math
import re
from typing import Any

_OPTION_LETTERS = tuple("ABCDEFGH")

_ANSWER_PREFIXES = (
    "Final Answer:",
    "The best answer is",
    "The correct answer is",
    "The answer is",
    "The answer",
    "The best option is",
    "The correct option is",
    "Best answer:",
    "Best option:",
    "Answer:",
    "Option:",
)


_OPTION_MATCH_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "onto",
    "while",
    "before",
    "after",
    "person",
    "user",
    "uses",
    "use",
    "used",
    "using",
}


def _normalize_answer_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _content_tokens(value: Any) -> set[str]:
    return {
        token
        for token in _normalize_answer_text(value).split()
        if len(token) > 2 and token not in _OPTION_MATCH_STOPWORDS
    }


def _extract_letter_from_options(prediction: str, options: dict[str, Any] | None) -> str:
    if not options:
        return ""
    pred_norm = _normalize_answer_text(prediction)
    if not pred_norm:
        return ""

    # High-confidence exact/containment match against the option text.
    for letter in _OPTION_LETTERS:
        opt_norm = _normalize_answer_text(options.get(letter, ""))
        if len(opt_norm) >= 15 and (opt_norm in pred_norm or (len(pred_norm) >= 15 and pred_norm in opt_norm)):
            return letter

    pred_tokens = _content_tokens(prediction)
    if not pred_tokens:
        return ""
    scored: list[tuple[float, int, str]] = []
    for letter in _OPTION_LETTERS:
        opt_tokens = _content_tokens(options.get(letter, ""))
        if not opt_tokens:
            continue
        overlap = len(pred_tokens & opt_tokens)
        scored.append((overlap / len(opt_tokens), overlap, letter))
    scored.sort(reverse=True)
    if not scored:
        return ""
    best = scored[0]
    second_ratio = scored[1][0] if len(scored) > 1 else 0.0
    if best[0] >= 0.55 and best[1] >= 3 and best[0] >= second_ratio + 0.20:
        return best[2]
    return ""


def extract_letter(prediction: str, options: dict[str, Any] | None = None) -> str:
    """Extract a single A..H letter from a model prediction string.

    Returns '' if no letter could be extracted with reasonable confidence.
    """
    if prediction is None:
        return ""
    s = str(prediction).strip()
    if not s:
        return ""
    for prefix in _ANSWER_PREFIXES:
        s = s.replace(prefix, "")
    s = s.strip()

    # Prefer a standalone letter early in the response.
    m = re.search(r"\b([A-H])\b", s)
    if m is not None:
        return m.group(1)
    m = re.search(r"(?:^|answer\s*[:：]?\s*|option\s*[:：]?\s*)[\(\[]?([A-H])[\)\].:：、]?(?:\s|$)", s, re.I)
    if m is not None:
        return m.group(1).upper()
    return _extract_letter_from_options(s, options)
