#!/usr/bin/env python3
"""Normalize QA JSONL records to eight shuffled answer options.

Use --mode checks_only for offline normalization of existing eight-option
records. Other modes optionally repair text and augment, validate, or reduce
distractors with Gemini (GEMINI_API_KEY). --dry-run reports routing and checks
without creating files. Inputs may be flat QA objects or contain a nested
record object; see docs/DATA_PROCESSING.md for fields and examples.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import statistics
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from egotools.data.benchmark.qa_common import (
    GeminiClient,
    QARecord,
    assert_output_safe,
    extract_frames,
    latest_matching_eval,
    load_env,
    write_normalization_report,
)

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover — optional progress display
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else iter(())

# ─────────────────────────────────────────────────────────────────────
# Video-evidence helpers
# ─────────────────────────────────────────────────────────────────────

def _frames_for_record(rec: QARecord, fps: float = 1.0, max_frames: int = 12,
                       resolution: int = 384) -> list[str]:
    """Sample an explicit QA clip or the corresponding source-video interval.

    clip_path identifies an already trimmed clip, so sampling starts at zero.
    source_video_path identifies a full recording and uses source timestamps.
    Relative paths are resolved against --media-root (default: input directory).
    """
    raw = rec.raw or {}
    clip = raw.get("clip_path")
    video = clip or raw.get("source_video_path")
    if not video:
        return []
    start = float(raw.get("clip_start_seconds") or 0.0)
    end = float(raw.get("clip_end_seconds") or start + 16.0)
    if clip:
        end, start = max(0.0, end - start), 0.0
    if end <= start:
        return []
    return extract_frames(str(video), start, end, fps=fps, max_frames=max_frames,
                          rotation=int(raw.get("rotation") or 0), resolution=resolution)

LOG = logging.getLogger("normalize_to_8choice")

DEFAULT_MODEL = "gemini-flash"  # → gemini-3-flash-preview, see qa_common.MODEL_IDS
TARGET_N_OPTIONS = 8


# ─────────────────────────────────────────────────────────────────────
# Question-type classifier
# ─────────────────────────────────────────────────────────────────────

QTYPE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("placeholder",      re.compile(r"^(hi|why|test|1|2)\s*$", re.I)),
    ("identity",         re.compile(r"\bname of (the|this) (person|woman|man|user)", re.I)),
    ("causal_why",       re.compile(r"^\s*why\s+(did|does|is|are|was|were)\b", re.I)),
    ("purpose_what_for", re.compile(r"\b(what|why)\s+.*\b(for|in order to)\b", re.I)),
    ("count",            re.compile(r"^\s*how many\b", re.I)),
    ("temporal",         re.compile(r"\b(when|before|after|first|next|last|then)\b", re.I)),
    ("spatial",          re.compile(r"\b(where|which side|left|right|above|below|behind|in front)\b", re.I)),
    ("tool_identify",    re.compile(r"\bwhat (tool|object|item|utensil|implement)\b", re.I)),
    ("state_describe",   re.compile(r"\bwhat (is|are|was|were).*(state|condition|status|color|shape)\b", re.I)),
    ("state_compare",    re.compile(r"\b(compared to|versus|vs\.?|more|less|than)\b", re.I)),
    ("procedural_how",   re.compile(r"^\s*how (does|do|is|are|did|can)\b", re.I)),
    ("subjective",       re.compile(r"\b(habitual|consistently|usually|prefer|like|favourite)\b", re.I)),
)


def classify_qtype(question: str) -> str:
    q = (question or "").strip()
    for tag, pat in QTYPE_PATTERNS:
        if pat.search(q):
            return tag
    return "other"


# ─────────────────────────────────────────────────────────────────────
# Gemini prompts
# ─────────────────────────────────────────────────────────────────────

# Anti-hackability rules referenced from every prompt. Kept as a single
# constant so changes propagate to all three modes.
ANTI_HACK_RULES = """\
HARD RULES — distractors must satisfy ALL of these:
1. Each distractor is plausible if the video is unseen and would mislead a
   human who skims without watching carefully.
2. Each distractor MUST be within ±30% of the answer's character length AND
   ±2 tokens of the answer's word count. This is enforced post-generation —
   distractors outside the band cause the WHOLE set to be rejected, not just
   trimmed. If the answer is "None" or 1-2 words, distractors must also be
   "None"-tier short noun phrases. NO distractor longer than answer*1.3.
3. NO distractor is a paraphrase, synonym, or token-permutation of the answer.
4. NO distractor is a substring/superstring of the answer or another distractor.
5. NEVER use meta-options: no "all of the above", "none of the above",
   "both A and B", "I don't know", "unknown".
6. For "why-did/does" questions: distractors must describe DIFFERENT plausible
   intents/causes for actions visible in egocentric tool-use scenes — NOT
   variations of the answer's intent. Avoid templating like the answer.
7. Distractors must be drawn from the egocentric tool-use vocabulary
   (kitchen tools, lab instruments, hand actions, object states) — not from
   abstract or out-of-domain concepts.
8. Distractors must be MUTUALLY EXCLUSIVE with each other AND with the answer.
"""


AUGMENT_PROMPT = """\
You are generating multiple-choice distractors for an egocentric video VQA
benchmark called egotools, focused on intensive tool-use scenarios.

Question type: {qtype}
Question: {question}
Correct answer: {answer}
Existing distractors ({n_existing}):
{existing_block}

Generate {n_needed} ADDITIONAL distractors so that TOTAL options = {target}
(1 correct + {target_minus_1} distractors).

{anti_hack}

Reason step-by-step BEFORE writing the final JSON:
  1. Identify the answer's grammatical category, character length, and the
     egocentric tool-use vocabulary it draws on.
  2. List 5-8 candidate distractors covering different plausible
     misinterpretations of the scene; each must satisfy the HARD RULES.
  3. Eliminate candidates that are paraphrases / token-permutations /
     sub-strings / super-strings of the answer or of any existing distractor.
  4. Eliminate candidates whose character length is outside ±30% of the
     answer's character length.
  5. Choose the {n_needed} hardest survivors and emit the JSON below.

Output JSON only, no prose:
{{
  "new_distractors": ["...", "...", ...],
  "self_check": {{
    "all_mutually_exclusive": true|false,
    "no_meta_options": true|false,
    "lengths_balanced": true|false,
    "no_paraphrase_of_answer": true|false,
    "domain_appropriate": true|false
  }}
}}
"""


REDUCE_PROMPT = """\
You are pruning multiple-choice distractors for an egocentric video VQA
benchmark called egotools, focused on intensive tool-use scenarios.

Question type: {qtype}
Question: {question}
Correct answer: {answer}
Current distractors ({n_existing}, too many — need exactly {target_minus_1}):
{existing_block}

Select EXACTLY {target_minus_1} distractors that maximize difficulty subject
to ALL of the following constraints:

{anti_hack}

REMOVE FIRST in this priority order:
  a. Any meta-option ("all of the above" etc.) — drop unconditionally.
  b. Token-permutations of the correct answer.
  c. Substrings/superstrings of the answer or of another distractor.
  d. Lazy distractors ("unknown", "none", "n/a").
  e. Easy/implausible distractors that a model could reject from the
     question text alone.
After removals, if you still have more than {target_minus_1}, prefer
distractors that share grammatical structure with the answer.

Reason step-by-step BEFORE writing the final JSON:
  1. Apply the priority-order removals above; track which distractor was
     removed and the rule that fired.
  2. Among the survivors, score each by how hard it would be to eliminate
     without watching the video (length match + topic match + grammatical
     parallelism with the answer).
  3. Pick the top {target_minus_1} as `kept_distractors`; the rest go to
     `removed_distractors` with their removal reason.

Output JSON only:
{{
  "kept_distractors": ["...", ... exactly {target_minus_1} items ...],
  "removed_distractors": ["..."],
  "remove_reasons": {{"<distractor>": "<short_reason>"}},
  "self_check": {{
    "all_mutually_exclusive": true|false,
    "no_meta_options": true|false,
    "lengths_balanced": true|false,
    "no_paraphrase_of_answer": true|false,
    "domain_appropriate": true|false
  }}
}}
"""


FIX_PROMPT = """\
You are repairing an existing egotools VQA record so we can preserve the
annotator's effort instead of dropping the QA. Annotator work is precious;
your job is to produce the smallest correct edit, not a rewrite.

Question type: {qtype}
Original question: {question}
Original answer: {answer}
Existing distractors:
{existing_block}
Flags from the audit pipeline (may be empty — still scan the record):
{flag_block}

ALWAYS check and fix these surface issues, even if no flag was raised:
- Strip leading "Question:" / "Q:" / "Q." prefixes from the question.
- Strip leading "Answer:" / "A:" / "A." prefixes from the answer.
- If the question contains ANY embedded timestamp — including phrases
  like "at 02:10", "around 01:55", "near 0:45", "at the 70-second mark",
  "around minute 4", "01:00 anchor", or any other clock-style time
  reference — REMOVE every such phrase and replace each with a single
  scene-relative phrase ("at this moment" / "earlier in the clip" /
  "later in the clip"). When multiple timestamps appear, preserve their
  ordering with relative phrases ("earlier ... later ..."). The clip
  window already carries absolute time information; embedded timestamps
  leak position bias and let models bypass visual reasoning.
- Standard grammar/spelling/capitalization clean-up: capitalize the first
  letter of the question (after stripping any prefix), fix obvious typos
  ("tomatos"→"tomatoes", "successfully"→"successful"), end the question
  with a single "?" if it's interrogative.
- Convert "I/me/my/mine" → "the person/the person/the person's" anywhere
  it appears (first-person leakage), even if no C-flag was raised.

Flag-specific rules:
- B-pii / identity: replace any proper noun referring to a person with a
  generic role ("the person", "the user", "the operator"). Keep tools and
  locations.
- F2-dangling-ref ("at ," or trailing "at."): remove the broken anchor and
  replace with "at this moment" / "during this scene". Never invent a
  timestamp.
- G1-truncated-answer (answer starts lowercase mid-word): reconstruct the
  missing leading character only when context makes it unambiguous;
  otherwise just capitalize the first letter and emit fix_status =
  needs_human_review.
- M-meta-answer ("None" / "unknown" / "n/a" as the answer): rephrase the
  question to a yes/no form if possible (e.g. "Which tool was used?" with
  answer "None" → "Was a tool used?" with answer "No"). If reformulation
  would change meaning, emit needs_human_review.
- E1 / E2 (lazy or permutation distractors): do NOT touch question/answer
  here; those are handled by the option-normalization stage.

Hard guardrails:
- Preserve the original meaning. Do NOT change what the annotator was
  asking about or the correct answer's substance.
- If you are unsure whether an edit changes meaning, emit
  fix_status = needs_human_review and leave the field unchanged.
- If after all checks NOTHING needs editing, emit fix_status =
  no_change_needed and copy the original text verbatim.

Reason step-by-step BEFORE writing the final JSON:
  1. Walk the question and answer left-to-right, listing every defect you
     detect (prefix, embedded timestamp, first-person, typo, dangling
     reference, truncation, PII, meta-answer, grammar/capitalization).
  2. For each defect, decide if a meaning-preserving edit is obvious. If
     yes, draft the edit; if no, flag for human review.
  3. Apply only the meaning-preserving drafts and recheck that the answer
     still answers the (edited) question.
  4. Emit the JSON. Set `fix_status = needs_human_review` if any defect is
     unfixed; otherwise `fixed`; otherwise `no_change_needed` if you found
     nothing to edit.

Output JSON only, no prose:
{{
  "fixed_question": "...",
  "fixed_answer": "...",
  "edits_applied": ["pii_scrub", "typo_fix", "strip_prefix", "embedded_timestamp", "first_person", ...],
  "fix_status": "fixed" | "no_change_needed" | "needs_human_review",
  "self_check": {{
    "preserves_original_meaning": true|false,
    "no_meta_options_introduced": true|false,
    "answer_remains_correct": true|false
  }}
}}
"""


VALIDATE_PROMPT = """\
You are auditing an already 8-option multiple-choice item for an egocentric
video VQA benchmark (egotools).

Question type: {qtype}
Question: {question}
Correct answer: {answer}
Distractors (7 expected):
{existing_block}

Audit ONLY for the rules below — do NOT propose new distractors unless one
is unfixable. Return whether the set is acceptable as-is.

{anti_hack}

Reason step-by-step BEFORE writing the final JSON:
  1. For each of the 7 distractors, list which HARD RULES it satisfies and
     which it violates.
  2. If 0 violations: verdict = "accept".
  3. If 1-2 distractors are individually fixable (length / paraphrase /
     domain): verdict = "fix" and propose a single replacement per offender
     in `suggested_replacements`.
  4. If ≥3 violations or the set is structurally broken (e.g., the answer
     is itself a meta-option): verdict = "reject".

Output JSON only:
{{
  "verdict": "accept" | "fix" | "reject",
  "violations": ["<rule_short_name>", ...],
  "suggested_replacements": {{"<bad_distractor>": "<replacement_or_null>"}},
  "self_check": {{
    "all_mutually_exclusive": true|false,
    "no_meta_options": true|false,
    "lengths_balanced": true|false,
    "no_paraphrase_of_answer": true|false,
    "domain_appropriate": true|false
  }}
}}
"""


# ─────────────────────────────────────────────────────────────────────
# Anti-hackability post-checks (deterministic, run AFTER Gemini)
# ─────────────────────────────────────────────────────────────────────

META_OPTION_PATTERN = re.compile(
    r"^\s*(all|none|both|neither|either|i (don'?t|do not) know|n/?a|unknown)\b",
    re.I,
)
WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set[str]:
    return set(WORD_RE.findall((s or "").lower()))


def _is_meta(option: str) -> bool:
    return bool(META_OPTION_PATTERN.match(option or ""))


def _is_token_permutation(a: str, b: str) -> bool:
    return _tokens(a) == _tokens(b) and len(_tokens(a)) > 0


def _is_sub_or_superstring(a: str, b: str) -> bool:
    al, bl = (a or "").strip().lower(), (b or "").strip().lower()
    if not al or not bl or al == bl:
        return False
    return al in bl or bl in al


def _length_outlier(answer: str, distractors: list[str]) -> str | None:
    """Flag bidirectional length bias — answer must be neither much longer
    nor much shorter than the distractor mean (z within ±1.5σ). Asymmetric
    only-too-long check missed cases where Gemini generates longer
    distractors than the annotator's terse answer (pick-shortest hack).
    """
    lens = [len(d) for d in distractors]
    if len(lens) < 2:
        return None
    mu, sd = statistics.mean(lens), statistics.pstdev(lens) or 1.0
    z = (len(answer) - mu) / max(sd, 1.0)
    if z > 1.5:
        return f"answer too long: {len(answer)} vs distractor μ={mu:.1f} σ={sd:.1f} (z={z:+.2f})"
    if z < -1.5:
        return f"answer too short: {len(answer)} vs distractor μ={mu:.1f} σ={sd:.1f} (z={z:+.2f})"
    return None


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _question_overlap_skew(question: str, answer: str, distractors: list[str]) -> str | None:
    q = _tokens(question)
    a_overlap = _jaccard(q, _tokens(answer))
    d_overlaps = [_jaccard(q, _tokens(d)) for d in distractors]
    if not d_overlaps:
        return None
    mu = sum(d_overlaps) / len(d_overlaps)
    if a_overlap - mu > 0.15:
        return f"answer-question overlap ({a_overlap:.2f}) > distractor mean ({mu:.2f}) + 0.15"
    return None


@dataclass
class CheckResult:
    passed: bool
    violations: list[str] = field(default_factory=list)


def run_anti_hack_checks(question: str, answer: str, distractors: list[str]) -> CheckResult:
    """Run the deterministic post-checks. Returns ALL violations, not just first."""
    violations: list[str] = []
    if not question.strip():
        violations.append("empty_question")
    if not answer.strip():
        violations.append("empty_answer")
    if any(not distractor.strip() for distractor in distractors):
        violations.append("empty_distractor")

    for d in distractors:
        if _is_meta(d):
            violations.append(f"meta-option: {d!r}")
        if _is_token_permutation(answer, d):
            violations.append(f"token-permutation of answer: {d!r}")
        if _is_sub_or_superstring(answer, d):
            violations.append(f"sub/superstring of answer: {d!r}")
    seen: list[str] = []
    for d in distractors:
        for prev in seen:
            if _is_sub_or_superstring(prev, d):
                violations.append(f"sub/superstring within distractors: {prev!r} ↔ {d!r}")
        seen.append(d)

    if (msg := _length_outlier(answer, distractors)):
        violations.append(msg)
    if (msg := _question_overlap_skew(question, answer, distractors)):
        violations.append(msg)

    return CheckResult(passed=not violations, violations=violations)


# ─────────────────────────────────────────────────────────────────────
# Mode dispatch
# ─────────────────────────────────────────────────────────────────────

def decide_mode(record: QARecord) -> str:
    """augment | validate | reduce — based on current distractor count."""
    n = len(record.distractors)
    target_d = TARGET_N_OPTIONS - 1
    if n < target_d:
        return "augment"
    if n == target_d:
        return "validate"
    return "reduce"


def shuffle_options(qa_id: str, answer: str, distractors: list[str]) -> tuple[list[str], int]:
    """Shuffle deterministically using the question ID as the random seed."""
    options = [answer] + list(distractors)
    rng = random.Random(qa_id)
    rng.shuffle(options)
    return options, options.index(answer)


def force_target_distractors(distractors: list[str], target_d: int) -> tuple[list[str], list[str]]:
    """Force a relaxed-mode fallback candidate to the exact distractor count.

    Relaxed mode is allowed to keep anti-hack violations for recall, but the
    benchmark format still requires exactly 8 options. Reduce failures can
    otherwise leak the original 8/9 distractors through and produce 9/10-option
    records.
    """
    cleaned = [str(d).strip() for d in distractors if str(d).strip()]
    notes: list[str] = []
    if len(cleaned) > target_d:
        notes.append(f"relaxed_truncated_to_{target_d}_distractors: {len(cleaned)}->{target_d}")
        cleaned = cleaned[:target_d]
    elif len(cleaned) < target_d:
        notes.append(f"relaxed_underfilled_after_fallback: {len(cleaned)}/{target_d}")
    return cleaned, notes


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from Gemini output. Tolerates markdown fences."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S)
    # Find first { ... last }.
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(s[start:end + 1])
    except json.JSONDecodeError:
        return None


def _format_existing_distractors(distractors: list[str]) -> str:
    return "\n".join(f"  - {d}" for d in distractors) if distractors else "  (none)"


MAX_AUGMENT_RETRIES = 2  # Bounded: original + 2 retries = 3 calls max per QA.


def _augment_once(
    client: GeminiClient, rec: QARecord, prior_violations: list[str],
    frames: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Single Gemini augment attempt; returns (combined_distractors, violations).
    On retry, prior_violations is fed back so Gemini can correct.
    """
    target_d = TARGET_N_OPTIONS - 1
    n_needed = target_d - len(rec.distractors)
    feedback = ""
    if prior_violations:
        feedback = (
            "\nPRIOR ATTEMPT WAS REJECTED for these violations — fix them now:\n"
            + "\n".join(f"  - {v}" for v in prior_violations)
            + "\nMake sure each new distractor's character length is within 30% of "
            f"the answer's length ({len(rec.answer)} chars).\n"
        )
    video_clause = (
        "\nVIDEO EVIDENCE: The user has attached frames from the clip. Each new "
        "distractor MUST refer to objects/actions/states either visible in the "
        "frames OR plausibly absent (a similar-looking tool that could be confused "
        "for what's there). Do NOT generate distractors that contradict the visible "
        "scene (wrong room, wrong materials, objects that clearly aren't present).\n"
        if frames else ""
    )
    prompt = AUGMENT_PROMPT.format(
        qtype=classify_qtype(rec.question),
        question=rec.question, answer=rec.answer,
        n_existing=len(rec.distractors),
        existing_block=_format_existing_distractors(rec.distractors),
        n_needed=n_needed, target=TARGET_N_OPTIONS,
        target_minus_1=target_d, anti_hack=ANTI_HACK_RULES + video_clause + feedback,
    )
    resp = client.generate(prompt, frames=frames)
    if resp.error:
        return [], [f"gemini_error: {resp.error}"]
    parsed = _extract_json(resp.text)
    if not parsed or not isinstance(parsed.get("new_distractors"), list):
        return [], ["bad_json_from_gemini"]
    new_d = [str(x).strip() for x in parsed["new_distractors"] if str(x).strip()]
    combined = (list(rec.distractors) + new_d)[:target_d]
    if len(combined) != target_d:
        return [], [f"insufficient_distractors: got {len(combined)}, need {target_d}"]
    return combined, []


def _gemini_augment(client: GeminiClient, rec: QARecord) -> tuple[list[str], bool, list[str]]:
    """Generate distractors to reach 8 total. Retries up to MAX_AUGMENT_RETRIES
    times with prior violations fed back. Returns the BEST attempt — last try
    if none pass anti-hack checks (caller decides whether to skip)."""
    target_d = TARGET_N_OPTIONS - 1
    if len(rec.distractors) >= target_d:
        return rec.distractors[:target_d], True, []
    last: tuple[list[str], list[str]] = ([], [])
    for attempt in range(MAX_AUGMENT_RETRIES + 1):
        combined, vio = _augment_once(client, rec, last[1] if attempt > 0 else [])
        if vio:
            last = (combined, vio)
            continue
        checks = run_anti_hack_checks(rec.question, rec.answer, combined)
        if checks.passed:
            return combined, True, []
        last = (combined, checks.violations)
    return last[0], False, last[1]


def _gemini_reduce(client: GeminiClient, rec: QARecord) -> tuple[list[str], bool, list[str]]:
    """Pick the 7 hardest distractors. Anti-hack check after."""
    target_d = TARGET_N_OPTIONS - 1
    prompt = REDUCE_PROMPT.format(
        qtype=classify_qtype(rec.question),
        question=rec.question, answer=rec.answer,
        n_existing=len(rec.distractors),
        existing_block=_format_existing_distractors(rec.distractors),
        target_minus_1=target_d, anti_hack=ANTI_HACK_RULES,
    )
    resp = client.generate(prompt)
    if resp.error:
        return [], False, [f"gemini_error: {resp.error}"]
    parsed = _extract_json(resp.text)
    if not parsed or not isinstance(parsed.get("kept_distractors"), list):
        return [], False, ["bad_json_from_gemini"]
    kept = [str(x).strip() for x in parsed["kept_distractors"] if str(x).strip()]
    # Gemini sometimes returns paraphrases; restrict to original set.
    original = {d.strip().lower(): d for d in rec.distractors}
    kept_in_original = [original[k.lower()] for k in kept if k.lower() in original]
    if len(kept_in_original) < target_d:
        # Fall back to first N originals not in remove list.
        remove = {str(x).strip().lower() for x in (parsed.get("removed_distractors") or [])}
        for d in rec.distractors:
            if d not in kept_in_original and d.strip().lower() not in remove:
                kept_in_original.append(d)
                if len(kept_in_original) == target_d:
                    break
    kept_in_original = kept_in_original[:target_d]
    if len(kept_in_original) != target_d:
        return [], False, [f"reduce_underfilled: {len(kept_in_original)}/{target_d}"]
    checks = run_anti_hack_checks(rec.question, rec.answer, kept_in_original)
    return kept_in_original, checks.passed, checks.violations


def _gemini_fix(
    client: GeminiClient, rec: QARecord, fix_flags: list[str],
) -> tuple[str, str, list[str], str]:
    """Repair question/answer text issues per the audit flags. Returns
    (fixed_question, fixed_answer, edits_applied, fix_status).

    fix_status ∈ {"fixed", "no_change_needed", "needs_human_review", "error"}.
    Caller should surface needs_human_review records to a separate review CSV
    rather than auto-accepting the fix.
    """
    if not fix_flags:
        return rec.question, rec.answer, [], "no_change_needed"
    flag_block = "\n".join(f"  - {f}" for f in fix_flags)
    prompt = FIX_PROMPT.format(
        qtype=classify_qtype(rec.question),
        question=rec.question, answer=rec.answer,
        existing_block=_format_existing_distractors(rec.distractors),
        flag_block=flag_block,
    )
    resp = client.generate(prompt)
    if resp.error:
        return rec.question, rec.answer, [], "error"
    parsed = _extract_json(resp.text)
    if not parsed:
        return rec.question, rec.answer, [], "error"
    status = str(parsed.get("fix_status", "needs_human_review"))
    fixed_q = str(parsed.get("fixed_question") or rec.question).strip() or rec.question
    fixed_a = str(parsed.get("fixed_answer") or rec.answer).strip() or rec.answer
    edits = [str(e) for e in (parsed.get("edits_applied") or [])]
    # Safety: if Gemini says it changed meaning or answer-correctness, demote.
    sc = parsed.get("self_check") or {}
    if not all([sc.get("preserves_original_meaning"), sc.get("answer_remains_correct")]):
        status = "needs_human_review"
    return fixed_q, fixed_a, edits, status


# Flags that route to the FIX prompt (textual repair). Distractor-level
# flags (E1, E2) are handled by the augment/reduce path, not here.
TEXT_FIX_FLAGS = {
    "B-pii", "F1-typo", "F2-dangling-ref", "G1-truncated-answer",
    "C-first-person-leak", "M-meta-answer",
    "S1-leading-prefix", "S2-embedded-timestamp", "S3-grammar-pass",
}


# Surface-noise patterns we tag deterministically before calling Gemini.
# These ensure every record carries at least an S3-grammar-pass flag so
# the fix-all default puts every record through the FIX_PROMPT.
_RE_LEADING_PREFIX = re.compile(r"^\s*(question|q|answer|a)\s*[:.]\s*", re.I)
_RE_EMBEDDED_TS = re.compile(
    # any clock-style time anywhere in the sentence, with common preposition
    # cues: at / around / near / by / before / after / approximately /
    # roughly / at-the-N-second-mark / around-minute-N. The regex matches
    # the *whole* phrase incl. the preposition so the FIX_PROMPT can drop
    # it cleanly.
    r"\b(?:at|around|near|by|before|after|approximately|roughly)\s+"
    r"(?:the\s+)?\d{1,2}:\d{2}(?::\d{2})?\b"
    r"|\bat\s+the\s+\d{1,3}\s*-?\s*second\s+mark\b"
    r"|\baround\s+minute\s+\d{1,3}\b"
    r"|\b\d{1,2}:\d{2}(?::\d{2})?\s+(?:anchor|mark|timestamp|frame|moment)\b",
    re.I,
)


def detect_surface_flags(question: str, answer: str) -> list[str]:
    """Cheap regex pre-pass that always runs. Adds:
      S1-leading-prefix   if "Question:" / "Q:" / "Answer:" / "A:" prefix
      S2-embedded-timestamp if a clock-style time reference is in the question
      S3-grammar-pass     unconditionally — guarantees Gemini sees every
                          record exactly once for a baseline grammar/style
                          clean-up.
    """
    flags = ["S3-grammar-pass"]
    if _RE_LEADING_PREFIX.search(question or "") or _RE_LEADING_PREFIX.search(answer or ""):
        flags.append("S1-leading-prefix")
    if _RE_EMBEDDED_TS.search(question or ""):
        flags.append("S2-embedded-timestamp")
    return flags


def needs_text_fix(fix_flags: Iterable[str]) -> list[str]:
    return [f for f in fix_flags if f in TEXT_FIX_FLAGS]


def _gemini_validate(client: GeminiClient, rec: QARecord) -> tuple[list[str], bool, list[str]]:
    """For n_dist == 7. Audit only; replace bad distractors if suggested."""
    target_d = TARGET_N_OPTIONS - 1
    prompt = VALIDATE_PROMPT.format(
        qtype=classify_qtype(rec.question),
        question=rec.question, answer=rec.answer,
        existing_block=_format_existing_distractors(rec.distractors),
        anti_hack=ANTI_HACK_RULES,
    )
    resp = client.generate(prompt)
    if resp.error:
        return [], False, [f"gemini_error: {resp.error}"]
    parsed = _extract_json(resp.text)
    if not parsed:
        return [], False, ["bad_json_from_gemini"]
    verdict = str(parsed.get("verdict", "")).lower()
    if verdict == "reject":
        return [], False, ["gemini_rejected: " + ";".join(map(str, parsed.get("violations") or []))]
    distractors = list(rec.distractors)
    if verdict == "fix":
        replacements = parsed.get("suggested_replacements") or {}
        if isinstance(replacements, dict):
            distractors = [
                str(replacements.get(d) or d).strip() if str(replacements.get(d) or "").strip()
                else d
                for d in distractors
            ]
    distractors = distractors[:target_d]
    if len(distractors) != target_d:
        return [], False, [f"validate_wrong_count: {len(distractors)}/{target_d}"]
    checks = run_anti_hack_checks(rec.question, rec.answer, distractors)
    return distractors, checks.passed, checks.violations


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalize VQA records to 8-choice format")
    p.add_argument("--input-jsonl", type=Path, required=True,
                   help="Input QA JSONL (read-only)")
    p.add_argument("--reasons-csv", type=Path,
                   help="Optional QA review CSV containing qa_id, decision, reason")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--env-file", type=Path, help="Optional local KEY=VALUE file")
    p.add_argument("--media-root", type=Path, help="Root for input media paths; defaults to input directory")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Gemini model id from qa_common.MODEL_IDS")
    p.add_argument("--mode", choices=("auto", "augment", "validate", "reduce", "checks_only"),
                   default="auto",
                   help="auto = decide per QA (recommended). 'checks_only' normalizes existing options without Gemini")
    p.add_argument("--sample-limit", type=int, default=0,
                   help="Process only first N records (0 = all)")
    p.add_argument("--dry-run", action="store_true",
                   help="Classify and route, but do NOT call Gemini and do NOT write outputs")
    p.add_argument("--resume", action="store_true",
                   help="Resume from a previous run in --output-dir: skip qa_ids already in "
                        "normalized.jsonl or skipped.jsonl, append new results instead of "
                        "truncating. Use this if a previous run was killed mid-stream.")
    p.add_argument("--relaxed", action="store_true",
                   help="Relaxed mode: when Gemini augment/reduce/validate produces "
                        "distractors that fail anti-hack QC (length balance, word-count "
                        "balance, meta-options), STILL emit the record with quality_violations "
                        "tagged. Only truly broken QAs (no answer / no distractors / can't "
                        "form options) are dropped. Use for higher recall at "
                        "the cost of some QAs being length/word-count guessable.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            yield row


def record_from_jsonl_row(row: dict[str, Any]) -> QARecord:
    """Read either a flat QA object or a nested record object."""
    inner = row.get("record") if isinstance(row.get("record"), dict) else row
    video = inner.get("_video") or {}
    distractors = inner.get("distractors")
    if distractors is None and inner.get("normalized_options"):
        options = inner["normalized_options"]
        correct = inner.get("correct_index")
        if not isinstance(correct, int) or not 0 <= correct < len(options):
            raise ValueError("normalized_options requires a valid correct_index")
        distractors = [text for index, text in enumerate(options) if index != correct]
        inner = dict(inner, answer=options[correct])
    return QARecord(
        qa_id=row.get("qa_id") or inner.get("annotation_id") or "",
        source_file=row.get("source_file") or "",
        source_id=video.get("source_id") or row.get("source_id") or "",
        canonical_video_id=video.get("canonical_video_id") or row.get("canonical_video_id") or "",
        display_name=video.get("display_name") or row.get("display_name") or "",
        annotation_id=inner.get("annotation_id") or "",
        question=inner.get("question") or "",
        answer=inner.get("answer") or "",
        distractors=list(distractors or []),
        annotator_id=inner.get("annotator_id"),
        evals=list(inner.get("evals") or []),
        raw=inner,
    )


def load_fix_flags_from_reasons(reasons_csv: Path | None) -> dict[str, list[str]]:
    """Aggregate optional review flags by qa_id.

    Picks up rows marked for revision or exclusion and groups their reason
    fields as {qa_id: sorted unique list of rule_ids}.
    Returns {} if the CSV is missing — callers should treat as 'no flags'.
    """
    out: dict[str, set[str]] = {}
    if reasons_csv is None or not reasons_csv.is_file():
        return {}
    with reasons_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            decision = (row.get("decision") or "").lower()
            if decision in ("keep", "kept", ""):
                continue
            qa_id = row.get("qa_id") or ""
            reason = (row.get("reason") or "").strip()
            if not qa_id or not reason or reason == "ok":
                continue
            out.setdefault(qa_id, set()).add(reason.split(":", 1)[0])
    return {k: sorted(v) for k, v in out.items()}


def main() -> int:  # noqa: PLR0912 — orchestration
    args = parse_args()
    setup_logging(args.verbose)

    if not args.input_jsonl.is_file():
        LOG.error("input-jsonl not found: %s", args.input_jsonl)
        return 2

    # Output safety vs. the input file's parent dir.
    assert_output_safe(args.input_jsonl.parent, args.output_dir)
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and args.mode != "checks_only":
        load_env(args.env_file)
        client = GeminiClient(model=args.model)
    else:
        client = None

    # Load optional per-QA review flags. If absent,
    # fix_flags will simply be empty per QA (text-fix path becomes a no-op).
    fix_flags_by_qa = load_fix_flags_from_reasons(args.reasons_csv)
    LOG.info("fix_flags loaded for %d QAs from %s", len(fix_flags_by_qa), args.reasons_csv)

    counts: dict[str, int] = {
        "input": 0, "augment": 0, "validate": 0, "reduce": 0,
        "normalized": 0, "skipped": 0, "anti_hack_failed": 0,
        "resumed": 0,
    }
    normalized_path = args.output_dir / "normalized.jsonl"
    skipped_path = args.output_dir / "skipped.jsonl"
    reasons_path = args.output_dir / "reasons.csv"

    # Resume skips IDs already written to normalized.jsonl or skipped.jsonl.
    # Option shuffling is deterministic; external model responses may vary.
    done_ids: set[str] = set()
    if args.resume and not args.dry_run:
        for p in (normalized_path, skipped_path):
            if p.is_file():
                for line in p.read_text(errors="replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    qid = d.get("qa_id")
                    if qid:
                        done_ids.add(qid)
        LOG.info("--resume: %d qa_ids already processed in %s", len(done_ids), args.output_dir)

    write_mode = "a" if (args.resume and done_ids) else "w"
    norm_fh = None if args.dry_run else normalized_path.open(write_mode, encoding="utf-8")
    skip_fh = None if args.dry_run else skipped_path.open(write_mode, encoding="utf-8")
    reasons_fh = None if args.dry_run else reasons_path.open(write_mode, encoding="utf-8", newline="")
    reasons_writer = None
    if reasons_fh is not None:
        reasons_writer = csv.writer(reasons_fh)
        if write_mode == "w":
            reasons_writer.writerow(["qa_id", "mode", "decision", "reason", "score", "detail"])

    # Pre-load records so the progress display has an exact total.
    all_rows = list(iter_jsonl(args.input_jsonl))
    if args.sample_limit:
        all_rows = all_rows[: args.sample_limit]
    LOG.info("processing %d records from %s", len(all_rows), args.input_jsonl)

    pbar = tqdm(
        all_rows,
        total=len(all_rows),
        desc="normalize",
        unit="qa",
        dynamic_ncols=True,
        disable=not sys.stderr.isatty() and not args.verbose,
    )

    try:
        for row in pbar:
            counts["input"] += 1
            if done_ids and (row.get("qa_id") in done_ids):
                counts["resumed"] += 1
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(norm=counts["normalized"], skip=counts["skipped"],
                                     resumed=counts["resumed"], refresh=False)
                continue
            rec = record_from_jsonl_row(row)
            if not rec.qa_id or not rec.canonical_video_id:
                raise ValueError("Each QA needs qa_id and canonical_video_id")
            media_root = args.media_root or args.input_jsonl.parent
            rec.raw = dict(rec.raw)
            for media_field in ("clip_path", "source_video_path"):
                if rec.raw.get(media_field):
                    media_path = Path(rec.raw[media_field])
                    rec.raw[media_field] = str(media_path if media_path.is_absolute() else media_root / media_path)
            mode = args.mode if args.mode in ("augment", "validate", "reduce") else decide_mode(rec)
            counts[mode] = counts.get(mode, 0) + 1

            # In dry-run / checks_only we skip Gemini and only run the
            # deterministic post-checks against the EXISTING set.
            if args.dry_run or args.mode == "checks_only" or client is None:
                checks = run_anti_hack_checks(rec.question, rec.answer, rec.distractors)
                if checks.passed and len(rec.distractors) == TARGET_N_OPTIONS - 1:
                    counts["normalized"] += 1
                    options, correct_index = shuffle_options(rec.qa_id, rec.answer, rec.distractors)
                    if norm_fh is not None:
                        norm_fh.write(json.dumps({
                            "qa_id": rec.qa_id, "source_file": rec.source_file,
                            "source_id": rec.source_id, "canonical_video_id": rec.canonical_video_id,
                            "annotation_id": rec.annotation_id, "question": rec.question,
                            "answer": rec.answer, "normalized_options": options,
                            "correct_index": correct_index, "qtype": classify_qtype(rec.question),
                            "mode_used": "checks_only", "n_options": len(options),
                            "quality_tier": "strict", "quality_violations": [],
                        }, ensure_ascii=False) + "\n")
                else:
                    counts["skipped"] += 1
                    if skip_fh is not None:
                        skip_fh.write(json.dumps({"qa_id": rec.qa_id, "mode": "checks_only",
                            "violations": checks.violations,
                            "n_distractors_in": len(rec.distractors)}, ensure_ascii=False) + "\n")
                if reasons_writer is not None:
                    reasons_writer.writerow([
                        rec.qa_id, mode,
                        "kept" if checks.passed and len(rec.distractors) == TARGET_N_OPTIONS - 1 else "skip",
                        ";".join(checks.violations) or f"n_dist={len(rec.distractors)}",
                        "", "",
                    ])
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(norm=counts["normalized"], skip=counts["skipped"], refresh=False)
                continue

            # ─────────── live Gemini path ───────────
            # Skip Gemini when a stored eval already covers the current QA
            # shape.
            cached = latest_matching_eval(rec)
            normalized_options: list[str] | None = None
            mode_used = mode
            quality_violations: list[str] = []
            decision = "kept"
            reason_tag = ""
            if cached and len(cached.get("options") or []) == TARGET_N_OPTIONS:
                # Use cached options[].text in their stored letter order;
                # but only accept if anti-hack checks pass — cached evals can
                # carry whatever length-skew / meta-options the original
                # annotation had, and we don't want them shortcutting QC.
                cached_opts = [o.get("text", "") for o in cached.get("options") or []]
                if cached.get("answer", "") in cached_opts and len(cached_opts) == TARGET_N_OPTIONS:
                    cand_idx = cached_opts.index(cached.get("answer", ""))
                    cand_distractors = [o for i, o in enumerate(cached_opts) if i != cand_idx]
                    cached_checks = run_anti_hack_checks(rec.question, rec.answer, cand_distractors)
                    if cached_checks.passed:
                        normalized_options = cached_opts
                        correct_idx = cand_idx
                        mode_used = "cached_eval"
                        reason_tag = "cache_hit"
                    # else: leave normalized_options=None so we fall through
                    # to live Gemini augment/reduce/validate below.
            if normalized_options is None:
                # Step 1: text repair. Fix-all default — surface-noise
                # detector adds S3-grammar-pass to every record, so EVERY
                # record gets one Gemini text-pass even if no audit flag
                # was raised. PII/first-person/dangling-ref/typo/etc. add
                # extra flags on top.
                fix_flags = list(row.get("fix_flags") or fix_flags_by_qa.get(rec.qa_id) or [])
                fix_flags = sorted(set(fix_flags) | set(detect_surface_flags(rec.question, rec.answer)))
                text_fixes = needs_text_fix(fix_flags)
                edits_applied: list[str] = []
                fix_status = "no_change_needed"
                if text_fixes:
                    fixed_q, fixed_a, edits_applied, fix_status = _gemini_fix(client, rec, text_fixes)
                    if fix_status == "needs_human_review":
                        # Don't auto-accept; carry the original forward but
                        # surface for review.
                        if skip_fh is not None:
                            skip_fh.write(json.dumps({
                                "qa_id": rec.qa_id,
                                "source_file": rec.source_file,
                                "stage": "fix",
                                "reason": "needs_human_review",
                                "fix_flags": fix_flags,
                                "edits_applied": edits_applied,
                            }) + "\n")
                        counts["skipped"] += 1
                        continue
                    elif fix_status == "fixed":
                        rec = QARecord(
                            qa_id=rec.qa_id, source_file=rec.source_file,
                            source_id=rec.source_id, canonical_video_id=rec.canonical_video_id,
                            display_name=rec.display_name, annotation_id=rec.annotation_id,
                            question=fixed_q, answer=fixed_a,
                            distractors=rec.distractors, annotator_id=rec.annotator_id,
                            evals=rec.evals, raw=rec.raw,
                        )
                # Step 2: option normalization on the (possibly fixed) record.
                if mode == "augment":
                    distractors_out, ok, vio = _gemini_augment(client, rec)
                elif mode == "reduce":
                    distractors_out, ok, vio = _gemini_reduce(client, rec)
                else:  # validate
                    distractors_out, ok, vio = _gemini_validate(client, rec)
                quality_violations: list[str] = []
                if not ok:
                    # Strict path (default): drop record, log skip with reason.
                    if not args.relaxed:
                        counts["skipped"] += 1
                        counts["anti_hack_failed"] += int(bool(vio))
                        if skip_fh is not None:
                            skip_fh.write(json.dumps({
                                "qa_id": rec.qa_id,
                                "source_file": rec.source_file,
                                "mode": mode,
                                "violations": vio,
                                "n_distractors_in": len(rec.distractors),
                            }) + "\n")
                            skip_fh.flush()
                        if reasons_writer is not None:
                            reasons_writer.writerow([rec.qa_id, mode, "skip", ";".join(vio), "", ""])
                        if hasattr(pbar, "set_postfix"):
                            pbar.set_postfix(norm=counts["normalized"], skip=counts["skipped"], refresh=False)
                        continue
                    # Relaxed path: emit anyway, fall back to original distractors
                    # if Gemini's augmented set is empty/short. Only truly broken
                    # QAs (no answer or no distractors at all) get dropped here.
                    fallback = list(distractors_out) if distractors_out else list(rec.distractors)
                    fallback, fallback_notes = force_target_distractors(fallback, TARGET_N_OPTIONS - 1)
                    if not rec.answer.strip() or len(fallback) != TARGET_N_OPTIONS - 1:
                        counts["skipped"] += 1
                        counts["anti_hack_failed"] += int(bool(vio))
                        if skip_fh is not None:
                            skip_fh.write(json.dumps({
                                "qa_id": rec.qa_id,
                                "source_file": rec.source_file,
                                "mode": mode,
                                "stage": "relaxed_truly_broken",
                                "violations": vio + fallback_notes + (
                                    ["empty_answer"] if not rec.answer.strip() else []
                                ) + (
                                    ["wrong_distractor_count_after_fallback"] if len(fallback) != TARGET_N_OPTIONS - 1 else []
                                ),
                                "n_distractors_in": len(rec.distractors),
                            }) + "\n")
                            skip_fh.flush()
                        if reasons_writer is not None:
                            reasons_writer.writerow([rec.qa_id, mode, "skip_truly_broken",
                                                    ";".join(vio), "", ""])
                        continue
                    distractors_out = fallback
                    quality_violations = list(vio) + fallback_notes
                    counts["anti_hack_failed"] += int(bool(vio))
                    counts["relaxed_kept"] = counts.get("relaxed_kept", 0) + 1
                    decision = "kept_relaxed"
                    reason_tag = "anti_hack_violations_kept_in_relaxed_mode"
                    mode_used = f"{mode}_relaxed"
                normalized_options, correct_idx = shuffle_options(rec.qa_id, rec.answer, distractors_out)

            counts["normalized"] += 1
            out_record = {
                "qa_id": rec.qa_id,
                "source_file": rec.source_file,
                "source_id": rec.source_id,
                "canonical_video_id": rec.canonical_video_id,
                "annotation_id": rec.annotation_id,
                "question": rec.question,
                "answer": rec.answer,
                "normalized_options": normalized_options,
                "correct_index": correct_idx,
                "qtype": classify_qtype(rec.question),
                "mode_used": mode_used,
                "n_options": len(normalized_options) if normalized_options else 0,
                "quality_tier": "relaxed" if (mode_used or "").endswith("_relaxed") else "strict",
                "quality_violations": quality_violations,
                "model": args.model if mode_used != "cached_eval" else "cached",
                "fix_flags": list(row.get("fix_flags") or []),
                "fix_edits_applied": edits_applied if normalized_options is not None and mode_used != "cached_eval" else [],
                "fix_status": fix_status if normalized_options is not None and mode_used != "cached_eval" else "skipped",
            }
            if norm_fh is not None:
                norm_fh.write(json.dumps(out_record) + "\n")
                norm_fh.flush()
            if reasons_writer is not None:
                reasons_writer.writerow([rec.qa_id, mode_used, decision, reason_tag, "", ""])
                reasons_fh.flush() if reasons_fh else None
            # Update tqdm postfix so the user sees live counters.
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(
                    norm=counts["normalized"],
                    skip=counts["skipped"],
                    cache=sum(1 for _ in [None] if mode_used == "cached_eval"),
                    refresh=False,
                )
    finally:
        for fh in (norm_fh, skip_fh, reasons_fh):
            if fh is not None:
                fh.close()

    if not args.dry_run:
        write_normalization_report(
            args.output_dir,
            args=vars(args) | {"input_jsonl": str(args.input_jsonl), "output_dir": str(args.output_dir),
                               "env_file": "<not-logged>"},
            counts=counts,
            extra={"target_n_options": TARGET_N_OPTIONS},
        )
    LOG.info("counts: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
