#!/usr/bin/env python3
"""Assign research-facing tracks for the finalized EgoTools benchmark.

These tracks are intentionally coarse and capability-oriented. The goal is to
support benchmark reporting for tool-use-centric VLM understanding, not merely
to mirror surface qtype labels.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TRACKS = {
    "tool_perception_state_grounding": {
        "name": "Tool Perception & State Grounding",
        "definition": (
            "Recognizing tools, manipulated objects, visible attributes, "
            "states, quantities, and grounded scene facts."
        ),
    },
    "spatial_intelligence": {
        "name": "Spatial Intelligence",
        "definition": (
            "Embodied spatial reasoning over egocentric scenes: left/right, "
            "front/back, depth, distance, containment, support, layout, and "
            "relative hand/tool/object geometry."
        ),
    },
    "procedural_temporal_dynamics": {
        "name": "Procedural & Temporal Tool-Use Dynamics",
        "definition": (
            "How tool-use unfolds over time: action order, before/after "
            "dependencies, manipulation steps, workflow transitions, and "
            "fine-grained process dynamics."
        ),
    },
    "functional_affordance_causality": {
        "name": "Functional Affordance & Causal Tool Reasoning",
        "definition": (
            "Why a tool/object is used, selected, switched, or suitable; "
            "purpose, affordance, causal explanation, and task-level intent."
        ),
    },
    "dropped": {
        "name": "Dropped",
        "definition": "QA explicitly dropped during manual review.",
    },
}

SUBTRACKS = {
    "tool_object_identity": {
        "name": "Tool/Object Identity",
        "definition": "Identifying which tool, object, ingredient, or manipulated entity is visible or being used.",
    },
    "state_attribute_grounding": {
        "name": "State & Attribute Grounding",
        "definition": "Grounding visible state or attribute changes such as clean/dirty, open/closed, wet/dry, full/empty, cooked/raw.",
    },
    "quantity_consistency_grounding": {
        "name": "Quantity & Consistency Grounding",
        "definition": "Counting, quantity estimation, and explicit factual consistency checks.",
    },
    "egocentric_relation": {
        "name": "Egocentric Relation",
        "definition": "Left/right, front/back, near/far, and other viewer-centered spatial relations.",
    },
    "depth_layout_support": {
        "name": "Depth, Layout & Support",
        "definition": "Depth, distance, containment, support, occlusion, surface layout, and object arrangement.",
    },
    "interaction_geometry": {
        "name": "Interaction Geometry",
        "definition": "Spatial geometry of hands, tools, and target objects during manipulation.",
    },
    "action_order_transition": {
        "name": "Action Order & Transition",
        "definition": "Before/after/next/when relations and transitions between tool-use steps.",
    },
    "tool_manipulation_process": {
        "name": "Tool Manipulation Process",
        "definition": "Fine-grained manipulation procedures such as cutting, stirring, washing, transferring, pouring, or scraping.",
    },
    "task_staging_and_reset": {
        "name": "Task Staging & Reset",
        "definition": "Preparation, staging, cleanup, reset, and workspace transitions that structure tool use.",
    },
    "tool_selection_affordance": {
        "name": "Tool Selection, Switching & Affordance Fit",
        "definition": (
            "Questions about why a specific tool/object is chosen, switched, "
            "rejected, or better suited because of its physical or functional affordance."
        ),
    },
    "workflow_causal_planning": {
        "name": "Workflow Causality & Long-Horizon Planning",
        "definition": (
            "Questions whose answer depends on connecting an earlier action to "
            "a later step, precondition, staging decision, interruption, or workflow dependency."
        ),
    },
    "task_goal_purpose": {
        "name": "Local Task Goal & Action Purpose",
        "definition": (
            "Questions about the immediate purpose of using a tool/object or "
            "performing an action within the current task context."
        ),
    },
}

TRACK_SUBTRACKS = {
    "tool_perception_state_grounding": (
        "tool_object_identity",
        "state_attribute_grounding",
        "quantity_consistency_grounding",
    ),
    "spatial_intelligence": (
        "egocentric_relation",
        "depth_layout_support",
        "interaction_geometry",
    ),
    "procedural_temporal_dynamics": (
        "action_order_transition",
        "tool_manipulation_process",
        "task_staging_and_reset",
    ),
    "functional_affordance_causality": (
        "task_goal_purpose",
        "tool_selection_affordance",
        "workflow_causal_planning",
    ),
}

CAPABILITY_AXES = {
    "tool_perception": {
        "name": "Tool Perception",
        "definition": "The answer mainly requires detecting, recognizing, localizing, or reading visible tool/object state.",
    },
    "tool_reasoning": {
        "name": "Tool Reasoning",
        "definition": "The answer mainly requires inferring purpose, affordance, causal dependency, or task-level intent.",
    },
    "perception_reasoning_mixed": {
        "name": "Perception-Reasoning Mixed",
        "definition": "The answer requires grounded perception plus nontrivial spatial, temporal, or procedural inference.",
    },
}


SPATIAL_PATTERNS = (
    r"\bwhere\b",
    r"\bleft\b",
    r"\bright\b",
    r"\bin front of\b",
    r"\bbehind\b",
    r"\bnext to\b",
    r"\bbeside\b",
    r"\bbetween\b",
    r"\babove\b",
    r"\bbelow\b",
    r"\bunder\b",
    r"\bon top of\b",
    r"\binside\b",
    r"\boutside\b",
    r"\bnearer\b",
    r"\bfarther\b",
    r"\bcloser\b",
    r"\bdistance\b",
    r"\bdepth\b",
    r"\bposition",
    r"\blocation\b",
    r"\blayout\b",
    r"\barranged\b",
    r"\borientation\b",
    r"\bopening\b",
    r"\bcover\b",
)


FUNCTIONAL_PATTERNS = (
    r"\bwhy\b",
    r"\breason\b",
    r"\bpurpose\b",
    r"\bwhat .* for\b",
    r"\bexplain",
    r"\bbecause\b",
    r"\bto prevent\b",
    r"\bto avoid\b",
    r"\bto make\b",
    r"\bto allow\b",
    r"\bto help\b",
    r"\bso that\b",
    r"\bin order to\b",
    r"\bsuitable\b",
    r"\bmore appropriate\b",
    r"\badvantage\b",
    r"\bafford",
    r"\bswitch\b",
    r"\bchange the tool\b",
)

TOOL_SELECTION_CORE_PATTERNS = (
    r"\bswitch(?:es|ed|ing)?\b",
    r"\bchange(?:s|d|ing)? (?:the )?(?:tool|utensil|implement|instrument|object|item)\b",
    r"\bchoose(?:s|n)?\b",
    r"\bselect(?:s|ed|ing)?\b",
    r"\babandon(?:s|ed|ing)?\b",
    r"\breject(?:s|ed|ing)?\b",
    r"\bcommit(?:s|ted|ting)? to\b",
    r"\bmore (?:useful|suitable|appropriate|efficient|precise|stable|controlled)\b",
    r"\bbetter (?:suited|fit|matches|works)\b",
    r"\bafford(?:ance|s)?\b",
    r"\bcapacity\b",
    r"\bprecision\b",
    r"\bgrip\b",
    r"\breach\b",
    r"\bflat\b",
    r"\bbroad\b",
    r"\bnarrow\b",
    r"\bsharp\b",
    r"\bblunt\b",
    r"\bshape\b",
    r"\bsize\b",
)

TOOL_CONTRAST_PATTERNS = (
    r"\binstead of\b",
    r"\brather than\b",
)

TOOL_OBJECT_TERMS = (
    r"\btool\b",
    r"\butensil\b",
    r"\binstrument\b",
    r"\bimplement\b",
    r"\bknife\b",
    r"\bscissors\b",
    r"\bspoon\b",
    r"\bfork\b",
    r"\bchopsticks?\b",
    r"\bturner\b",
    r"\bspatula\b",
    r"\bskimmer\b",
    r"\bpaper towel\b",
    r"\btowel\b",
    r"\bpipett(?:e|or)\b",
    r"\btongs?\b",
    r"\blid\b",
    r"\bpot\b",
    r"\bpan\b",
    r"\bwok\b",
    r"\btray\b",
    r"\bcontainer\b",
    r"\bbowl\b",
    r"\bplate\b",
    r"\bbottle\b",
)

WORKFLOW_CAUSAL_PATTERNS = (
    r"\bwhat later step\b",
    r"\blater\b",
    r"\bearlier\b",
    r"\bprevious(?:ly)?\b",
    r"\bafter\b",
    r"\bbefore\b",
    r"\bwhen\b",
    r"\bonce\b",
    r"\bintervening\b",
    r"\bnext\b",
    r"\bsequence\b",
    r"\border\b",
    r"\bstage(?:s|d|ing)?\b",
    r"\bprecondition\b",
    r"\bsetup\b",
    r"\bset up\b",
    r"\bprepare(?:s|d|ing)? for\b",
    r"\bready\b",
    r"\bworkflow\b",
    r"\btransition\b",
    r"\bexplains why\b",
    r"\bbefore (?:returning|moving|switching|starting)\b",
    r"\bafter (?:returning|moving|switching|starting|setting|finishing)\b",
)


PROCEDURAL_TEMPORAL_PATTERNS = (
    r"\bwhen\b",
    r"\bafter\b",
    r"\bbefore\b",
    r"\bonce\b",
    r"\bwhile\b",
    r"\bright after\b",
    r"\bimmediately\b",
    r"\bfirst\b",
    r"\bnext\b",
    r"\blater\b",
    r"\bprevious\b",
    r"\bsequence\b",
    r"\border\b",
    r"\bstep\b",
    r"\bthen\b",
    r"\bhow\b",
    r"\bhandle\b",
    r"\btransfer\b",
    r"\bmove\b",
    r"\btransport\b",
    r"\buse .* to\b",
)


SCENE_STATE_PATTERNS = (
    r"\bwhat tool\b",
    r"\bwhich tool\b",
    r"\bwhat object\b",
    r"\bwhich object\b",
    r"\bwhat item\b",
    r"\bwhich item\b",
    r"\bhow many\b",
    r"\bhow much\b",
    r"\bcount\b",
    r"\btrue\b",
    r"\binconsistent\b",
    r"\bconsistent\b",
    r"\bclean\b",
    r"\bdirty\b",
    r"\bwet\b",
    r"\bdry\b",
    r"\bempty\b",
    r"\bfull\b",
    r"\bopen\b",
    r"\bclosed\b",
    r"\bcooked\b",
    r"\braw\b",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(row)
    return rows


def has(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def answer_form(answer: str) -> str:
    text = answer.strip().lower()
    if text.startswith(("to ", "so that ", "in order to ", "because ", "since ")):
        return "causal_or_purpose"
    if re.match(r"^(yes|no)\b", text):
        return "yes_no"
    if re.match(r"^\d+\b|^(one|two|three|four|five|six|seven|eight|nine|ten)\b", text):
        return "quantity"
    if has(SPATIAL_PATTERNS, text):
        return "spatial_relation"
    if has(SCENE_STATE_PATTERNS, text):
        return "state_or_fact"
    if len(text.split()) <= 5:
        return "entity"
    return "action_or_description"


def question_form(question: str) -> str:
    text = question.strip().lower()
    if has(SPATIAL_PATTERNS, text):
        return "spatial"
    if has(FUNCTIONAL_PATTERNS, text):
        return "causal_affordance"
    if has(PROCEDURAL_TEMPORAL_PATTERNS, text):
        return "procedural_temporal"
    if has(SCENE_STATE_PATTERNS, text):
        return "scene_state"
    return "general"


def assign_functional_subtrack(row: dict[str, Any]) -> tuple[str, list[str]]:
    q = str(row.get("question") or "")
    a = str(row.get("answer") or "")
    question_text = q.lower()
    answer_text = a.lower()
    text = f"{question_text} {answer_text}"
    signals: list[str] = []

    # Tool switching / affordance fit should stay distinct from generic
    # "why did they use the tool" questions: the research signal is a
    # contrast between possible tools or a physical/functional suitability claim.
    if has(TOOL_SELECTION_CORE_PATTERNS, text):
        signals.append("subtrack_signal=explicit_tool_selection_or_affordance_fit")
        return "tool_selection_affordance", signals
    if has(TOOL_CONTRAST_PATTERNS, question_text) and has(TOOL_OBJECT_TERMS, question_text):
        signals.append("subtrack_signal=tool_contrast")
        return "tool_selection_affordance", signals

    # Long-horizon causal questions often use temporal surface forms, but the
    # target ability is connecting distant steps rather than naming action order.
    if (
        has(WORKFLOW_CAUSAL_PATTERNS, question_text)
        or answer_text.startswith(("because ", "since "))
        or re.search(r"\bso (?:the|that|it|they|person|cook)\b", answer_text)
    ):
        signals.append("subtrack_signal=workflow_dependency")
        return "workflow_causal_planning", signals

    signals.append("subtrack_signal=immediate_action_purpose")
    return "task_goal_purpose", signals


def assign_perception_subtrack(row: dict[str, Any]) -> tuple[str, list[str]]:
    q = str(row.get("question") or "").lower()
    a = str(row.get("answer") or "").lower()
    text = f"{q} {a}"
    qtype = str(row.get("qtype") or "").lower()

    if qtype in {"count", "state_compare"} or re.search(r"\b(how many|count|consistent|inconsistent|true|false)\b", text):
        return "quantity_consistency_grounding", ["subtrack_signal=quantity_or_consistency"]
    if qtype == "state_describe" or has(SCENE_STATE_PATTERNS, text):
        return "state_attribute_grounding", ["subtrack_signal=visible_state_or_attribute"]
    return "tool_object_identity", ["subtrack_signal=tool_or_object_identity"]


def assign_spatial_subtrack(row: dict[str, Any]) -> tuple[str, list[str]]:
    q = str(row.get("question") or "").lower()
    a = str(row.get("answer") or "").lower()
    text = f"{q} {a}"

    if re.search(r"\b(hand|finger|grasp|hold|reach|touch|tool|knife|spoon|chopsticks?|pipett(?:e|or)|tongs?)\b", text):
        return "interaction_geometry", ["subtrack_signal=hand_tool_object_geometry"]
    if re.search(r"\b(depth|distance|inside|outside|under|above|below|on top of|support|surface|cover|occlud|contain|layout|arranged|between)\b", text):
        return "depth_layout_support", ["subtrack_signal=depth_layout_support_or_containment"]
    return "egocentric_relation", ["subtrack_signal=egocentric_relation"]


def assign_procedural_subtrack(row: dict[str, Any]) -> tuple[str, list[str]]:
    q = str(row.get("question") or "").lower()
    a = str(row.get("answer") or "").lower()
    text = f"{q} {a}"

    if re.search(r"\b(set ?up|stage|staging|prepare|ready|reset|cleanup|clean up|sink|rinse|wash|clear|returning|workspace|counter)\b", text):
        return "task_staging_and_reset", ["subtrack_signal=staging_cleanup_or_reset"]
    if re.search(r"\b(before|after|when|once|next|then|first|later|previous|sequence|order|transition|right after|immediately)\b", q):
        return "action_order_transition", ["subtrack_signal=action_order_or_transition"]
    return "tool_manipulation_process", ["subtrack_signal=fine_grained_tool_manipulation"]


def assign_subtrack(row: dict[str, Any], track_id: str) -> tuple[str | None, list[str]]:
    if track_id == "functional_affordance_causality":
        return assign_functional_subtrack(row)
    if track_id == "tool_perception_state_grounding":
        return assign_perception_subtrack(row)
    if track_id == "spatial_intelligence":
        return assign_spatial_subtrack(row)
    if track_id == "procedural_temporal_dynamics":
        return assign_procedural_subtrack(row)
    return None, []


def assign_capability_axis(track_id: str, subtrack_id: str | None) -> str | None:
    if track_id == "dropped":
        return None
    if track_id == "tool_perception_state_grounding":
        return "tool_perception"
    if track_id == "functional_affordance_causality":
        return "tool_reasoning"
    if track_id == "spatial_intelligence":
        if subtrack_id == "egocentric_relation":
            return "tool_perception"
        return "perception_reasoning_mixed"
    return "perception_reasoning_mixed"


def assign_track(row: dict[str, Any]) -> tuple[str, list[str]]:
    if row.get("is_dropped") or row.get("review_status") == "drop":
        return "dropped", ["review_status=drop"]

    q = str(row.get("question") or "")
    a = str(row.get("answer") or "")
    text = f"{q} {a}".lower()
    qtype = str(row.get("qtype") or "").lower()
    q_form = question_form(q)
    a_form = answer_form(a)
    signals = [f"qtype={qtype or 'missing'}", f"question_form={q_form}", f"answer_form={a_form}"]

    if (
        qtype in {"count", "state_compare", "state_describe", "tool_identify", "identity", "subjective"}
        or q_form == "scene_state"
        or a_form in {"quantity", "yes_no", "entity"}
    ):
        if not (q_form == "spatial" and qtype not in {"count", "subjective"}):
            return "tool_perception_state_grounding", signals

    # Spatial intelligence is a standalone research track because future
    # benchmark expansion will add left/right/depth/layout questions here.
    # We avoid swallowing causal "why" questions unless they explicitly depend
    # on spatial geometry.
    if q_form == "spatial" or a_form == "spatial_relation" or qtype == "spatial":
        if not (has(FUNCTIONAL_PATTERNS, q.lower()) and not has(SPATIAL_PATTERNS, a.lower())):
            return "spatial_intelligence", signals

    if qtype in {"causal_why", "purpose_what_for"} or q_form == "causal_affordance" or a_form == "causal_or_purpose":
        return "functional_affordance_causality", signals

    if qtype in {"temporal", "procedural_how"} or q_form == "procedural_temporal":
        return "procedural_temporal_dynamics", signals

    if has(PROCEDURAL_TEMPORAL_PATTERNS, text):
        return "procedural_temporal_dynamics", signals + ["fallback=procedure_or_time_mention"]
    if has(SCENE_STATE_PATTERNS, text):
        return "tool_perception_state_grounding", signals + ["fallback=scene_state_mention"]
    return "tool_perception_state_grounding", signals + ["fallback=general_scene_grounding"]


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["research_track_id"]].append(row)

    summary = {}
    for track_id, items in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        summary[track_id] = {
            "track_name": TRACKS[track_id]["name"],
            "definition": TRACKS[track_id]["definition"],
            "count": len(items),
            "status_counts": dict(Counter(r.get("review_status") for r in items)),
            "manual_review_counts": dict(Counter(str(bool(r.get("manual_reviewed"))).lower() for r in items)),
            "capability_axis_counts": dict(Counter(r.get("tool_capability_axis") or "none" for r in items).most_common()),
            "subtrack_counts": dict(Counter(r.get("research_subtrack_id") or "none" for r in items).most_common()),
            "qtype_counts": dict(Counter(r.get("qtype") or "missing" for r in items).most_common()),
            "question_form_counts": dict(Counter(r.get("research_question_form") for r in items).most_common()),
            "answer_form_counts": dict(Counter(r.get("research_answer_form") for r in items).most_common()),
            "examples": [
                {
                    "qa_id": r.get("qa_id"),
                    "qtype": r.get("qtype"),
                    "question": r.get("question"),
                    "answer": r.get("answer"),
                }
                for r in items[:8]
            ],
        }

    functional_rows = groups.get("functional_affordance_causality", [])
    functional_subtrack_summary = {}
    for subtrack_id, items in sorted(
        defaultdict(
            list,
            {
                subtrack_id: [
                    row for row in functional_rows if row.get("research_subtrack_id") == subtrack_id
                ]
                for subtrack_id in TRACK_SUBTRACKS["functional_affordance_causality"]
            },
        ).items(),
        key=lambda kv: (-len(kv[1]), kv[0]),
    ):
        if not items:
            continue
        functional_subtrack_summary[subtrack_id] = {
            "subtrack_name": SUBTRACKS[subtrack_id]["name"],
            "definition": SUBTRACKS[subtrack_id]["definition"],
            "count": len(items),
            "status_counts": dict(Counter(r.get("review_status") for r in items)),
            "qtype_counts": dict(Counter(r.get("qtype") or "missing" for r in items).most_common()),
            "examples": [
                {
                    "qa_id": r.get("qa_id"),
                    "qtype": r.get("qtype"),
                    "question": r.get("question"),
                    "answer": r.get("answer"),
                }
                for r in items[:8]
            ],
        }
    return {
        "total": len(rows),
        "track_count": len(groups),
        "track_definitions": TRACKS,
        "subtrack_definitions": SUBTRACKS,
        "track_subtracks": TRACK_SUBTRACKS,
        "capability_axis_definitions": CAPABILITY_AXES,
        "track_counts": dict(Counter(r["research_track_id"] for r in rows).most_common()),
        "capability_axis_counts": dict(Counter(r.get("tool_capability_axis") or "none" for r in rows).most_common()),
        "subtrack_counts": dict(Counter(r.get("research_subtrack_id") or "none" for r in rows).most_common()),
        "functional_subtrack_counts": dict(
            Counter(
                r.get("research_subtrack_id")
                for r in functional_rows
                if r.get("research_subtrack_id")
            ).most_common()
        ),
        "track_summary": summary,
        "functional_subtrack_summary": functional_subtrack_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--exclude-dropped", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    out_rows = []
    for row in rows:
        if args.exclude_dropped and row.get("is_dropped"):
            continue
        row = dict(row)
        track_id, signals = assign_track(row)
        row["research_track_id"] = track_id
        row["research_track_name"] = TRACKS[track_id]["name"]
        row["research_question_form"] = question_form(str(row.get("question") or ""))
        row["research_answer_form"] = answer_form(str(row.get("answer") or ""))
        row["research_track_assignment_signals"] = signals
        subtrack_id, subtrack_signals = assign_subtrack(row, track_id)
        axis_id = assign_capability_axis(track_id, subtrack_id)
        row["tool_capability_axis"] = axis_id
        row["tool_capability_axis_name"] = CAPABILITY_AXES[axis_id]["name"] if axis_id else None
        row["tool_capability_axis_definition"] = CAPABILITY_AXES[axis_id]["definition"] if axis_id else None
        if subtrack_id:
            row["research_subtrack_id"] = subtrack_id
            row["research_subtrack_name"] = SUBTRACKS[subtrack_id]["name"]
            row["research_subtrack_definition"] = SUBTRACKS[subtrack_id]["definition"]
            row["research_subtrack_assignment_signals"] = subtrack_signals
        else:
            row["research_subtrack_id"] = None
            row["research_subtrack_name"] = None
            row["research_subtrack_definition"] = None
            row["research_subtrack_assignment_signals"] = []
        out_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = build_report(out_rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(out_rows)} rows to {args.output}")
    print(f"Wrote report to {args.report}")
    print(json.dumps(report["track_counts"], ensure_ascii=False, indent=2))
    print(json.dumps(report["functional_subtrack_counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
