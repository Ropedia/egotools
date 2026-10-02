from __future__ import annotations

import json

import pandas as pd
import pytest

from egotools.evaluation.datasets.egotools import EgotoolsBench
from egotools.score import score_predictions, score_rows


@pytest.mark.parametrize("track_column", ["research_track_id", "research_track", "track"])
def test_file_and_dataframe_scoring_agree_with_option_text_and_missing_groups(tmp_path, track_column):
    rows = [
        {
            "qa_id": "option-text",
            "answer": " b ",
            "prediction": "Wooden spatula with a flat head",
            "A": "Small metal spoon with a round bowl",
            "B": "Wooden spatula with a flat head",
            "research_track_id": None,
            track_column: " AC ",
            "qtype": " affordance ",
        },
        {"qa_id": "unparsed", "answer": "A", "prediction": "unknown", track_column: None, "qtype": None},
        {"qa_id": "empty", "answer": "B", "prediction": None},
    ]
    path = tmp_path / "predictions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    frame = pd.DataFrame(rows)
    original_frame = frame.copy(deep=True)

    metrics, scored = score_predictions(path)
    assert EgotoolsBench.score_dataframe(frame) == metrics
    assert metrics["overall"] == {"correct": 1, "total": 3, "accuracy": 1 / 3}
    assert metrics["per_track"] == {
        "AC": {"correct": 1, "total": 1, "accuracy": 1.0},
        "_unknown": {"correct": 0, "total": 2, "accuracy": 0.0},
    }
    assert metrics["per_qtype"]["_unknown"]["total"] == 2
    assert metrics["extraction_coverage"]["rate"] == metrics["valid_predictions"]["rate"] == 1 / 3
    assert [row["predicted_letter"] for row in scored] == ["B", "", ""]
    assert [row["score"] for row in scored] == [row["correct"] for row in scored] == [1, 0, 0]
    pd.testing.assert_frame_equal(frame, original_frame)
    assert all("score" not in row for row in rows)


@pytest.mark.parametrize("answer", [None, "", float("nan"), "I", "AB"])
def test_both_scoring_entries_reject_invalid_gold(tmp_path, answer):
    rows = [{"answer": answer, "prediction": "A"}]
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="invalid gold answer"):
        score_predictions(path)
    with pytest.raises(ValueError, match="invalid gold answer"):
        EgotoolsBench.score_dataframe(pd.DataFrame(rows))


def test_nullable_columns_do_not_create_track_labels_or_valid_predictions():
    frame = pd.DataFrame(
        {
            "answer": pd.Series(["A", "B"], dtype="string"),
            "prediction": pd.Series([pd.NA, "B"], dtype="string"),
            "research_track_id": pd.Series([pd.NA, pd.NA], dtype="string"),
        }
    )
    metrics = EgotoolsBench.score_dataframe(frame)
    assert metrics["overall"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert metrics["extraction_coverage"]["extracted"] == 1
    assert metrics["per_qtype"] == {}
    assert "per_track" not in metrics


def test_shared_scoring_copies_rows_and_handles_lowercase_gold():
    original = [{"answer": " a ", "prediction": "A", "research_track_id": float("nan")}]
    metrics, scored = score_rows(original)
    assert metrics["overall"]["correct"] == scored[0]["correct"] == scored[0]["score"] == 1
    assert "per_track" not in metrics
    assert "predicted_letter" not in original[0]


def test_empty_predictions_are_not_a_zero_accuracy_run():
    with pytest.raises(ValueError, match="empty predictions"):
        score_rows([])
    with pytest.raises(ValueError, match="empty predictions"):
        EgotoolsBench.score_dataframe(pd.DataFrame(columns=["answer", "prediction"]))
