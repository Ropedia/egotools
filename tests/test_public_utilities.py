from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from egotools.answers import extract_letter
from egotools.manifest import OPTION_LETTERS, read_manifest, validate_manifest
from egotools.prepare import prepare_manifest
from egotools.resources import download_resource, load_resources, resolve_resource
from egotools.score import score_predictions


def question(index=0):
    return {
        "index": index, "qa_id": f"question-{index}", "canonical_video_id": "source-video",
        "video": "videos/example.mp4", "question": "Which tool is used?",
        **dict.fromkeys(OPTION_LETTERS, ""), "A": "Small metal spoon with a round bowl", "B": "Wooden spatula with a flat head",
        "answer": "B", "research_track_id": "AC",
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize(("prediction", "letter"), [
    ("B", "B"), ("The answer is (b)", "B"), ("Final Answer: H", "H"),
    ("B because the blade is flat.", "B"), ("unknown", ""), (None, ""),
])
def test_development_answer_parser(prediction, letter):
    assert extract_letter(prediction) == letter


def test_option_text_fallback():
    assert extract_letter("Wooden spatula with a flat head", question()) == "B"


def test_variable_options_and_asset_validation(tmp_path):
    manifest = write_jsonl(tmp_path / "manifest.jsonl", [question()])
    assert validate_manifest(manifest)["valid"]
    assert not validate_manifest(manifest, asset_root=tmp_path, asset_mode="full")["valid"]
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos/example.mp4").write_bytes(b"synthetic media placeholder")
    assert validate_manifest(manifest, asset_root=tmp_path, asset_mode="full")["valid"]


@pytest.mark.parametrize("changes", [
    {"video": "../private.mp4"}, {"video": "/absolute.mp4"}, {"video": "C:\\videos\\example.mp4"},
    {"video": "https://example.com/video.mp4"}, {"qa_id": None}, {"canonical_video_id": ""},
    {"question": None}, {"answer": "H"}, {"B": None},
])
def test_rejects_invalid_manifest_rows(tmp_path, changes):
    manifest = write_jsonl(tmp_path / "manifest.jsonl", [question() | changes])
    assert not validate_manifest(manifest)["valid"]


def test_reindex_preserves_questions_and_scoring_metadata(tmp_path):
    original = [question(0) | {"index": 7}, question(1) | {"index": 7}]
    source = write_jsonl(tmp_path / "input.jsonl", original)
    assert not validate_manifest(source)["valid"]
    output = tmp_path / "manifest.tsv"
    assert prepare_manifest(source, output, reindex=True) == 2
    assert validate_manifest(output, expected_rows=2)["valid"]
    _, prepared = read_manifest(output)
    assert [row["index"] for row in prepared] == ["0", "1"]
    for before, after in zip(original, prepared, strict=True):
        assert {key: value for key, value in before.items() if key != "index"} == {
            key: value for key, value in after.items() if key != "index"
        }
    assert read_manifest(source)[1] == original


def test_reindex_rejects_duplicate_question_ids(tmp_path):
    source = write_jsonl(tmp_path / "input.jsonl", [question(), question()])
    with pytest.raises(ValueError, match="duplicate qa_id"):
        prepare_manifest(source, tmp_path / "out.tsv", reindex=True)
    assert not (tmp_path / "out.tsv").exists()


def test_scoring_uses_gold_metadata_and_option_text(tmp_path):
    manifest = write_jsonl(tmp_path / "gold.jsonl", [question(0), question(1)])
    predictions = write_jsonl(tmp_path / "predictions.jsonl", [
        {"qa_id": "question-0", "prediction": "Wooden spatula with a flat head", "answer": "A", "research_track_id": "forged"},
        {"index": 1, "prediction": "unknown"},
    ])
    metrics, rows = score_predictions(predictions, manifest_path=manifest)
    assert metrics["overall"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert metrics["coverage"] == {"predicted": 2, "expected": 2, "complete": True}
    assert metrics["per_track"]["AC"]["total"] == 2
    assert rows[0]["answer"] == "B"


def test_partial_results_are_explicit(tmp_path):
    manifest = write_jsonl(tmp_path / "gold.jsonl", [question(0), question(1)])
    predictions = write_jsonl(tmp_path / "predictions.jsonl", [{"qa_id": "question-0", "prediction": "B"}])
    with pytest.raises(ValueError, match="1 of 2"):
        score_predictions(predictions, manifest_path=manifest)
    metrics, _ = score_predictions(predictions, manifest_path=manifest, allow_partial=True)
    assert metrics["coverage"]["complete"] is False


def test_null_identifier_falls_back_without_losing_index_zero(tmp_path):
    manifest = write_jsonl(tmp_path / "gold.jsonl", [question(0), question(1)])
    predictions = write_jsonl(tmp_path / "predictions.jsonl", [
        {"qa_id": None, "index": 0, "prediction": "B"},
        {"qa_id": "question-1", "index": None, "prediction": "B"},
    ])
    metrics, _ = score_predictions(predictions, manifest_path=manifest)
    assert metrics["overall"]["correct"] == 2


@pytest.mark.parametrize("predictions", [
    [{"qa_id": "unknown", "index": 0, "prediction": "B"}],
    [{"qa_id": "question-0", "index": 1, "prediction": "B"}],
    [{"qa_id": "question-0", "prediction": "B"}] * 2,
])
def test_scoring_rejects_ambiguous_joins(tmp_path, predictions):
    manifest = write_jsonl(tmp_path / "gold.jsonl", [question(0), question(1)])
    path = write_jsonl(tmp_path / "predictions.jsonl", predictions)
    with pytest.raises(ValueError):
        score_predictions(path, manifest_path=manifest, allow_partial=True)


def test_empty_tsv_is_not_a_zero_accuracy_run(tmp_path):
    path = tmp_path / "predictions.tsv"
    path.write_text("index\tprediction\n")
    with pytest.raises(ValueError, match="empty manifest"):
        score_predictions(path)


def test_resource_config_controls_real_download_arguments(tmp_path):
    config = tmp_path / "resources.yaml"
    config.write_text("demo:\n  repo_id: example/dataset\n  repo_type: dataset\n  subdir: benchmark\n")
    resource = resolve_resource("demo", config=config)
    (tmp_path / "benchmark").mkdir()
    with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as download:
        assert download_resource(resource, output_dir=tmp_path, revision="v2", metadata_only=True) == tmp_path / "benchmark"
    assert download.call_args.kwargs["repo_id"] == "example/dataset"
    assert download.call_args.kwargs["revision"] == "v2"
    assert "benchmark/*.tsv" in download.call_args.kwargs["allow_patterns"]


def test_training_download_includes_added_media():
    preview = Path(__file__).resolve().parents[1] / "configs" / "resources.preview.yaml"
    assert "videos/train/**" in load_resources(preview)["training"].include_patterns
    resource = resolve_resource("training", repo_id="example/final", subdir="")
    assert resource.subdir is None
    assert resource.include_patterns == ()


def test_unconfigured_resources_do_not_download_or_create_output(tmp_path):
    for name in ("benchmark", "training", "model"):
        resource = resolve_resource(name, repo_id="")
        output = tmp_path / name
        with patch("huggingface_hub.snapshot_download") as download:
            with pytest.raises(ValueError, match="not configured"):
                download_resource(resource, output_dir=output)
        download.assert_not_called()
        assert not output.exists()
