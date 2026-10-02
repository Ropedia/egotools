import csv

import pytest

from egotools.evaluation.inference import run_real_model


def make_inputs(tmp_path, *, missing_option=False):
    root = tmp_path / "benchmark"
    root.mkdir()
    (root / "video.mp4").write_bytes(b"video path fixture")
    row = {
        "index": "001",
        "qa_id": "question-1",
        "video": "video.mp4",
        "question": "Which tool?",
        "answer": "A",
        **{letter: f"Tool {letter}" for letter in "ABCDEFGH"},
    }
    if missing_option:
        row["H"] = ""
    with (root / "manifest.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "run.py").write_text("raise AssertionError('dry-run must not launch inference')")
    return root, upstream


@pytest.mark.parametrize(
    ("model", "alias"),
    [
        ("EgoTools-8B", "EgotoolsBench_custom_full_64frame"),
        ("Qwen3-VL-8B-Thinking", "EgotoolsBench_custom_full_512frame"),
        ("Gemini-3.1-Pro", "EgotoolsBench_custom_full_1fps"),
    ],
)
def test_direct_inference_uses_paper_sampling_defaults(tmp_path, capsys, model, alias):
    root, upstream = make_inputs(tmp_path)
    assert (
        run_real_model(
            model=model,
            bench_version="custom",
            video_mode="full",
            limit=0,
            dataset_root=str(root),
            vlmevalkit_dir=str(upstream),
            out_dir=tmp_path / "results",
            dry_run=True,
        )
        == 0
    )
    assert alias in capsys.readouterr().out
    assert not (tmp_path / "results").exists()


def test_paper_evaluation_rejects_missing_options_before_model_setup(tmp_path):
    root, _ = make_inputs(tmp_path, missing_option=True)
    with pytest.raises(ValueError, match="requires eight nonempty options"):
        run_real_model(
            model="EgoTools-8B",
            bench_version="custom",
            video_mode="full",
            limit=0,
            dataset_root=str(root),
            out_dir=tmp_path / "results",
        )
    assert not (tmp_path / "results").exists()
