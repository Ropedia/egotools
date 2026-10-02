from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from egotools.evaluation.inference import (
    find_vlmevalkit_run,
    make_limited_dataset_root,
    run_real_model,
)
from egotools.evaluation.models import (
    default_sampling,
    ensure_hf_model_available,
    normalize_model_name,
    resolve_hf_model_id,
)


def _dataset(tmp_path):
    root = tmp_path / "dataset"
    (root / "videos").mkdir(parents=True)
    (root / "videos" / "v0.mp4").write_bytes(b"mock video placeholder")
    with (root / "manifest.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "video", "question", *"ABCDEFGH", "answer", "qtype", "qa_id", "canonical_video_id"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(
            [
                dict(
                    index=0,
                    video="videos/v0.mp4",
                    question="A multiline\nquestion",
                    A="option 1",
                    B="option 2",
                    C="option 3",
                    D="option 4",
                    E="option 5",
                    F="option 6",
                    G="option 7",
                    H="option 8",
                    answer="A",
                    qtype="other",
                    qa_id="qa0",
                    canonical_video_id="video0",
                ),
                dict(
                    index=1,
                    video="videos/v0.mp4",
                    question="second question",
                    A="one",
                    B="two",
                    C="three",
                    D="four",
                    E="five",
                    F="six",
                    G="seven",
                    H="eight",
                    answer="B",
                    qtype="other",
                    qa_id="qa1",
                    canonical_video_id="video0",
                ),
            ]
        )
    return root


def test_normalize_model_name_accepts_common_spellings():
    for name in ["qwen3vl8binstruct", "qwen3-vl-8b-instruct", "Qwen3-VL-8B-Instruct"]:
        assert normalize_model_name(name) == "Qwen3-VL-8B-Instruct"


def test_known_model_repo_is_resolved():
    assert resolve_hf_model_id("Qwen3-VL-8B-Instruct") == "Qwen/Qwen3-VL-8B-Instruct"
    assert resolve_hf_model_id("Qwen/Qwen3-VL-8B-Instruct") == "Qwen/Qwen3-VL-8B-Instruct"
    assert resolve_hf_model_id("/models/local-checkpoint") is None
    assert resolve_hf_model_id("qwen3vl4bthinking") == "Qwen/Qwen3-VL-4B-Thinking"
    assert normalize_model_name("ropedia-ai/egotools-8b") == "EgoTools-8B"
    assert resolve_hf_model_id("EgoTools-8B") == "ropedia-ai/egotools-8b"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("EgoTools-8B", (64, -1)),
        ("Qwen3-VL-8B-Instruct", (64, -1)),
        ("Qwen3-VL-8B-Thinking", (512, -1)),
        ("Qwen/Qwen3-VL-4B-Thinking", (512, -1)),
        ("MiMo-VL-7B-RL", (64, -1)),
        ("GLM4_1VThinking-9b", (64, -1)),
        ("Gemini-3.1-Pro", (0, 1.0)),
        ("gemini-3-flash", (0, 1.0)),
        ("gemini-3.1-flash-lite-preview", (0, 1.0)),
    ],
)
def test_paper_sampling_defaults_depend_on_model_variant(model, expected):
    assert default_sampling(model) == expected


@pytest.mark.parametrize(
    ("model", "overrides", "expected"),
    [
        ("qwen3vl8bthinking", [], (512, -1)),
        ("gemini-3.1-pro", [], (0, 1.0)),
        ("Qwen3-VL-8B-Thinking", ["--nframe", "64"], (64, -1)),
        ("Qwen3-VL-8B-Instruct", ["--fps", "2"], (0, 2.0)),
    ],
)
def test_cli_preserves_explicit_sampling_overrides(monkeypatch, model, overrides, expected):
    from egotools.evaluation import runner

    seen = {}

    def run(**kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(runner, "run_real_model", run)
    assert runner.main(["--model", model, *overrides]) == 0
    assert (seen["nframe"], seen["fps"]) == expected


def test_cached_model_does_not_download(monkeypatch):
    calls = []
    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return "/cache/model"

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    assert ensure_hf_model_available(model="Qwen3-VL-8B-Instruct") == "/cache/model"
    assert calls == [{"repo_id": "Qwen/Qwen3-VL-8B-Instruct", "repo_type": "model", "local_files_only": True}]


def test_disabled_download_only_checks_cache(monkeypatch):
    hub = types.ModuleType("huggingface_hub")
    calls = []

    def missing(**kwargs):
        calls.append(kwargs)
        raise FileNotFoundError("not cached")

    hub.snapshot_download = missing
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    with pytest.raises(RuntimeError, match="not present"):
        ensure_hf_model_available(model="Qwen3-VL-8B-Instruct", auto_download=False)
    assert len(calls) == 1


def test_limited_dataset_preserves_multiline_rows_and_relative_assets(tmp_path, monkeypatch):
    root = _dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    limited = make_limited_dataset_root(dataset_root=str(root), limit=1, out_dir=Path("run"))
    with (limited / "manifest.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 1
    assert rows[0]["question"] == "A multiline\nquestion"
    assert (limited / rows[0]["video"]).is_file()


@pytest.mark.parametrize("copy_fallback", [False, True])
def test_limited_dataset_preserves_media_at_the_dataset_root(tmp_path, monkeypatch, copy_fallback):
    root = _dataset(tmp_path)
    source = root / "videos" / "v0.mp4"
    source.rename(root / "v0.mp4")
    manifest = root / "manifest.tsv"
    manifest.write_text(manifest.read_text().replace("videos/v0.mp4", "v0.mp4"))
    if copy_fallback:
        def unsupported_symlink(*args, **kwargs):
            raise OSError("symlinks unsupported")

        monkeypatch.setattr(Path, "symlink_to", unsupported_symlink)

    limited = make_limited_dataset_root(dataset_root=str(root), limit=1, out_dir=tmp_path / "run")
    assert (limited / "v0.mp4").read_bytes() == b"mock video placeholder"
    assert (limited / "v0.mp4").is_symlink() is not copy_fallback


def test_real_model_dry_run_uses_checkout_without_download_or_output(tmp_path, monkeypatch, capsys):
    root = _dataset(tmp_path)
    checkout = tmp_path / "upstream"
    checkout.mkdir()
    (checkout / "run.py").write_text("raise AssertionError('must not execute during dry-run')")
    monkeypatch.setenv("VLMEVALKIT_DIR", str(checkout))
    assert find_vlmevalkit_run() == checkout / "run.py"
    assert (
        run_real_model(
            bench_version="custom",
            video_mode="full",
            model="qwen3vl8binstruct",
            limit=1,
            out_dir=tmp_path / "result",
            dataset_root=str(root),
            dry_run=True,
        )
        == 0
    )
    assert not (tmp_path / "result").exists()
    assert "EgotoolsBench_custom_full_64frame" in capsys.readouterr().out


@pytest.mark.parametrize("nproc_per_node", [1, 2])
def test_real_runner_hands_off_to_external_process_and_aggregates(tmp_path, monkeypatch, nproc_per_node):
    root = _dataset(tmp_path)
    checkout = tmp_path / "upstream"
    checkout.mkdir()
    (checkout / "run.py").write_text("# external entry point")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    seen = {}

    def run(command, *, env, check):
        seen.update(command=command, env=env)
        output = tmp_path / "result" / "fake_EgotoolsBench_custom_full.xlsx"
        import pandas as pd

        pd.DataFrame([{"index": 0, "qa_id": "qa0", "answer": "A", "prediction": "A"}]).to_excel(output, index=False)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr("egotools.evaluation.inference.subprocess.run", run)
    assert (
        run_real_model(
            bench_version="custom",
            video_mode="full",
            model="fake",
            limit=1,
            out_dir=tmp_path / "result",
            dataset_root=str(root),
            vlmevalkit_dir=str(checkout),
            model_path=str(checkpoint),
            nproc_per_node=nproc_per_node,
        )
        == 0
    )
    if nproc_per_node == 1:
        assert seen["command"][:3] == [sys.executable, "-m", "egotools.evaluation._torchrun_entry"]
    else:
        assert seen["command"][:4] == [sys.executable, "-m", "torch.distributed.run", "--standalone"]
        module_index = seen["command"].index("--module")
        assert seen["command"][module_index + 1] == "egotools.evaluation._torchrun_entry"
    assert seen["env"]["EGOTOOLS_MODEL_PATH"] == str(checkpoint)
    assert (Path(seen["env"]["EGOTOOLS_BENCH_ROOT"]) / "manifest.tsv").is_file()
    results = json.loads((tmp_path / "result" / "results.json").read_text())
    assert results["metrics"]["coverage"] == {"predicted": 1, "expected": 1, "complete": True}


def test_entry_initializes_upstream_before_importing_adapter(tmp_path, monkeypatch):
    from egotools.evaluation import _torchrun_entry

    upstream = tmp_path / "run.py"
    upstream.write_text(
        "import os\n"
        "assert os.environ.get('ADAPTER_READY') is None\n"
        "os.environ['UPSTREAM_READY'] = '1'\n"
        "def load_env(): pass\n"
        "def main():\n"
        "    assert os.environ['ADAPTER_READY'] == '1'\n"
        "    os.environ['UPSTREAM_MAIN_RAN'] = '1'\n"
    )
    registration = types.ModuleType("egotools.evaluation.adapters.vlmevalkit")

    def register_adapter():
        assert os.environ["UPSTREAM_READY"] == "1"
        monkeypatch.setenv("ADAPTER_READY", "1")
        return True

    registration.register_egotools_bench = register_adapter
    monkeypatch.setitem(sys.modules, "egotools.evaluation.adapters.vlmevalkit", registration)
    monkeypatch.setenv("EGOTOOLS_VLMEVALKIT_RUN", str(upstream))
    monkeypatch.delenv("ADAPTER_READY", raising=False)
    monkeypatch.setenv("UPSTREAM_READY", "0")
    monkeypatch.setenv("UPSTREAM_MAIN_RAN", "0")
    original_path = list(sys.path)
    monkeypatch.setattr(sys, "argv", ["entry", "--", "--help"])
    _torchrun_entry.main()
    assert os.environ["UPSTREAM_MAIN_RAN"] == "1"
    assert sys.path == original_path


def test_installed_entry_runs_outside_checkout_without_early_torch_import(tmp_path):
    upstream = tmp_path / "run.py"
    upstream.write_text(
        "import sys, types\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'vlmeval' not in sys.modules\n"
        "assert 'egotools.evaluation.datasets.egotools' not in sys.modules\n"
        "registration = types.ModuleType('egotools.evaluation.adapters.vlmevalkit')\n"
        "def register():\n"
        "    registration.ready = True\n"
        "    return True\n"
        "registration.register_egotools_bench = register\n"
        "sys.modules[registration.__name__] = registration\n"
        "def load_env():\n"
        "    assert registration.ready\n"
        "def main():\n"
        "    assert sys.argv[1:] == ['--data', 'fake']\n"
        "    print('entry completed')\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-m", "egotools.evaluation._torchrun_entry", "--", "--data", "fake"],
        cwd=tmp_path,
        env={**os.environ, "EGOTOOLS_VLMEVALKIT_RUN": str(upstream)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "entry completed"


def test_aggregation_deduplicates_upstream_prediction_symlink(tmp_path):
    from egotools.evaluation.metrics import aggregate_results

    model_root = tmp_path / "Model"
    run_root = model_root / "run-1"
    run_root.mkdir(parents=True)
    predictions = run_root / "Model_EgotoolsBench_custom_full_64frame.tsv"
    predictions.write_text("index\tanswer\tprediction\n0\tA\tA\n1\tB\tA\n")
    (model_root / predictions.name).symlink_to(predictions.relative_to(model_root))

    results = aggregate_results(tmp_path, "Model")
    assert results["prediction_file"] == str(predictions.resolve())
    assert results["n_evaluated"] == 2
    assert results["metrics"]["overall"]["accuracy"] == 0.5
    assert (tmp_path / "results.json").is_file()


def test_aggregation_still_reports_two_distinct_prediction_tables(tmp_path):
    from egotools.evaluation.metrics import _locate_pred_file

    for run_name in ("run-1", "run-2"):
        run_root = tmp_path / "Model" / run_name
        run_root.mkdir(parents=True)
        (run_root / "Model_EgotoolsBench_custom_full_64frame.tsv").write_text("answer\tprediction\nA\tA\n")
    with pytest.raises(ValueError, match="multiple prediction files"):
        _locate_pred_file(tmp_path, "Model")


@pytest.mark.parametrize(
    ("rows", "error"),
    [
        ([{"index": 0, "qa_id": "qa0", "prediction": "A"}], "1 of 2"),
        (
            [{"index": 0, "qa_id": "qa0", "prediction": "A"}] * 2,
            "duplicate prediction",
        ),
        (
            [
                {"index": 1, "qa_id": "qa0", "prediction": "A"},
                {"index": 0, "qa_id": "qa1", "prediction": "B"},
            ],
            "conflicting qa_id and index",
        ),
    ],
)
def test_aggregation_requires_each_selected_question_exactly_once(tmp_path, rows, error):
    from egotools.evaluation.metrics import aggregate_results

    root = _dataset(tmp_path)
    predictions = tmp_path / "Model_EgotoolsBench_custom_full.jsonl"
    predictions.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match=error):
        aggregate_results(tmp_path, "Model", manifest_path=root / "manifest.tsv")
    assert not (tmp_path / "results.json").exists()
    assert not Path(str(predictions) + "_egotools_score.json").exists()


def test_aggregation_uses_manifest_gold_and_preserves_excel_string_ids(tmp_path):
    import pandas as pd

    from egotools.evaluation.metrics import aggregate_results

    root = _dataset(tmp_path)
    manifest = root / "manifest.tsv"
    manifest.write_text(manifest.read_text().replace("qa0", "001").replace("qa1", "002"))
    predictions = tmp_path / "Model_EgotoolsBench_custom_full.xlsx"
    pd.DataFrame(
        [
            {"index": 1, "qa_id": "002", "answer": "A", "prediction": "B"},
            {"index": 0, "qa_id": "001", "answer": "B", "prediction": "A"},
        ]
    ).to_excel(predictions, index=False)

    results = aggregate_results(tmp_path, "Model", manifest_path=manifest)
    assert results["metrics"]["overall"] == {"correct": 2, "total": 2, "accuracy": 1.0}
    assert results["metrics"]["coverage"] == {"predicted": 2, "expected": 2, "complete": True}
    assert results["metrics"]["per_qtype"]["other"]["total"] == 2
    assert json.loads(Path(str(predictions) + "_egotools_score.json").read_text()) == results["metrics"]


def test_aggregation_rejects_duplicate_identities_without_manifest(tmp_path):
    from egotools.evaluation.metrics import aggregate_results

    predictions = tmp_path / "Model_EgotoolsBench_custom_full.tsv"
    predictions.write_text("index\tanswer\tprediction\n0\tA\tA\n0\tA\tA\n")
    with pytest.raises(ValueError, match="duplicate index"):
        aggregate_results(tmp_path, "Model")
    assert not (tmp_path / "results.json").exists()
