from __future__ import annotations

import csv
import sys
import types
from pathlib import Path

import pytest

from eval.scripts.run_eval import (
    ensure_hf_model_available,
    find_vlmevalkit_run,
    make_limited_dataset_root,
    normalize_model_name,
    resolve_hf_model_id,
    run_real_model,
    run_smoke,
)


def _dataset(tmp_path):
    root = tmp_path / "dataset"
    (root / "videos").mkdir(parents=True)
    (root / "videos" / "v0.mp4").write_bytes(b"mock video placeholder")
    with (root / "manifest.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["index", "video", "question", "A", "B", "answer", "qtype", "qa_id"], delimiter="\t"
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
                    answer="A",
                    qtype="other",
                    qa_id="qa0",
                ),
                dict(
                    index=1,
                    video="videos/v0.mp4",
                    question="second question",
                    A="one",
                    B="two",
                    answer="B",
                    qtype="other",
                    qa_id="qa1",
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


def test_mock_smoke_scores_all_selected_rows(tmp_path):
    root = _dataset(tmp_path)
    result = run_smoke(
        bench_version="custom",
        video_mode="full",
        limit=0,
        out_dir=tmp_path / "results",
        dataset_root=str(root),
        mock_mode="fixed",
        mock_letter="A",
    )
    assert result["n_evaluated"] == 2
    assert result["metrics"]["overall"]["accuracy"] == 0.5
    assert (tmp_path / "results" / "predictions_scored.tsv").is_file()


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


def test_real_runner_hands_off_to_external_process_and_aggregates(tmp_path, monkeypatch):
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

        pd.DataFrame([{"answer": "A", "prediction": "A"}]).to_excel(output, index=False)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr("eval.scripts.run_eval.subprocess.run", run)
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
            nproc_per_node=2,
        )
        == 0
    )
    assert seen["command"][:4] == [sys.executable, "-m", "torch.distributed.run", "--standalone"]
    assert seen["env"]["EGOTOOLS_MODEL_PATH"] == str(checkpoint)
    assert (Path(seen["env"]["EGOTOOLS_BENCH_ROOT"]) / "manifest.tsv").is_file()
    assert (tmp_path / "result" / "results.json").is_file()


def test_entry_initializes_upstream_before_importing_adapter(tmp_path, monkeypatch):
    from eval.scripts import _torchrun_entry

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
    registration = types.ModuleType("vlmeval_ext.register")

    def register_adapter():
        import os

        assert os.environ["UPSTREAM_READY"] == "1"
        os.environ["ADAPTER_READY"] = "1"
        return True

    registration.register_egotools_bench = register_adapter
    monkeypatch.setitem(sys.modules, "vlmeval_ext.register", registration)
    monkeypatch.setenv("EGOTOOLS_VLMEVALKIT_RUN", str(upstream))
    monkeypatch.delenv("ADAPTER_READY", raising=False)
    monkeypatch.setenv("UPSTREAM_READY", "0")
    monkeypatch.setenv("UPSTREAM_MAIN_RAN", "0")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["entry", "--", "--help"])
    _torchrun_entry.main()
    import os

    assert os.environ["UPSTREAM_MAIN_RAN"] == "1"
