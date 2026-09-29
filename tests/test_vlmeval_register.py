from __future__ import annotations

import sys
import types
from functools import partial

import pandas as pd
import pytest

from egotools.evaluation.adapters import vlmevalkit as register
from egotools.evaluation.datasets.egotools import EgotoolsBench, build_mcq_prompt_text, extract_letter_ah


def test_pyav_compat_restores_exception_alias(monkeypatch):
    class FFmpegError(Exception):
        pass

    av = types.ModuleType("av")
    av.error = types.SimpleNamespace(FFmpegError=FFmpegError)
    monkeypatch.setitem(sys.modules, "av", av)
    monkeypatch.setattr(register, "_PYAV_COMPAT_PATCHED", False)
    with pytest.warns(UserWarning, match="restored av.AVError"):
        assert register.ensure_pyav_compat()
    assert av.AVError is FFmpegError


def test_arbitrary_dataset_label_and_frame_count_are_registered(monkeypatch):
    monkeypatch.setenv("EGOTOOLS_BENCH_VERSION", "my_benchmark")
    monkeypatch.setenv("EGOTOOLS_NFRAME", "16")
    monkeypatch.setenv("EGOTOOLS_FPS", "0.5")
    variants = register._video_variants()
    factory = variants["EgotoolsBench_my_benchmark_full_16frame"]
    assert factory.keywords["nframe"] == 16
    assert factory.keywords["version"] == "my_benchmark"
    assert "EgotoolsBench_my_benchmark_full_16frame" in EgotoolsBench.supported_datasets()
    assert variants["EgotoolsBench_my_benchmark_clip_0.5fps"].keywords["fps"] == 0.5


def test_local_checkpoint_changes_only_selected_preset(monkeypatch, tmp_path):
    config = types.ModuleType("vlmeval.config")
    factory = partial(dict, model_path="upstream/model", temperature=0.01)
    config.supported_VLM = {"selected": factory, "other": factory}
    monkeypatch.setitem(sys.modules, "vlmeval.config", config)
    monkeypatch.setenv("EGOTOOLS_MODEL_KEY", "selected")
    monkeypatch.setenv("EGOTOOLS_MODEL_PATH", str(tmp_path))
    register.configure_selected_model()
    assert config.supported_VLM["selected"].keywords == {"model_path": str(tmp_path), "temperature": 0.01}
    assert config.supported_VLM["other"].keywords["model_path"] == "upstream/model"


def test_prompt_keeps_original_option_letters_and_omits_empty_cells():
    prompt, letters = build_mcq_prompt_text(
        {"question": "Which tool?", "A": "hammer", "B": "pliers", "C": float("nan")}
    )
    assert letters == ["A", "B"]
    assert "A. hammer\nB. pliers" in prompt
    assert "one of: A, B" in prompt
    assert "C." not in prompt


def test_scoring_uses_shared_development_extraction_and_track_groups():
    df = pd.DataFrame(
        [
            {
                "prediction": "B, because the tool is held securely",
                "answer": "B",
                "research_track_id": "AC",
                "qtype": "affordance",
            },
            {"prediction": "no answer", "answer": "A", "research_track_id": "PG", "qtype": "grounding"},
        ]
    )
    metrics = EgotoolsBench.score_dataframe(df)
    assert extract_letter_ah(df.iloc[0]["prediction"]) == "B"
    assert metrics["overall"] == {"correct": 1, "total": 2, "accuracy": 0.5}
    assert metrics["per_track"]["AC"]["accuracy"] == 1.0
    assert metrics["extraction_coverage"]["rate"] == 0.5
    assert "score" not in df.columns
