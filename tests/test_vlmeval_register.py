from __future__ import annotations

import sys
import types
from functools import partial

import pandas as pd
import pytest

from egotools.evaluation.adapters import vlmevalkit as register
from egotools.evaluation.datasets import egotools as dataset_module
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


def test_alias_registration_preserves_baseline_settings_and_audio(monkeypatch, tmp_path):
    upstream = types.ModuleType("vlmeval")
    upstream.vlm = types.SimpleNamespace(Qwen2VLChat=dict)
    upstream.api = types.SimpleNamespace(Gemini=dict)
    config = types.ModuleType("vlmeval.config")
    omni = partial(dict, use_audio_in_video=True)
    baseline = partial(
        dict,
        model_path="Qwen/Qwen3-VL-8B-Instruct",
        use_vllm=False,
        use_custom_prompt=False,
        temperature=0.7,
        max_new_tokens=16384,
    )
    config.supported_VLM = {
        "Qwen2.5-Omni-7B-ForVideo": omni,
        "Qwen3-VL-8B-Instruct": baseline,
    }
    monkeypatch.setitem(sys.modules, "vlmeval", upstream)
    monkeypatch.setitem(sys.modules, "vlmeval.config", config)
    monkeypatch.setattr(register, "_EXTRA_MODEL_ALIASES_REGISTERED", False)

    assert register.ensure_extra_model_aliases()
    assert config.supported_VLM["Qwen2.5-Omni-7B-ForVideo"] is omni
    gemini = config.supported_VLM["gemini-3.1-pro-preview"].keywords
    assert gemini["backend"] == "genai"
    assert gemini["fps"] == 1

    egotools = config.supported_VLM["EgoTools-8B"]
    assert egotools.func is baseline.func
    assert egotools.args == baseline.args
    assert egotools.keywords == {**baseline.keywords, "model_path": "ropedia-ai/egotools-8b"}
    monkeypatch.setenv("EGOTOOLS_MODEL_KEY", "EgoTools-8B")
    monkeypatch.setenv("EGOTOOLS_MODEL_PATH", str(tmp_path))
    register.configure_selected_model()
    assert config.supported_VLM["EgoTools-8B"].keywords == {**baseline.keywords, "model_path": str(tmp_path)}
    assert config.supported_VLM["Qwen3-VL-8B-Instruct"] is baseline
    assert baseline.keywords["model_path"] == "Qwen/Qwen3-VL-8B-Instruct"


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


def test_dataset_preserves_manifest_text_after_upstream_type_inference(tmp_path, monkeypatch):
    from egotools.evaluation.metrics import aggregate_results

    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "index\tqa_id\tcanonical_video_id\tvideo\tquestion\tA\tB\tanswer\n"
        "001\t001\t0007\tvideos/0001.mp4\tWhich label?\t0001\t0002\tA\n"
        "NA\tN/A\tNULL\tvideos/0002.mp4\tWhich state?\tNA\tN/A\tB\n"
    )

    def upstream_init(self, *, dataset, pack, nframe, fps):
        prepared = self.prepare_dataset(dataset)
        self.data_root = prepared["root"]
        self.data_file = prepared["data_file"]
        self.data = pd.read_csv(self.data_file, sep="\t")
        # This is how the pinned VideoBaseDataset loads a manifest.
        assert self.data.loc[0, "index"] == 1
        assert pd.isna(self.data.loc[1, "index"])

    monkeypatch.setattr(dataset_module._VLMEvalVideoBase, "__init__", upstream_init)
    dataset = EgotoolsBench(dataset_root=str(tmp_path))
    assert dataset.data["index"].tolist() == ["001", "NA"]
    assert dataset.data["qa_id"].tolist() == ["001", "N/A"]
    assert dataset.data["canonical_video_id"].tolist() == ["0007", "NULL"]
    assert dataset.data["A"].tolist() == ["0001", "NA"]
    assert dataset.videos == ["videos/0001.mp4", "videos/0002.mp4"]
    prompt = dataset.build_prompt(dataset.data.iloc[0])[-1]["value"]
    assert "A. 0001\nB. 0002" in prompt

    # The upstream inference writer exports dataset.data as its result metadata.
    predictions = dataset.data.copy()
    predictions["prediction"] = ["A", "B"]
    predictions.to_excel(tmp_path / "Model_EgotoolsBench_custom_full.xlsx", index=False)
    results = aggregate_results(tmp_path, "Model", manifest_path=manifest)
    assert results["metrics"]["overall"] == {"correct": 2, "total": 2, "accuracy": 1.0}
    assert results["metrics"]["coverage"]["complete"] is True
