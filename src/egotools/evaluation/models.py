"""Resolve model names and prepare Hugging Face checkpoints for evaluation."""

from __future__ import annotations

import os
import time
from pathlib import Path

_MODEL_ALIASES = {
    "egotools8b": "EgoTools-8B",
    "egotools-8b": "EgoTools-8B",
    "ropedia-ai/egotools-8b": "EgoTools-8B",
    "qwen3vl8binstruct": "Qwen3-VL-8B-Instruct",
    "qwen3-vl-8b-instruct": "Qwen3-VL-8B-Instruct",
    "qwen3_vl_8b_instruct": "Qwen3-VL-8B-Instruct",
    "qwen/qwen3-vl-8b-instruct": "Qwen3-VL-8B-Instruct",
    "qwen3vl4binstruct": "Qwen3-VL-4B-Instruct",
    "qwen3-vl-4b-instruct": "Qwen3-VL-4B-Instruct",
    "qwen3_vl_4b_instruct": "Qwen3-VL-4B-Instruct",
    "qwen/qwen3-vl-4b-instruct": "Qwen3-VL-4B-Instruct",
    "qwen3vl8bthinking": "Qwen3-VL-8B-Thinking",
    "qwen3-vl-8b-thinking": "Qwen3-VL-8B-Thinking",
    "qwen/qwen3-vl-8b-thinking": "Qwen3-VL-8B-Thinking",
    "qwen3vl4bthinking": "Qwen3-VL-4B-Thinking",
    "qwen3-vl-4b-thinking": "Qwen3-VL-4B-Thinking",
    "qwen/qwen3-vl-4b-thinking": "Qwen3-VL-4B-Thinking",
    "qwen25vl7binstruct": "Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl-7b-instruct": "Qwen2.5-VL-7B-Instruct",
    "qwen/qwen2.5-vl-7b-instruct": "Qwen2.5-VL-7B-Instruct",
    "qwen25omni7binstruct": "Qwen2.5-Omni-7B-ForVideo",
    "qwen2.5-omni-7b-instruct": "Qwen2.5-Omni-7B-ForVideo",
    "qwen25omni7b": "Qwen2.5-Omni-7B-ForVideo",
    "qwen2.5-omni-7b": "Qwen2.5-Omni-7B-ForVideo",
    "qwen/qwen2.5-omni-7b": "Qwen2.5-Omni-7B-ForVideo",
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-3-flash-preview": "gemini-3-flash-preview",
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite-preview",
    "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite-preview",
}

_MODEL_TO_HF_REPO = {
    "EgoTools-8B": "ropedia-ai/egotools-8b",
    "Qwen3-VL-8B-Instruct": "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen3-VL-4B-Instruct": "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen3-VL-8B-Thinking": "Qwen/Qwen3-VL-8B-Thinking",
    "Qwen3-VL-4B-Thinking": "Qwen/Qwen3-VL-4B-Thinking",
    "Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen2-VL-7B-Instruct": "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen2.5-Omni-7B-ForVideo": "Qwen/Qwen2.5-Omni-7B",
    "Qwen2.5-Omni-7B": "Qwen/Qwen2.5-Omni-7B",
}


def _model_alias_key(model: str) -> str:
    return model.strip().lower().replace(" ", "").replace("_", "-")


def normalize_model_name(model: str) -> str:
    """Return the VLMEvalKit model key for common shorthand spellings."""
    stripped = model.strip()
    if not stripped:
        return stripped
    if Path(stripped).expanduser().exists():
        return stripped
    return _MODEL_ALIASES.get(_model_alias_key(stripped), stripped)


def default_sampling(model: str) -> tuple[int, float]:
    """Return the paper's frame/FPS setting for the listed model variants."""
    model = normalize_model_name(model)
    if model in {"Qwen3-VL-8B-Thinking", "Qwen3-VL-4B-Thinking"}:
        return 512, -1
    if model in {"gemini-3-flash-preview", "gemini-3.1-pro-preview", "gemini-3.1-flash-lite-preview"}:
        return 0, 1.0
    return 64, -1


def resolve_hf_model_id(model: str, explicit_hf_model_id: str | None = None) -> str | None:
    """Map a VLMEvalKit model key to its Hugging Face repo id when known."""
    if explicit_hf_model_id:
        return explicit_hf_model_id
    stripped = model.strip()
    if not stripped or Path(stripped).expanduser().exists():
        return None
    if "/" in stripped and not stripped.startswith(("/", "./", "../")):
        return stripped
    return _MODEL_TO_HF_REPO.get(normalize_model_name(stripped))


def ensure_hf_model_available(
    *,
    model: str,
    hf_model_id: str | None = None,
    auto_download: bool = True,
) -> str | None:
    """Ensure a known HF-backed model snapshot exists in the local HF cache.

    Returns the resolved local snapshot path when a Hugging Face repo id is
    known, otherwise ``None``. The function first probes the local cache and
    only uses the network when the snapshot is missing and ``auto_download`` is
    enabled.
    """
    repo_id = resolve_hf_model_id(model, explicit_hf_model_id=hf_model_id)
    if repo_id is None:
        return None

    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo_id=repo_id, repo_type="model", local_files_only=True)
    except Exception as local_exc:
        if not auto_download:
            raise RuntimeError(f"model {repo_id} is not present in the local Hugging Face cache") from local_exc
        retries = max(1, int(os.environ.get("EGOTOOLS_HF_DOWNLOAD_RETRIES", "3")))
        max_workers = int(os.environ.get("EGOTOOLS_HF_MAX_WORKERS", "1"))
        last_exc: Exception | None = None
        print(
            f"[egotools-evaluate] model snapshot not found locally; downloading {repo_id} "
            f"(retries={retries}, max_workers={max_workers})",
            flush=True,
        )
        for attempt in range(1, retries + 1):
            try:
                return snapshot_download(
                    repo_id=repo_id,
                    repo_type="model",
                    max_workers=max_workers,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= retries:
                    break
                sleep_s = min(60, 10 * attempt)
                print(
                    f"[egotools-evaluate] download attempt {attempt}/{retries} failed: "
                    f"{type(exc).__name__}: {exc}; retrying in {sleep_s}s",
                    flush=True,
                )
                time.sleep(sleep_s)
        raise RuntimeError(
            f"failed to download {repo_id} after {retries} attempts. "
            "You can pre-download it with hf download, then rerun "
            "with HF_HUB_OFFLINE=1 or --no-auto-download-model."
        ) from last_exc
