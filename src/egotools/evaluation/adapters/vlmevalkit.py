"""Register the EgoTools dataset and compatibility options in VLMEvalKit.

Registration changes in-memory registries only. The upstream checkout remains
unmodified. Qwen adapters use Transformers when vLLM is absent, with SDPA
attention unless EGOTOOLS_QWEN3VL_ATTN / EGOTOOLS_QWEN2VL_ATTN overrides it.
"""

from __future__ import annotations

import importlib
import os
import warnings
from collections.abc import Iterable
from functools import partial

from egotools.evaluation.datasets.egotools import EGOTOOLS_VERSIONS, EgotoolsBench

_REGISTERED = False
_QWEN3_VL_BACKEND_PATCHED = False
_QWEN3_VL_ATTN_PATCHED = False
_QWEN2_VL_ATTN_PATCHED = False
_PYAV_COMPAT_PATCHED = False
_EXTRA_MODEL_ALIASES_REGISTERED = False


def _video_variants() -> dict[str, partial]:
    """Build the {alias: partial(EgotoolsBench, ...)} dict.

    Each alias bakes in (version, video_mode, nframe). We provide a
    handful of common nframe settings to mirror the videomme_dataset
    pattern in upstream ``video_dataset_config.py``.
    """
    variants: dict[str, partial] = {}
    nframe_choices = sorted({8, 32, 64, int(os.environ.get("EGOTOOLS_NFRAME", "64"))} - {0})
    versions = set(EGOTOOLS_VERSIONS) | {os.environ.get("EGOTOOLS_BENCH_VERSION", "custom")}
    for version in sorted(versions):
        for mode in ("full", "clip"):
            base_alias = f"EgotoolsBench_{version}_{mode}"
            # default (no frame split -> raw video to a video-LLM)
            variants[base_alias] = partial(
                EgotoolsBench,
                dataset=base_alias,
                version=version,
                video_mode=mode,
            )
            for n in nframe_choices:
                variants[f"{base_alias}_{n}frame"] = partial(
                    EgotoolsBench,
                    dataset=base_alias,
                    version=version,
                    video_mode=mode,
                    nframe=n,
                )
            for fps in {1.0, float(os.environ.get("EGOTOOLS_FPS", "1"))}:
                if fps > 0:
                    variants[f"{base_alias}_{fps:g}fps"] = partial(
                        EgotoolsBench,
                        dataset=base_alias,
                        version=version,
                        video_mode=mode,
                        fps=fps,
                    )
    return variants


def _extend_unique(seq: list, items: Iterable) -> None:
    for item in items:
        if item not in seq:
            seq.append(item)


def register_egotools_bench(force: bool = False) -> bool:
    """Inject EgotoolsBench into VLMEvalKit's dataset registry.

    Returns True if registration ran (or already-was-registered), False
    only if vlmeval is not importable. ``force=True`` forces re-injection
    (useful in interactive sessions).
    """
    global _REGISTERED
    if _REGISTERED and not force:
        return True

    try:
        import vlmeval.dataset as _vds  # type: ignore
        import vlmeval.dataset.video_dataset_config as _vdc  # type: ignore
    except Exception as e:
        # Let the inference entry point report the missing upstream dependency.
        import warnings

        warnings.warn(
            f"vlmeval is not importable ({e}); EgotoolsBench will not be "
            "registered into VLMEvalKit.",
            stacklevel=2,
        )
        return False

    # 1. Top-level class registry.
    if hasattr(_vds, "DATASET_CLASSES") and EgotoolsBench not in _vds.DATASET_CLASSES:
        _vds.DATASET_CLASSES.append(EgotoolsBench)
    if hasattr(_vds, "VIDEO_DATASET") and EgotoolsBench not in _vds.VIDEO_DATASET:
        _vds.VIDEO_DATASET.append(EgotoolsBench)
    if hasattr(_vds, "SUPPORTED_DATASETS"):
        _extend_unique(_vds.SUPPORTED_DATASETS, [*EgotoolsBench.supported_datasets(), *_video_variants()])

    # 2. video_dataset_config.supported_video_datasets — this is what
    #    vlmeval.dataset.build_dataset() consults first for video benches.
    variants = _video_variants()
    if hasattr(_vdc, "supported_video_datasets"):
        for alias, factory in variants.items():
            _vdc.supported_video_datasets.setdefault(alias, factory)

    _REGISTERED = True

    # Preserve the evaluation recipe on hosts without optional vLLM / flash-attn.
    ensure_qwen3_vl_backend()
    ensure_qwen3_vl_attention()
    ensure_qwen2_vl_attention()
    ensure_extra_model_aliases()
    ensure_pyav_compat()
    configure_selected_model()

    return True


def ensure_extra_model_aliases(force: bool = False) -> bool:
    """Register EgoTools, MiMo, and Gemini while retaining upstream Omni audio."""
    global _EXTRA_MODEL_ALIASES_REGISTERED
    if _EXTRA_MODEL_ALIASES_REGISTERED and not force:
        return True

    try:
        from vlmeval import vlm  # type: ignore
        from vlmeval.config import supported_VLM  # type: ignore
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"vlmeval model registry not importable ({e}); skipping extra model aliases.", stacklevel=2)
        return False

    baseline = supported_VLM["Qwen3-VL-8B-Instruct"]
    supported_VLM["EgoTools-8B"] = partial(
        baseline.func,
        *baseline.args,
        **{**baseline.keywords, "model_path": "ropedia-ai/egotools-8b"},
    )

    # The upstream MiMo preset uses lmdeploy. This env does not install
    # lmdeploy, and Qwen2.5-VL transformers inference is already exercised by
    # the Qwen baselines, so force the lightweight transformers path here.
    mimo_factory = partial(
        vlm.Qwen2VLChat,
        model_path="XiaomiMiMo/MiMo-VL-7B-SFT",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
        use_lmdeploy=False,
        use_vllm=False,
    )
    supported_VLM["MiMo-VL-7B-SFT"] = mimo_factory
    supported_VLM["MiMo-VL-7B-SFT-64f"] = mimo_factory
    supported_VLM["MiMo-VL-7B-RL"] = partial(
        vlm.Qwen2VLChat,
        model_path="XiaomiMiMo/MiMo-VL-7B-RL",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
        use_lmdeploy=False,
        use_vllm=False,
    )
    supported_VLM["MiMo-VL-7B-RL-64f"] = supported_VLM["MiMo-VL-7B-RL"]

    # The upstream Qwen2.5-Omni ForVideo preset enables synchronized video
    # audio, as required by the paper. Its input media must exclude narration.

    # These API presets were used by the development evaluation configuration.
    from vlmeval import api

    for model_key, remote_model in {
        "gemini-3-flash": "gemini-3-flash-preview",
        "gemini-3-flash-preview": "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite-preview",
        "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    }.items():
        supported_VLM.setdefault(
            model_key,
            partial(api.Gemini, model=remote_model, backend="genai", fps=1, temperature=1.0, retry=10),
        )

    _EXTRA_MODEL_ALIASES_REGISTERED = True
    return True


def configure_selected_model() -> None:
    """Apply explicit model path and generation options to the selected preset."""
    model_key = os.environ.get("EGOTOOLS_MODEL_KEY")
    if not model_key:
        return
    from vlmeval.config import supported_VLM

    if model_key not in supported_VLM:
        raise ValueError(f"unknown VLMEvalKit model preset: {model_key}")
    factory = supported_VLM[model_key]
    model_path = os.environ.get("EGOTOOLS_MODEL_PATH")
    if model_path:
        if not isinstance(factory, partial):
            raise ValueError(f"model preset {model_key} does not accept a model-path override")
        kwargs = dict(factory.keywords or {})
        kwargs["model_path"] = model_path
        supported_VLM[model_key] = partial(factory.func, *factory.args, **kwargs)


def ensure_pyav_compat() -> bool:
    """Restore the ``av.AVError`` alias expected by older torchvision.

    PyAV 17 removed the top-level ``av.AVError`` name, while the torchvision
    video reader bundled in our eval env still catches ``av.AVError``. When a
    corrupted H264 stream triggers a PyAV decoding exception, Python evaluates
    the missing exception attribute and raises ``AttributeError`` instead of
    letting torchvision handle the decode failure. Re-adding the alias keeps
    the existing fallback path working without pinning PyAV globally.
    """
    global _PYAV_COMPAT_PATCHED
    if _PYAV_COMPAT_PATCHED:
        return True

    try:
        import av  # type: ignore
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"PyAV not importable ({e}); skipping PyAV compatibility patch.", stacklevel=2)
        return False

    if not hasattr(av, "AVError"):
        fallback = getattr(getattr(av, "error", None), "FFmpegError", None)
        if fallback is None:
            fallback = getattr(getattr(av, "error", None), "OSError", Exception)
        av.AVError = fallback
        warnings.warn(
            "[egotools.vlmevalkit] patched PyAV compatibility: restored av.AVError "
            f"as {fallback.__module__}.{fallback.__name__}.",
            stacklevel=2,
        )

    _PYAV_COMPAT_PATCHED = True
    return True


def ensure_qwen3_vl_backend(force_transformers: bool | None = None) -> bool:
    """Make sure Qwen3-VL entries in ``supported_VLM`` use a backend we have.

    By default we probe whether ``vllm`` is importable; if not, we mutate
    every ``Qwen3-VL-*`` partial in ``supported_VLM`` to add/override
    ``use_vllm=False`` so it falls through to the transformers path.

    Pass ``force_transformers=True`` to unconditionally flip the flag (e.g.
    for hosts where vllm imports but doesn't actually work for these
    weights). Pass ``force_transformers=False`` to no-op even if vllm is
    missing.

    Returns True iff the patch ran (or was already applied).
    """
    global _QWEN3_VL_BACKEND_PATCHED
    if _QWEN3_VL_BACKEND_PATCHED:
        return True

    try:
        from vlmeval.config import supported_VLM  # type: ignore
    except Exception as e:  # noqa: BLE001
        warnings.warn(
            f"vlmeval.config not importable ({e}); skipping Qwen3-VL backend "
            "patch. Real-model runs will not work in this process.",
            stacklevel=2,
        )
        return False

    if force_transformers is None:
        try:
            importlib.import_module("vllm")
            vllm_present = True
        except Exception:
            vllm_present = False
        force_transformers = not vllm_present

    if not force_transformers:
        _QWEN3_VL_BACKEND_PATCHED = True
        return True

    patched = []
    for name, factory in list(supported_VLM.items()):
        if not name.startswith("Qwen3-VL"):
            continue
        if not isinstance(factory, partial):
            continue
        # Only touch partials that target Qwen3VLChat. Avoids accidentally
        # rewriting unrelated entries that share the prefix.
        target = getattr(factory, "func", None)
        target_name = getattr(target, "__name__", "")
        if target_name != "Qwen3VLChat":
            continue
        new_kw = dict(factory.keywords or {})
        if new_kw.get("use_vllm") is False:
            continue  # already correct
        new_kw["use_vllm"] = False
        supported_VLM[name] = partial(factory.func, *factory.args, **new_kw)
        patched.append(name)

    _QWEN3_VL_BACKEND_PATCHED = True
    if patched:
        warnings.warn(
            "[egotools.vlmevalkit] vllm not importable; forced use_vllm=False on "
            f"{len(patched)} Qwen3-VL preset(s): {patched[:6]}{'...' if len(patched) > 6 else ''}",
            stacklevel=2,
        )
    return True


def ensure_qwen3_vl_attention() -> bool:
    """Patch upstream Qwen3VLChat to avoid requiring flash-attn.

    The pinned VLMEvalKit Qwen3-VL adapter hard-codes
    ``attn_implementation='flash_attention_2'`` in the transformers fallback
    path. Our ``egotools_eval`` env intentionally does not require flash-attn,
    so we rewrite that single from_pretrained kwarg at runtime. The default is
    PyTorch SDPA; override with ``EGOTOOLS_QWEN3VL_ATTN=eager`` if needed.
    """
    global _QWEN3_VL_ATTN_PATCHED
    if _QWEN3_VL_ATTN_PATCHED:
        return True

    try:
        from vlmeval.vlm.qwen3_vl.model import Qwen3VLChat  # type: ignore
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"Qwen3VLChat not importable ({e}); skipping attention patch.", stacklevel=2)
        return False

    if getattr(Qwen3VLChat, "_egotools_attn_patched", False):
        _QWEN3_VL_ATTN_PATCHED = True
        return True

    orig_init = Qwen3VLChat.__init__

    def patched_init(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        from transformers import AutoModelForImageTextToText

        original_from_pretrained = AutoModelForImageTextToText.from_pretrained
        attn_impl = os.environ.get("EGOTOOLS_QWEN3VL_ATTN", "sdpa")

        def patched_from_pretrained(cls, *fp_args, **fp_kwargs):  # noqa: ANN001, ANN002, ANN003
            if fp_kwargs.get("attn_implementation") == "flash_attention_2":
                fp_kwargs["attn_implementation"] = attn_impl
            return original_from_pretrained(*fp_args, **fp_kwargs)

        AutoModelForImageTextToText.from_pretrained = classmethod(patched_from_pretrained)
        try:
            return orig_init(self, *args, **kwargs)
        finally:
            AutoModelForImageTextToText.from_pretrained = original_from_pretrained

    Qwen3VLChat.__init__ = patched_init
    Qwen3VLChat._egotools_attn_patched = True
    _QWEN3_VL_ATTN_PATCHED = True
    warnings.warn(
        "[egotools.vlmevalkit] patched Qwen3-VL transformers fallback to use "
        f"attn_implementation={os.environ.get('EGOTOOLS_QWEN3VL_ATTN', 'sdpa')!r} "
        "instead of flash_attention_2.",
        stacklevel=2,
    )
    return True


def ensure_qwen2_vl_attention() -> bool:
    """Patch upstream Qwen2/Qwen2.5-VL adapter to avoid requiring flash-attn."""
    global _QWEN2_VL_ATTN_PATCHED
    if _QWEN2_VL_ATTN_PATCHED:
        return True

    try:
        import transformers
        from vlmeval.vlm.qwen2_vl.model import Qwen2VLChat  # type: ignore
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"Qwen2VLChat not importable ({e}); skipping attention patch.", stacklevel=2)
        return False

    if getattr(Qwen2VLChat, "_egotools_attn_patched", False):
        _QWEN2_VL_ATTN_PATCHED = True
        return True

    orig_init = Qwen2VLChat.__init__

    def patched_init(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        originals = {}
        attn_impl = os.environ.get("EGOTOOLS_QWEN2VL_ATTN", "sdpa")
        env_max_new_tokens = os.environ.get("EGOTOOLS_QWEN2VL_MAX_NEW_TOKENS")
        if env_max_new_tokens:
            kwargs["max_new_tokens"] = int(env_max_new_tokens)

        def patch_model_cls(cls_name: str) -> None:
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                return
            original = cls.from_pretrained
            originals[cls] = original

            def patched_from_pretrained(inner_cls, *fp_args, **fp_kwargs):  # noqa: ANN001, ANN002, ANN003
                if fp_kwargs.get("attn_implementation") == "flash_attention_2":
                    fp_kwargs["attn_implementation"] = attn_impl
                gpu_memory = os.environ.get("EGOTOOLS_QWEN2VL_GPU_MEMORY")
                cpu_memory = os.environ.get("EGOTOOLS_QWEN2VL_CPU_MEMORY")
                if gpu_memory and "max_memory" not in fp_kwargs:
                    try:
                        import torch

                        max_memory: dict[int | str, str] = {i: gpu_memory for i in range(torch.cuda.device_count())}
                        if cpu_memory:
                            max_memory["cpu"] = cpu_memory
                        fp_kwargs["max_memory"] = max_memory
                    except Exception as e:  # noqa: BLE001
                        warnings.warn(f"[egotools.vlmevalkit] failed to set Qwen2 max_memory: {e}", stacklevel=2)
                return original(*fp_args, **fp_kwargs)

            cls.from_pretrained = classmethod(patched_from_pretrained)

        for cls_name in (
            "Qwen2VLForConditionalGeneration",
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen2_5OmniForConditionalGeneration",
        ):
            patch_model_cls(cls_name)

        try:
            return orig_init(self, *args, **kwargs)
        finally:
            for cls, original in originals.items():
                cls.from_pretrained = original

    Qwen2VLChat.__init__ = patched_init
    Qwen2VLChat._egotools_attn_patched = True
    _QWEN2_VL_ATTN_PATCHED = True
    warnings.warn(
        "[egotools.vlmevalkit] patched Qwen2/Qwen2.5-VL transformers fallback to use "
        f"attn_implementation={os.environ.get('EGOTOOLS_QWEN2VL_ATTN', 'sdpa')!r} "
        "instead of flash_attention_2.",
        stacklevel=2,
    )
    return True


__all__ = [
    "register_egotools_bench",
    "ensure_qwen3_vl_backend",
    "ensure_qwen3_vl_attention",
    "ensure_qwen2_vl_attention",
    "ensure_pyav_compat",
]
