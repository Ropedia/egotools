"""Smoke test for the `egotools_eval` conda env.

Prints versions of the core libs the eval pipeline depends on and exits 0
on success. Exits non-zero if any required import fails.

Usage:
    conda run -n egotools_eval python scripts/verify_eval.py
"""

from __future__ import annotations

import importlib
import platform
import sys

# Required: a missing one means the env is broken.
REQUIRED = [
    "egotools",
    "torch",
    "torchvision",
    "transformers",
    "huggingface_hub",
    "decord",
    "PIL",  # pillow
    "cv2",  # opencv-python
    "numpy",
    "pandas",
    "tqdm",
    "yaml",  # pyyaml
    "vlmeval",  # editable install of VLMEvalKit
]

# Optional: useful but not fatal if absent (e.g. accelerate sometimes lags).
OPTIONAL = [
    "accelerate",
    "einops",
    "sentencepiece",
    "matplotlib",
]


def _version(mod_name: str) -> str:
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:  # noqa: BLE001
        return f"IMPORT_ERROR ({type(exc).__name__}: {exc})"
    for attr in ("__version__", "VERSION", "version"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
        if v is not None:
            return str(v)
    return "unknown"


def _torch_cuda_summary() -> str | None:
    try:
        import torch  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001
        return f"torch import failed: {exc}"
    try:
        avail = torch.cuda.is_available()
        n = torch.cuda.device_count() if avail else 0
        names = [torch.cuda.get_device_name(i) for i in range(n)] if avail else []
        return (
            f"cuda_available={avail} device_count={n} "
            f"cuda_runtime={getattr(torch.version, 'cuda', None)} "
            f"devices={names}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"cuda probe failed: {exc}"


def main() -> int:
    print("=" * 72)
    print("egotools_eval :: verify_eval.py")
    print("=" * 72)
    print(f"python              : {sys.version.split()[0]} ({sys.executable})")
    print(f"platform            : {platform.platform()}")
    print("-" * 72)

    print("[required]")
    failures = []
    for name in REQUIRED:
        v = _version(name)
        marker = "OK " if not v.startswith("IMPORT_ERROR") else "FAIL"
        print(f"  [{marker}] {name:<20} {v}")
        if v.startswith("IMPORT_ERROR"):
            failures.append(name)

    print("[optional]")
    for name in OPTIONAL:
        v = _version(name)
        marker = "OK " if not v.startswith("IMPORT_ERROR") else "skip"
        print(f"  [{marker}] {name:<20} {v}")

    print("-" * 72)
    print(f"torch cuda          : {_torch_cuda_summary()}")
    print("=" * 72)

    if failures:
        print(f"FAILED required imports: {failures}", file=sys.stderr)
        return 1
    print("All required imports OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
