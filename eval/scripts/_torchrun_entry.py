"""Bootstrap the EgoTools adapter before executing an upstream run.py."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    run_py = Path(os.environ["EGOTOOLS_VLMEVALKIT_RUN"])
    for path in (repo_root / "src", repo_root / "eval", repo_root, run_py.parent):
        sys.path.insert(0, str(path))
    inner_args = sys.argv[1:]
    if inner_args and inner_args[0] == "--":
        inner_args = inner_args[1:]
    sys.argv = [str(run_py), *inner_args]
    # The upstream script partitions CUDA_VISIBLE_DEVICES before importing
    # torch. Execute that setup before importing our dataset subclass, then
    # register it before main() constructs any datasets or models.
    upstream = runpy.run_path(str(run_py), run_name="egotools_vlmeval_runner")
    from vlmeval_ext.register import register_egotools_bench

    if not register_egotools_bench():
        raise RuntimeError("VLMEvalKit is not importable; run eval/setup/verify_env.py")
    upstream["load_env"]()
    upstream["main"]()


if __name__ == "__main__":
    main()
