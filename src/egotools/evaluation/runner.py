"""Evaluate EgoTools with a model through external VLMEvalKit."""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from egotools.evaluation.inference import run_real_model
from egotools.evaluation.models import default_sampling, normalize_model_name

_RESULTS_DIR = Path("outputs/evaluation")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_run_id(bench_version: str, model: str) -> str:
    return f"{bench_version}_{re.sub(r'[^a-zA-Z0-9_.-]+', '-', model)}_{_utc_stamp()}"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="egotools-evaluate",
        description="Run the egotools VLM benchmark via VLMEvalKit.",
    )
    p.add_argument(
        "--bench-version",
        default="custom",
        help="Dataset label or version subfolder (default: custom)",
    )
    p.add_argument(
        "--video-mode",
        choices=("full", "clip"),
        default="full",
        help="Which video column to use from the manifest (default: full)",
    )
    p.add_argument(
        "--model",
        required=True,
        help="VLMEvalKit model preset (for example Qwen3-VL-8B-Instruct). "
        "Common shorthand like qwen3vl8binstruct is normalized.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap the number of QAs evaluated (0 = full bench).",
    )
    p.add_argument(
        "--results-dir",
        default=str(_RESULTS_DIR),
        help=f"Where to write run results (default: {_RESULTS_DIR}).",
    )
    p.add_argument(
        "--dataset-root",
        default=None,
        help="Override the dataset root path. This can point either to "
        "the concrete version dir containing manifest.tsv, or to the "
        "parent dir containing <version>/manifest.tsv. "
        "Required unless EGOTOOLS_BENCH_ROOT is set.",
    )
    p.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="If >1, wrap real-model runs in torchrun --nproc_per_node=N "
        "for per-rank dataset sharding (default: 1, single-process).",
    )
    p.add_argument(
        "--hf-model-id",
        default=None,
        help="Explicit Hugging Face model repo id to pre-download before a "
        "real-model run. Defaults to the known repo id for supported model keys.",
    )
    p.add_argument(
        "--no-auto-download-model",
        action="store_true",
        help="Do not download a missing Hugging Face model snapshot. The run "
        "will fail if the model is not already cached.",
    )
    sampling = p.add_mutually_exclusive_group()
    sampling.add_argument(
        "--nframe", type=int, default=None,
        help="Override uniform frame count; defaults: Qwen3 Thinking 512, Gemini 1 FPS, other models 64; 0 uses model defaults",
    )
    sampling.add_argument("--fps", type=float, default=None, help="Sample at this frame rate instead of a fixed count")
    p.add_argument("--vlmevalkit-dir", help="Upstream checkout containing run.py; also accepts VLMEVALKIT_DIR")
    p.add_argument("--model-path", help="Local checkpoint directory for the selected VLMEvalKit model preset")
    p.add_argument(
        "--dry-run", action="store_true", help="Check inputs and print the command without loading/downloading a model"
    )
    p.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args passed verbatim to VLMEvalKit's run.py (real-model runs only).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    if args.limit < 0 or args.nproc_per_node < 1:
        parser.error("--limit must be nonnegative and --nproc-per-node must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.bench_version):
        parser.error("--bench-version must be a simple label containing letters, digits, underscores, dots, or hyphens")
    if args.nframe is not None and args.nframe < 0:
        parser.error("--nframe must be nonnegative")
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be positive")
    args.model = normalize_model_name(args.model)
    if args.nframe is None and args.fps is None:
        args.nframe, args.fps = default_sampling(args.model)
    elif args.nframe is not None:
        args.fps = -1
    else:
        args.nframe = 0

    run_id = os.environ.get("EGOTOOLS_RUN_ID") or make_run_id(
        args.bench_version,
        args.model,
    )
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        parser.error("EGOTOOLS_RUN_ID must be a single directory name")
    out_dir = Path(args.results_dir).expanduser().resolve() / run_id
    print(f"[egotools-evaluate] run_id={run_id}")
    print(f"[egotools-evaluate] out_dir={out_dir}")

    return run_real_model(
        bench_version=args.bench_version,
        video_mode=args.video_mode,
        model=args.model,
        limit=args.limit,
        out_dir=out_dir,
        dataset_root=args.dataset_root,
        extra_args=args.extra,
        nproc_per_node=args.nproc_per_node,
        hf_model_id=args.hf_model_id,
        auto_download_model=not args.no_auto_download_model,
        nframe=args.nframe,
        fps=args.fps,
        vlmevalkit_dir=args.vlmevalkit_dir,
        model_path=args.model_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
