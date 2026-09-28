#!/usr/bin/env python
"""Run EgoTools mock checks or model inference through external VLMEvalKit."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# ----------------------------------------------------------------------
# sys.path bootstrap so this script works either as
#   python eval/scripts/run_eval.py ...
# or as
#   python -m eval.scripts.run_eval ...
# without requiring a pip-install of the eval/ tree.
# ----------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]  # <repo>/eval/scripts/run_eval.py -> <repo>
_EVAL_ROOT = _REPO_ROOT / "eval"

for p in (_REPO_ROOT / "src", _EVAL_ROOT, _REPO_ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# The lightweight adapter also supports the CPU mock path.
from vlmeval_ext.egotools_dataset import (  # noqa: E402
    EgotoolsBench,
    build_mcq_prompt_text,
    extract_letter_ah,
    resolve_dataset_root,
)

# ----------------------------------------------------------------------
# Result paths
# ----------------------------------------------------------------------

_RESULTS_DIR = _EVAL_ROOT / "results"

_MODEL_ALIASES = {
    "qwen3vl8binstruct": "Qwen3-VL-8B-Instruct",
    "qwen3-vl-8b-instruct": "Qwen3-VL-8B-Instruct",
    "qwen3_vl_8b_instruct": "Qwen3-VL-8B-Instruct",
    "qwen/qwen3-vl-8b-instruct": "Qwen3-VL-8B-Instruct",
    "qwen3vl4binstruct": "Qwen3-VL-4B-Instruct",
    "qwen3-vl-4b-instruct": "Qwen3-VL-4B-Instruct",
    "qwen3_vl_4b_instruct": "Qwen3-VL-4B-Instruct",
    "qwen/qwen3-vl-4b-instruct": "Qwen3-VL-4B-Instruct",
    "qwen25vl7binstruct": "Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl-7b-instruct": "Qwen2.5-VL-7B-Instruct",
    "qwen/qwen2.5-vl-7b-instruct": "Qwen2.5-VL-7B-Instruct",
    "qwen25omni7binstruct": "Qwen2.5-Omni-7B-ForVideo",
    "qwen2.5-omni-7b-instruct": "Qwen2.5-Omni-7B-ForVideo",
    "qwen25omni7b": "Qwen2.5-Omni-7B-ForVideo",
    "qwen2.5-omni-7b": "Qwen2.5-Omni-7B-ForVideo",
    "qwen/qwen2.5-omni-7b": "Qwen2.5-Omni-7B-ForVideo",
}

_MODEL_TO_HF_REPO = {
    "Qwen3-VL-8B-Instruct": "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen3-VL-4B-Instruct": "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen2-VL-7B-Instruct": "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen2.5-Omni-7B-ForVideo": "Qwen/Qwen2.5-Omni-7B",
    "Qwen2.5-Omni-7B": "Qwen/Qwen2.5-Omni-7B",
}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_run_id(bench_version: str, model: str) -> str:
    return f"{bench_version}_{re.sub(r'[^a-zA-Z0-9_.-]+', '-', model)}_{_utc_stamp()}"


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
            f"[run_eval] model snapshot not found locally; downloading {repo_id} "
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
                    f"[run_eval] download attempt {attempt}/{retries} failed: "
                    f"{type(exc).__name__}: {exc}; retrying in {sleep_s}s",
                    flush=True,
                )
                time.sleep(sleep_s)
        raise RuntimeError(
            f"failed to download {repo_id} after {retries} attempts. "
            "You can pre-download it with hf download, then rerun "
            "with HF_HUB_OFFLINE=1 or --no-auto-download-model."
        ) from last_exc


def make_limited_dataset_root(*, dataset_root: str, limit: int, out_dir: Path) -> Path:
    """Create a small manifest-only dataset root for real-model smoke runs."""
    if limit <= 0:
        return Path(dataset_root).expanduser().resolve()
    src_root = Path(dataset_root).expanduser().resolve()
    src_manifest = src_root / "manifest.tsv"
    if not src_manifest.exists():
        raise FileNotFoundError(f"manifest.tsv not found under {src_root}")

    limited_root = out_dir.expanduser().resolve() / f"_limited_dataset_{limit}"
    limited_root.mkdir(parents=True, exist_ok=True)
    with src_manifest.open(newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src, delimiter="\t")
        rows = list(reader)
        with (limited_root / "manifest.tsv").open("w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows[:limit])

    asset_dirs = {
        Path(row.get(column, "")).parts[0]
        for row in rows[:limit]
        for column in ("video", "clip_video")
        if row.get(column) and len(Path(row[column]).parts) > 1
    }
    for name in asset_dirs | {"videos", "clips"}:
        source = src_root / name
        target = limited_root / name
        if not source.exists() or target.exists() or target.is_symlink():
            continue
        try:
            target.symlink_to(source, target_is_directory=True)
        except OSError:
            shutil.copytree(source, target, dirs_exist_ok=True)

    src_jsonl = src_root / "manifest.jsonl"
    if src_jsonl.exists():
        with src_jsonl.open() as src, (limited_root / "manifest.jsonl").open("w") as dst:
            for line_no, line in enumerate(src):
                if line_no < limit:
                    dst.write(line)
                else:
                    break
    return limited_root


# ----------------------------------------------------------------------
# Mock model adapter (used by --smoke)
# ----------------------------------------------------------------------


class MockModel:
    """Deterministic mock model.

    Emits A..H letters using a seeded RNG so smoke runs are reproducible.
    Optionally returns a fixed letter ('--mock-letter A') for sanity
    checks against a single-letter baseline.
    """

    def __init__(self, mode: str = "random", fixed_letter: str = "A", seed: int = 42):
        if mode not in ("random", "fixed"):
            raise ValueError(f"unknown mock mode: {mode}")
        self.mode = mode
        self.fixed_letter = fixed_letter
        self._rng = random.Random(seed)

    def generate(self, prompt_text: str, kept_letters: list[str]) -> str:
        if self.mode == "fixed":
            return self.fixed_letter
        return self._rng.choice(kept_letters or list("ABCDEFGH"))


# ----------------------------------------------------------------------
# Smoke runner: runs the dataset directly with a mock model and writes
# results without going through VLMEvalKit's heavy machinery.
# ----------------------------------------------------------------------


def run_smoke(
    *,
    bench_version: str,
    video_mode: str,
    limit: int,
    out_dir: Path,
    mock_mode: str = "random",
    mock_letter: str = "A",
    seed: int = 42,
    dataset_root: str | None = None,
) -> dict[str, Any]:
    """Execute a smoke run end-to-end and return the metrics dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = EgotoolsBench(
        dataset=f"EgotoolsBench_{bench_version}_{video_mode}",
        version=bench_version,
        video_mode=video_mode,
        dataset_root=dataset_root,
    )

    n_total = len(ds)
    if limit and limit > 0:
        n_run = min(limit, n_total)
    else:
        n_run = n_total

    model = MockModel(mode=mock_mode, fixed_letter=mock_letter, seed=seed)

    rows: list[dict[str, Any]] = []
    for i in range(n_run):
        row = dict(ds.data.iloc[i])
        prompt_text, kept_letters = build_mcq_prompt_text(row)
        prediction = model.generate(prompt_text, kept_letters)

        rec = {
            **row,
            "index": row.get("index", i),
            "qa_id": row.get("qa_id"),
            "qtype": row.get("qtype"),
            "canonical_video_id": row.get("canonical_video_id"),
            "video": row.get("video"),
            "clip_video": row.get("clip_video"),
            "answer": row.get("answer"),
            "n_options": len(kept_letters),
            "prompt": prompt_text,
            "prediction": prediction,
        }
        rows.append(rec)

    df = pd.DataFrame(rows)

    # Persist raw predictions (TSV is the VLMEvalKit-friendly format).
    pred_tsv = out_dir / "predictions.tsv"
    df.to_csv(pred_tsv, sep="\t", index=False)

    # Score.
    metrics = EgotoolsBench.score_dataframe(df, write_alongside=pred_tsv)

    # Persist scored predictions (with predicted_letter + score columns).
    df["predicted_letter"] = df["prediction"].apply(extract_letter_ah)
    df["score"] = (df["predicted_letter"] == df["answer"].astype(str)).astype(int)
    df.to_csv(out_dir / "predictions_scored.tsv", sep="\t", index=False)

    # Compact summary alongside raw and scored predictions.
    results = {
        "run_kind": "smoke",
        "model": f"mock-{mock_mode}" + (f"-{mock_letter}" if mock_mode == "fixed" else ""),
        "bench_version": bench_version,
        "video_mode": video_mode,
        "n_total_in_manifest": n_total,
        "n_evaluated": n_run,
        "metrics": metrics,
        "data_root": ds.data_root,
        "data_file": ds.data_file,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    return results


def find_vlmevalkit_run(directory: str | None = None) -> Path:
    """Find run.py in an explicit checkout or the editable upstream install."""
    requested = directory or os.environ.get("VLMEVALKIT_DIR")
    if requested:
        candidate = Path(requested).expanduser().resolve() / "run.py"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"VLMEvalKit run.py not found: {candidate}")
    local = _EVAL_ROOT / "VLMEvalKit" / "run.py"
    if local.is_file():
        return local
    spec = importlib.util.find_spec("vlmeval")
    if spec and spec.origin:
        installed = Path(spec.origin).resolve().parents[1] / "run.py"
        if installed.is_file():
            return installed
    raise FileNotFoundError("VLMEvalKit run.py not found; run eval/setup/setup_env.sh or pass --vlmevalkit-dir")


def run_real_model(
    *,
    bench_version: str,
    video_mode: str,
    model: str,
    limit: int,
    out_dir: Path,
    dataset_root: str | None = None,
    extra_args: list[str] | None = None,
    nproc_per_node: int = 1,
    hf_model_id: str | None = None,
    auto_download_model: bool = True,
    nframe: int = 64,
    fps: float = -1,
    vlmevalkit_dir: str | None = None,
    model_path: str | None = None,
    dry_run: bool = False,
) -> int:
    """Launch upstream inference, then score its complete prediction table."""
    root = resolve_dataset_root(dataset_root, bench_version)
    with (root / "manifest.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("manifest contains no questions")
    selected = rows[:limit] if limit else rows
    ids = [row.get("index") for row in selected]
    if any(index is None or not str(index).strip() for index in ids) or len(set(ids)) != len(ids):
        raise ValueError("manifest indices must be nonempty and unique; run egotools-prepare-manifest first")
    for row in selected:
        path = row.get("clip_video") if video_mode == "clip" else row.get("video")
        path = path or row.get("video")
        if path and (Path(path).is_absolute() or ".." in Path(path).parts):
            raise ValueError(f"manifest media paths must be relative to the dataset root: {path!r}")
        if not path or not (root / path).is_file():
            raise FileNotFoundError(f"video missing for qa_id={row.get('qa_id')!r}: {path!r} under {root}")
    run_py = find_vlmevalkit_run(vlmevalkit_dir)
    model = normalize_model_name(model)
    if model_path:
        model_path = str(Path(model_path).expanduser().resolve())
        if not Path(model_path).is_dir():
            raise FileNotFoundError(f"model checkpoint directory does not exist: {model_path}")
    dataset_alias = f"EgotoolsBench_{bench_version}_{video_mode}"
    if fps > 0:
        dataset_alias += f"_{fps:g}fps"
    elif nframe > 0:
        dataset_alias += f"_{nframe}frame"
    passthrough_args = list(extra_args or [])
    reserved = {"--data", "--model", "--work-dir", "--config"}
    if any(arg.split("=", 1)[0] in reserved for arg in passthrough_args):
        raise ValueError("--extra cannot override --data, --model, --work-dir, or --config; use runner options")
    entry = _EVAL_ROOT / "scripts" / "_torchrun_entry.py"
    inner_args = ["--data", dataset_alias, "--model", model, "--work-dir", str(out_dir), *passthrough_args]
    cmd = [sys.executable]
    if nproc_per_node > 1:
        cmd += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc_per_node}"]
    cmd += [str(entry), "--", *inner_args]
    env = os.environ.copy()
    env.update(
        {
            "EGOTOOLS_BENCH_ROOT": str(root),
            "EGOTOOLS_BENCH_VERSION": bench_version,
            "EGOTOOLS_NFRAME": str(nframe),
            "EGOTOOLS_FPS": str(fps),
            "EGOTOOLS_VLMEVALKIT_RUN": str(run_py),
            "EGOTOOLS_MODEL_KEY": model,
        }
    )
    # Upstream image caches are scoped to this run so another dataset root with
    # the same relative video names cannot reuse different video frames.
    env.setdefault("LMUData", str(out_dir / "cache"))
    if model_path:
        env["EGOTOOLS_MODEL_PATH"] = model_path
    else:
        env.pop("EGOTOOLS_MODEL_PATH", None)
    print(f"[run_eval] {len(selected)}/{len(rows)} questions; dataset={dataset_alias}")
    print(f"[run_eval] upstream={run_py}")
    print(f"[run_eval] command: {shlex.join(cmd)}", flush=True)
    if dry_run:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    if limit:
        env["EGOTOOLS_BENCH_ROOT"] = str(
            make_limited_dataset_root(
                dataset_root=str(root),
                limit=limit,
                out_dir=out_dir,
            )
        )
    if not model_path:
        snapshot = ensure_hf_model_available(model=model, hf_model_id=hf_model_id, auto_download=auto_download_model)
        if snapshot:
            env["EGOTOOLS_MODEL_PATH"] = snapshot
    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode:
        return result.returncode
    from eval.scripts._aggregate_metrics import aggregate_results

    aggregate_results(out_dir, model)
    return 0


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_eval.py",
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
        help="Model name. Use 'mock' for the built-in mock model, otherwise "
        "a VLMEvalKit-known model (for example Qwen3-VL-8B-Instruct). "
        "Common shorthand like qwen3vl8binstruct is normalized.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap the number of QAs evaluated (0 = full bench).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run the in-process smoke runner with the mock model. Skips "
        "VLMEvalKit's heavy machinery. Always uses the full TSV unless "
        "--limit is set.",
    )
    p.add_argument(
        "--mock-mode",
        choices=("random", "fixed"),
        default="random",
        help="Mock model behavior (smoke / --model mock only).",
    )
    p.add_argument(
        "--mock-letter",
        default="A",
        choices=list("ABCDEFGH"),
        help="Letter returned by 'fixed'-mode mock model (default: A).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the mock model.",
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
        "--nframe", type=int, default=None, help="Uniform frame count (default: 64); 0 uses model defaults"
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


def _print_metrics(metrics: dict, n_total: int, n_run: int) -> None:
    overall = metrics["overall"]
    print()
    print("=" * 60)
    print(f"Smoke run summary  ({n_run}/{n_total} QAs evaluated)")
    print("=" * 60)
    print(f"Overall accuracy: {overall['accuracy'] * 100:.2f}%  ({overall['correct']}/{overall['total']})")
    cov = metrics["extraction_coverage"]
    print(f"Extraction coverage: {cov['rate'] * 100:.2f}%  ({cov['extracted']}/{cov['total']})")
    print()
    print("Per-qtype accuracy:")
    rows = sorted(
        metrics["per_qtype"].items(),
        key=lambda kv: (-kv[1]["total"], kv[0]),
    )
    for qt, v in rows:
        print(f"  {qt:<24s}  {v['accuracy'] * 100:6.2f}%  ({v['correct']}/{v['total']})")
    print("=" * 60)


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
    args.nframe = args.nframe if args.nframe is not None else (0 if args.fps is not None else 64)
    args.fps = args.fps if args.fps is not None else -1
    if args.dry_run and (args.smoke or args.model.lower() == "mock"):
        parser.error("--dry-run checks a real-model command; use --model mock for a CPU smoke run")
    if not args.smoke and args.model.lower() != "mock":
        args.model = normalize_model_name(args.model)

    run_id = os.environ.get("EGOTOOLS_RUN_ID") or make_run_id(
        args.bench_version,
        args.model if not args.smoke else f"mock-{args.mock_mode}",
    )
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        parser.error("EGOTOOLS_RUN_ID must be a single directory name")
    out_dir = Path(args.results_dir).expanduser().resolve() / run_id
    print(f"[run_eval] run_id={run_id}")
    print(f"[run_eval] out_dir={out_dir}")

    # Smoke OR --model mock => smoke runner.
    if args.smoke or args.model.lower() == "mock":
        try:
            results = run_smoke(
                bench_version=args.bench_version,
                video_mode=args.video_mode,
                limit=args.limit,
                out_dir=out_dir,
                mock_mode=args.mock_mode,
                mock_letter=args.mock_letter,
                seed=args.seed,
                dataset_root=args.dataset_root,
            )
        except FileNotFoundError as e:
            print(f"[run_eval] FATAL: {e}", file=sys.stderr)
            return 2
        except Exception as e:
            print(f"[run_eval] FATAL: {e}", file=sys.stderr)
            traceback.print_exc()
            return 1

        _print_metrics(
            results["metrics"],
            results["n_total_in_manifest"],
            results["n_evaluated"],
        )
        print(f"[run_eval] results -> {out_dir}/results.json")
        return 0

    # Real model path.
    rc = run_real_model(
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
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
