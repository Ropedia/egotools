"""Launch VLMEvalKit inference and collect the resulting prediction table."""

from __future__ import annotations

import csv
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from egotools.evaluation.data import resolve_dataset_root
from egotools.evaluation.models import default_sampling, ensure_hf_model_available, normalize_model_name
from egotools.manifest import OPTION_LETTERS


def make_limited_dataset_root(*, dataset_root: str, limit: int, out_dir: Path) -> Path:
    """Select the requested rows and retain their relative media paths."""
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

    asset_names = {
        Path(row.get(column, "")).parts[0]
        for row in rows[:limit]
        for column in ("video", "clip_video")
        if row.get(column) and Path(row[column]).parts
    }
    for name in asset_names:
        source = src_root / name
        target = limited_root / name
        if not source.exists() or target.exists() or target.is_symlink():
            continue
        try:
            target.symlink_to(source, target_is_directory=source.is_dir())
        except OSError:
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)

    return limited_root


def find_vlmevalkit_run(directory: str | None = None) -> Path:
    """Find run.py in an explicit checkout or the editable upstream install."""
    requested = directory or os.environ.get("VLMEVALKIT_DIR")
    if requested:
        candidate = Path(requested).expanduser().resolve() / "run.py"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"VLMEvalKit run.py not found: {candidate}")
    local = Path("third_party/VLMEvalKit/run.py").resolve()
    if local.is_file():
        return local
    spec = importlib.util.find_spec("vlmeval")
    if spec and spec.origin:
        installed = Path(spec.origin).resolve().parents[1] / "run.py"
        if installed.is_file():
            return installed
    raise FileNotFoundError("VLMEvalKit run.py not found; run scripts/setup_eval.sh or pass --vlmevalkit-dir")


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
    nframe: int | None = None,
    fps: float | None = None,
    vlmevalkit_dir: str | None = None,
    model_path: str | None = None,
    dry_run: bool = False,
) -> int:
    """Launch upstream inference, then score its complete prediction table."""
    model = normalize_model_name(model)
    if nframe is None and fps is None:
        nframe, fps = default_sampling(model)
    else:
        nframe = 0 if nframe is None else nframe
        fps = -1 if fps is None else fps
    if nframe < 0 or (nframe > 0 and fps > 0):
        raise ValueError("choose either a nonnegative frame count or a positive frame rate")
    root = resolve_dataset_root(dataset_root, bench_version)
    with (root / "manifest.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("manifest contains no questions")
    selected = rows[:limit] if limit else rows
    ids = [row.get("index") for row in selected]
    if any(index is None or not str(index).strip() for index in ids) or len(set(ids)) != len(ids):
        raise ValueError("manifest indices must be nonempty and unique; run egotools-prepare-benchmark with --reindex")
    for row in selected:
        if any(not str(row.get(letter) or "").strip() for letter in OPTION_LETTERS):
            raise ValueError(f"qa_id={row.get('qa_id')!r}: EgoTools-Bench requires eight nonempty options A-H")
        path = row.get("clip_video") if video_mode == "clip" else row.get("video")
        path = path or row.get("video")
        if path and (Path(path).is_absolute() or ".." in Path(path).parts):
            raise ValueError(f"manifest media paths must be relative to the dataset root: {path!r}")
        if not path or not (root / path).is_file():
            raise FileNotFoundError(f"video missing for qa_id={row.get('qa_id')!r}: {path!r} under {root}")
    run_py = find_vlmevalkit_run(vlmevalkit_dir)
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
    inner_args = ["--data", dataset_alias, "--model", model, "--work-dir", str(out_dir), *passthrough_args]
    cmd = [sys.executable]
    if nproc_per_node > 1:
        cmd += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={nproc_per_node}",
            "--module",
        ]
    else:
        cmd += ["-m"]
    cmd += ["egotools.evaluation._torchrun_entry", "--", *inner_args]
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
    print(f"[egotools-evaluate] {len(selected)}/{len(rows)} questions; dataset={dataset_alias}")
    print(f"[egotools-evaluate] upstream={run_py}")
    print(f"[egotools-evaluate] command: {shlex.join(cmd)}", flush=True)
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
    from egotools.evaluation.metrics import aggregate_results

    aggregate_results(
        out_dir,
        model,
        manifest_path=Path(env["EGOTOOLS_BENCH_ROOT"]) / "manifest.tsv",
        input_settings={"video_mode": video_mode, "nframe": nframe, "fps": fps},
    )
    return 0
