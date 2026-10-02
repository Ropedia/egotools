"""Prepare a validated benchmark directory from a manifest and local media."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .manifest import read_manifest, validate_manifest


def prepare_manifest(source: Path, output: Path, *, reindex: bool = False) -> int:
    """Write a new TSV, preserving indices unless explicitly reindexing."""
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    columns, rows = read_manifest(source)
    seen: set[str] = set()
    for row_number, row in enumerate(rows):
        qa_id = str(row.get("qa_id") or "").strip()
        if not qa_id or qa_id in seen:
            raise ValueError(f"row {row_number + 1}: missing or duplicate qa_id {qa_id!r}")
        seen.add(qa_id)
        if reindex:
            row["index"] = row_number
    if reindex and "index" not in columns:
        columns.insert(0, "index")
    if source == output:
        raise ValueError("choose a separate output path to retain the original manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def resolve_asset_root(root: Path) -> Path:
    """Accept a media root or a cache containing exactly one media snapshot."""
    root = root.expanduser().resolve()
    if (root / "videos").is_dir() or (root / "clips").is_dir():
        return root
    snapshots = root / "snapshots"
    candidates = sorted(p for p in snapshots.glob("*") if (p / "videos").is_dir() or (p / "clips").is_dir())
    if len(candidates) == 1:
        return candidates[0].resolve()
    if len(candidates) > 1:
        raise ValueError(f"multiple snapshots found under {root}; select the required snapshot directory explicitly")
    raise FileNotFoundError(f"no videos/ or clips/ directory under {root} or {snapshots}")


def prepare_benchmark(
    *,
    manifest: Path,
    assets: Path,
    output: Path,
    link_mode: str = "symlink",
    asset_mode: str = "full",
    reindex: bool = False,
) -> dict[str, Any]:
    if link_mode not in {"symlink", "copy"}:
        raise ValueError("link_mode must be symlink or copy")
    if asset_mode not in {"full", "clip", "both"}:
        raise ValueError("asset_mode must be full, clip, or both")
    asset_root = resolve_asset_root(assets)
    output = output.expanduser().resolve()
    target_manifest = output / "manifest.tsv"
    with tempfile.TemporaryDirectory() as staging:
        prepared = Path(staging) / "manifest.tsv"
        prepare_manifest(manifest, prepared, reindex=reindex)
        report = validate_manifest(prepared, asset_root=asset_root, asset_mode=asset_mode)
        if not report["valid"]:
            raise ValueError("invalid manifest:\n" + "\n".join(report["errors"]))
        if target_manifest.exists() or target_manifest.is_symlink():
            if not target_manifest.is_file() or target_manifest.read_bytes() != prepared.read_bytes():
                raise FileExistsError(f"a different manifest already exists at {target_manifest}; choose another output")

        _, rows = read_manifest(prepared)
        directories = set()
        for row in rows:
            for column in ("video", "clip_video"):
                value = str(row.get(column) or "").strip()
                if not value:
                    continue
                parts = Path(value).parts
                if len(parts) < 2 or parts[0] == "manifest.tsv":
                    raise ValueError(f"expected a relative media path under an asset directory: {value!r}")
                directories.add(parts[0])
        pairs = [(asset_root / name, output / name) for name in sorted(directories) if (asset_root / name).is_dir()]
        for source, target in pairs:
            if output.is_relative_to(source.resolve()):
                raise ValueError(f"output directory cannot be inside the source media directory: {source}")
            if (target.exists() or target.is_symlink()) and (link_mode == "copy" or target.resolve() != source.resolve()):
                raise FileExistsError(f"asset directory already exists for {link_mode} mode: {target}")

        output.mkdir(parents=True, exist_ok=True)
        for source, target in pairs:
            if target.exists():
                continue
            if link_mode == "symlink":
                target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=True)
            else:
                shutil.copytree(source, target)
        if not target_manifest.exists():
            with target_manifest.open("xb") as handle:
                handle.write(prepared.read_bytes())
    return {
        "manifest": str(target_manifest),
        "resolved_asset_root": str(asset_root),
        "rows": report["rows"],
        "link_mode": link_mode,
        "asset_mode": asset_mode,
        "checked_assets": report["checked_assets"],
        "reindexed": reindex,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="egotools-prepare-benchmark", description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Input TSV, CSV, or JSONL manifest.")
    parser.add_argument("--assets", type=Path, required=True, help="Media directory or unambiguous HF snapshot cache.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for manifest.tsv and linked/copied media.")
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--asset-mode", choices=("full", "clip", "both"), default="full")
    parser.add_argument("--reindex", action="store_true", help="Replace indices with 0..N-1; preserves qa_id and row order.")
    args = parser.parse_args(argv)
    try:
        report = prepare_benchmark(**vars(args))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
