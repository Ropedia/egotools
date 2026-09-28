#!/usr/bin/env python
"""Pair a local TSV manifest with externally downloaded video assets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any


def resolve_asset_root(root: Path) -> Path:
    """Accept an asset directory or an unambiguous Hugging Face snapshot cache."""
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


def materialize_manifest_dataset(
    *,
    manifest_path: Path,
    output_dir: Path,
    source_root: Path,
    link_mode: str = "symlink",
    asset_mode: str = "full",
    skip_asset_check: bool = False,
) -> dict[str, Any]:
    if link_mode not in {"symlink", "copy"}:
        raise ValueError("link_mode must be symlink or copy")
    if asset_mode not in {"full", "clip", "both"}:
        raise ValueError("asset_mode must be full, clip, or both")
    manifest_path = manifest_path.expanduser().resolve()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "video" not in reader.fieldnames:
            raise ValueError("manifest must contain a video column")
        rows = list(reader)
    if not rows:
        raise ValueError("manifest contains no rows")
    asset_root = resolve_asset_root(source_root)
    checked = 0
    asset_directories = set()
    for row in rows:
        full = (row.get("video") or "").strip()
        clip = (row.get("clip_video") or "").strip() or full
        selected = {"full": [full], "clip": [clip], "both": [full, clip]}[asset_mode]
        for value in (full, (row.get("clip_video") or "").strip()):
            if not value:
                continue
            path = Path(value)
            if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
                raise ValueError(f"expected a relative asset path under an asset directory: {value!r}")
            asset_directories.add(path.parts[0])
        if not skip_asset_check:
            for value in selected:
                checked += 1
                if not value or not (asset_root / value).is_file():
                    raise FileNotFoundError(f"missing {asset_mode} asset: {asset_root / value}")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_manifest = output_dir / "manifest.tsv"
    if target_manifest.exists() and target_manifest.read_bytes() != manifest_path.read_bytes():
        raise FileExistsError(
            f"a different manifest already exists at {target_manifest}; choose another output directory"
        )
    for name in sorted(asset_directories):
        source = asset_root / name
        target = output_dir / name
        if not source.is_dir():
            continue
        if target.exists() or target.is_symlink():
            if target.resolve() == source.resolve():
                continue
            raise FileExistsError(f"asset directory already exists: {target}")
        if link_mode == "symlink":
            target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=True)
        else:
            shutil.copytree(source, target)
    if target_manifest.resolve() != manifest_path:
        shutil.copy2(manifest_path, target_manifest)
    return {
        "manifest": str(target_manifest),
        "resolved_asset_root": str(asset_root),
        "output_dir": str(output_dir),
        "rows": len(rows),
        "link_mode": link_mode,
        "asset_mode": asset_mode,
        "checked_assets": checked,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--asset-mode", choices=("full", "clip", "both"), default="full")
    parser.add_argument("--skip-asset-check", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            materialize_manifest_dataset(
                manifest_path=args.manifest,
                source_root=args.source_root,
                output_dir=args.output_dir,
                link_mode=args.link_mode,
                asset_mode=args.asset_mode,
                skip_asset_check=args.skip_asset_check,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
