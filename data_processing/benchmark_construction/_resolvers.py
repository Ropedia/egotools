"""Resolve normalized QA rows through explicit per-source media manifests.

Each source has <source_id>/annotation_manifest.json below --workspace-sources-root.
A manifest provides canonical_video_id, source_video_path, and annotations with
annotation_id and clip_path. Relative media paths are relative to that manifest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VideoRefs:
    """Resolved on-disk locations for a single normalized QA row.

    ``source_video_path`` is the full per-canonical-video file (shared across
    every QA that has the same canonical_video_id). ``clip_path`` is the
    per-annotation short clip — may be ``None`` if the annotation is not
    listed in the manifest, or if the file is missing on disk. The builder
    treats a missing clip as a non-fatal warning (clip_video is blank in the
    output manifest), but a missing full video drops the row.
    """

    canonical_video_id: str
    source_video_path: Path | None
    clip_path: Path | None
    # Diagnostic — populated on resolution failure to be surfaced in
    # skipped_build.csv. Empty string on success.
    reason: str = ""


# --------------------------------------------------------------------------- #
# Manifest loading (cached — same source_id is hit for many rows)
# --------------------------------------------------------------------------- #


@cache
def _load_manifest(manifest_path: str) -> dict | None:
    """Load and cache an annotation_manifest.json. Returns None if missing."""
    p = Path(manifest_path)
    if not p.exists():
        return None
    try:
        with p.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def source_id_from_source_file(source_file: str) -> str:
    """Convert a source JSON filename to its directory identifier."""
    if source_file.endswith(".json"):
        return source_file[: -len(".json")]
    return source_file


def manifest_path_for(workspace_sources_root: Path, source_id: str) -> Path:
    if not source_id or source_id in {".", ".."} or any(c in source_id for c in "/\\"):
        raise ValueError(f"source_file must be a basename, got {source_id!r}")
    return workspace_sources_root / source_id / "annotation_manifest.json"


# --------------------------------------------------------------------------- #
# Public resolver
# --------------------------------------------------------------------------- #


def resolve_video_refs(
    *,
    workspace_sources_root: Path,
    source_file: str,
    annotation_id: str,
    canonical_video_id_from_row: str,
) -> VideoRefs:
    """Resolve full-video + clip paths for one normalized row.

    The row provides the canonical ID; a conflicting manifest ID is rejected.
    The manifest supplies full-video paths and per-annotation clip lookup.

    Failure modes (each fills ``reason`` and clears one or both paths):
      * Empty ``canonical_video_id_from_row`` → ``empty_canonical_video_id``.
        Caller should drop the row.
      * Manifest file missing → ``manifest_missing``. Caller should drop.
      * ``source_video_path`` missing on disk → ``source_video_missing``.
        Caller should drop.
      * ``annotation_id`` not in manifest, or its ``clip_path`` is empty /
        missing on disk → ``clip_path`` is ``None`` but ``source_video_path``
        is still set; caller keeps the row but writes empty ``clip_video``.
    """
    cid = (canonical_video_id_from_row or "").strip()
    if not cid:
        return VideoRefs(
            canonical_video_id="",
            source_video_path=None,
            clip_path=None,
            reason="empty_canonical_video_id",
        )

    source_id = source_id_from_source_file(source_file)
    mpath = manifest_path_for(workspace_sources_root, source_id)
    manifest = _load_manifest(str(mpath))
    if manifest is None:
        return VideoRefs(
            canonical_video_id=cid,
            source_video_path=None,
            clip_path=None,
            reason=f"manifest_missing:{mpath}",
        )

    manifest_cid = str(manifest.get("canonical_video_id") or "").strip()
    if manifest_cid and manifest_cid != cid:
        raise ValueError(f"Canonical video ID differs between row and {mpath}")
    svp_str = manifest.get("source_video_path") or ""
    svp = Path(svp_str) if svp_str else None
    if svp is not None and not svp.is_absolute():
        svp = mpath.parent / svp
    if svp is None or not svp.is_file():
        return VideoRefs(
            canonical_video_id=cid,
            source_video_path=None,
            clip_path=None,
            reason=f"source_video_missing:{svp_str}",
        )

    # Per-annotation clip lookup (best-effort).
    clip_path: Path | None = None
    clip_reason = ""
    annot = None
    for a in manifest.get("annotations", []):
        if a.get("annotation_id") == annotation_id:
            annot = a
            break
    if annot is None:
        clip_reason = "annotation_id_not_in_manifest"
    else:
        cp = annot.get("clip_path") or ""
        if not cp:
            clip_reason = "clip_path_empty_in_manifest"
        else:
            cp_p = Path(cp)
            if not cp_p.is_absolute():
                cp_p = mpath.parent / cp_p
            if not cp_p.is_file():
                clip_reason = f"clip_path_missing_on_disk:{cp}"
            else:
                clip_path = cp_p

    return VideoRefs(
        canonical_video_id=cid,
        source_video_path=svp,
        clip_path=clip_path,
        reason=clip_reason,  # empty when clip resolved, set when only clip is missing
    )
