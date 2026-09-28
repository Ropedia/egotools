from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eval.scripts.materialize_manifest_dataset import materialize_manifest_dataset, resolve_asset_root


class MaterializeManifestDatasetTest(unittest.TestCase):
    def test_resolves_hf_snapshot_parent_and_links_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.tsv"
            manifest.write_text("index\tvideo\tclip_video\n0\tvideos/a.mp4\tclips/q.mp4\n")

            snapshot = root / "hf_cache" / "snapshots" / "abc123"
            (snapshot / "videos").mkdir(parents=True)
            (snapshot / "clips").mkdir()
            (snapshot / "videos" / "a.mp4").write_bytes(b"video")
            (snapshot / "clips" / "q.mp4").write_bytes(b"clip")

            resolved = resolve_asset_root(root / "hf_cache")
            self.assertEqual(resolved, snapshot.resolve())

            output = root / "subset"
            summary = materialize_manifest_dataset(
                manifest_path=manifest,
                output_dir=output,
                source_root=root / "hf_cache",
                link_mode="symlink",
            )

            self.assertEqual(summary["rows"], 1)
            self.assertEqual((output / "manifest.tsv").read_text(), manifest.read_text())
            self.assertTrue((output / "videos").is_symlink())
            self.assertTrue((output / "clips").is_symlink())
            self.assertEqual((output / "videos" / "a.mp4").read_bytes(), b"video")
            self.assertEqual((output / "clips" / "q.mp4").read_bytes(), b"clip")

    def test_default_asset_check_only_requires_full_videos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.tsv"
            manifest.write_text("index\tvideo\tclip_video\n0\tvideos/a.mp4\tclips/missing.mp4\n")

            source = root / "source"
            (source / "videos").mkdir(parents=True)
            (source / "clips").mkdir()
            (source / "videos" / "a.mp4").write_bytes(b"video")

            summary = materialize_manifest_dataset(
                manifest_path=manifest,
                output_dir=root / "subset",
                source_root=source,
            )

            self.assertEqual(summary["rows"], 1)
            self.assertEqual(summary["asset_mode"], "full")


if __name__ == "__main__":
    unittest.main()


def test_rejects_ambiguous_snapshot_selection(tmp_path):
    import pytest

    for revision in ("first", "second"):
        (tmp_path / "snapshots" / revision / "videos").mkdir(parents=True)
    with pytest.raises(ValueError, match="multiple snapshots"):
        resolve_asset_root(tmp_path)


def test_full_assets_do_not_require_clip_directory(tmp_path):
    manifest = tmp_path / "input.tsv"
    manifest.write_text("index\tvideo\n0\tvideos/a.mp4\n")
    source = tmp_path / "source"
    (source / "videos").mkdir(parents=True)
    (source / "videos" / "a.mp4").write_bytes(b"video")
    output = tmp_path / "output"
    summary = materialize_manifest_dataset(manifest_path=manifest, source_root=source, output_dir=output)
    assert summary["rows"] == 1
    assert (output / "videos" / "a.mp4").is_file()


def test_materialize_preserves_existing_asset_directory(tmp_path):
    import pytest

    manifest = tmp_path / "input.tsv"
    manifest.write_text("index\tvideo\n0\tvideos/a.mp4\n")
    source = tmp_path / "source"
    (source / "videos").mkdir(parents=True)
    (source / "videos" / "a.mp4").write_bytes(b"video")
    output = tmp_path / "output"
    (output / "videos").mkdir(parents=True)
    sentinel = output / "videos" / "keep.mp4"
    sentinel.write_bytes(b"existing video")
    with pytest.raises(FileExistsError):
        materialize_manifest_dataset(manifest_path=manifest, source_root=source, output_dir=output)
    assert sentinel.read_bytes() == b"existing video"
