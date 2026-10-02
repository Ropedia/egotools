from __future__ import annotations

import csv
import json

import pytest

from egotools.manifest import OPTION_LETTERS, read_manifest, validate_manifest
from egotools.prepare import main, prepare_benchmark, resolve_asset_root


def _inputs(tmp_path, rows=None):
    row = {
        "index": 7,
        "qa_id": "q0",
        "canonical_video_id": "a",
        "video": "videos/a.mp4",
        "clip_video": "clips/missing.mp4",
        "question": "Which tool?",
        **dict.fromkeys(OPTION_LETTERS, ""),
        "A": "hammer",
        "B": "pliers",
        "answer": "A",
    }
    manifest = tmp_path / "input.jsonl"
    manifest.write_text("".join(json.dumps(item) + "\n" for item in (rows or [row])))
    assets = tmp_path / "assets"
    (assets / "videos").mkdir(parents=True)
    (assets / "videos" / "a.mp4").write_bytes(b"video")
    return manifest, assets


def test_cli_resolves_snapshot_preserves_indices_and_links_assets(tmp_path, capsys):
    manifest, assets = _inputs(tmp_path)
    (assets / "clips").mkdir()
    (assets / "clips" / "missing.mp4").write_bytes(b"clip")
    snapshot = tmp_path / "cache" / "snapshots" / "abc123"
    snapshot.parent.mkdir(parents=True)
    assets.rename(snapshot)
    assert resolve_asset_root(tmp_path / "cache") == snapshot
    output = tmp_path / "benchmark"
    command = ["--manifest", str(manifest), "--assets", str(tmp_path / "cache"), "--output", str(output)]
    assert main(command) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["rows"] == 1
    assert not summary["reindexed"]
    assert read_manifest(output / "manifest.tsv")[1][0]["index"] == "7"
    assert validate_manifest(output / "manifest.tsv")["valid"]
    assert (output / "videos").is_symlink()
    assert (output / "clips").is_symlink()
    assert (output / "clips" / "missing.mp4").read_bytes() == b"clip"
    assert main(command) == 0


@pytest.mark.parametrize("link_mode", ["symlink", "copy"])
def test_full_assets_do_not_require_clips(tmp_path, link_mode):
    manifest, assets = _inputs(tmp_path)
    output = tmp_path / "benchmark"
    summary = prepare_benchmark(manifest=manifest, assets=assets, output=output, link_mode=link_mode)
    assert summary["asset_mode"] == "full"
    assert summary["checked_assets"] == 1
    assert (output / "videos").is_symlink() == (link_mode == "symlink")
    assert (output / "videos" / "a.mp4").read_bytes() == b"video"
    with pytest.raises(ValueError, match="missing 1 checked asset"):
        prepare_benchmark(manifest=manifest, assets=assets, output=tmp_path / "both", asset_mode="both")
    assert not (tmp_path / "both").exists()


@pytest.mark.parametrize("indices", [(7, 7), ("01", "1"), ("1.0", "1"), ("1e0", "1")])
def test_reindex_is_explicit_and_preserves_question_data(tmp_path, indices):
    manifest, assets = _inputs(tmp_path)
    first = read_manifest(manifest)[1][0]
    rows = [first | {"index": index, "qa_id": f"q{number}"} for number, index in enumerate(indices)]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    original = manifest.read_bytes()
    output = tmp_path / "benchmark"
    assert "--reindex" in "\n".join(validate_manifest(manifest)["errors"])
    with pytest.raises(ValueError, match="duplicate index.*--reindex"):
        prepare_benchmark(manifest=manifest, assets=assets, output=output)
    assert not output.exists()
    assert prepare_benchmark(manifest=manifest, assets=assets, output=output, reindex=True)["reindexed"]
    prepared = read_manifest(output / "manifest.tsv")[1]
    assert [row["index"] for row in prepared] == ["0", "1"]
    for before, after in zip(rows, prepared, strict=True):
        assert {key: value for key, value in before.items() if key != "index"} == {
            key: value for key, value in after.items() if key != "index"
        }
    assert manifest.read_bytes() == original


def test_media_paths_are_normalized_in_the_written_tsv(tmp_path):
    manifest, assets = _inputs(tmp_path)
    row = read_manifest(manifest)[1][0] | {"video": " videos\\a.mp4 ", "clip_video": " \t "}
    manifest.write_text(json.dumps(row) + "\n")
    original = manifest.read_bytes()
    output = tmp_path / "benchmark"
    prepare_benchmark(manifest=manifest, assets=assets, output=output, asset_mode="clip")
    with (output / "manifest.tsv").open(newline="") as handle:
        prepared = next(csv.DictReader(handle, delimiter="\t"))
    assert prepared["video"] == "videos/a.mp4"
    assert prepared["clip_video"] == ""
    assert (output / prepared["video"]).is_file()
    assert manifest.read_bytes() == original


def test_copy_mode_refuses_to_reuse_symlinks(tmp_path):
    manifest, assets = _inputs(tmp_path)
    output = tmp_path / "benchmark"
    prepare_benchmark(manifest=manifest, assets=assets, output=output)
    original = (output / "manifest.tsv").read_bytes()
    with pytest.raises(FileExistsError, match="copy mode"):
        prepare_benchmark(manifest=manifest, assets=assets, output=output, link_mode="copy")
    assert (output / "videos").is_symlink()
    assert (output / "videos").resolve() == assets / "videos"
    assert (output / "manifest.tsv").read_bytes() == original


@pytest.mark.parametrize("conflict", ["manifest", "assets"])
def test_preserves_existing_outputs(tmp_path, conflict):
    manifest, assets = _inputs(tmp_path)
    output = tmp_path / "benchmark"
    sentinel = output / ("manifest.tsv" if conflict == "manifest" else "videos/keep.mp4")
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"existing content")
    with pytest.raises(FileExistsError):
        prepare_benchmark(manifest=manifest, assets=assets, output=output)
    assert sentinel.read_bytes() == b"existing content"
    if conflict == "assets":
        assert not (output / "manifest.tsv").exists()


def test_rejects_ambiguous_snapshot_selection(tmp_path):
    for revision in ("first", "second"):
        (tmp_path / "snapshots" / revision / "videos").mkdir(parents=True)
    with pytest.raises(ValueError, match="multiple snapshots"):
        resolve_asset_root(tmp_path)


def test_validates_paths_before_creating_output(tmp_path):
    manifest, assets = _inputs(tmp_path)
    row = read_manifest(manifest)[1][0] | {"video": "../outside.mp4"}
    manifest.write_text(json.dumps(row) + "\n")
    output = tmp_path / "benchmark"
    with pytest.raises(ValueError, match="video must be a non-empty relative path"):
        prepare_benchmark(manifest=manifest, assets=assets, output=output)
    assert not output.exists()
