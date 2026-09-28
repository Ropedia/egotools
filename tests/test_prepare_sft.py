from __future__ import annotations

import json

import pytest

from training.prepare_sft import prepare_sft


def test_preserves_model_inputs_and_order_while_removing_varied_provenance(tmp_path):
    rows = [
        {"messages": [{"role": "user", "content": "<video>What happens?"},
                      {"role": "assistant", "content": "A tool moves.", "loss_scale": 0.5}],
         "videos": ["videos/first.mp4"], "metadata": {"caption": ["one"]},
         "_index": 42, "start_frame": 0, "end_frame": 64},
        {"messages": [{"role": "user", "content": "<image>Which tool?"},
                      {"role": "assistant", "content": "螺丝刀"}],
         "images": ["images/second.jpg"], "metadata": {"qa_type": "single-image", "count": 1}},
    ]
    source = tmp_path / "original.jsonl"
    original = "\n".join(json.dumps(row) for row in rows) + "\n"
    source.write_text(original)
    output = tmp_path / "prepared.jsonl"
    report = prepare_sft(source, output)
    prepared = [json.loads(line) for line in output.read_text().splitlines()]
    assert report["rows"] == 2
    assert report["removed_columns"] == {"_index": 1, "end_frame": 1, "metadata": 2, "start_frame": 1}
    assert prepared == [
        {"messages": rows[0]["messages"], "videos": rows[0]["videos"]},
        {"messages": rows[1]["messages"], "images": rows[1]["images"]},
    ]
    assert source.read_text() == original


@pytest.mark.parametrize("bad_row", ["not json", "[]", '{}', '{"messages": []}',
                                      '{"messages": [{"role":"user","content":null}]}'])
def test_invalid_later_row_does_not_publish_partial_output(tmp_path, bad_row):
    source = tmp_path / "original.jsonl"
    source.write_text('{"messages":[{"role":"assistant","content":"valid"}]}\n' + bad_row + "\n")
    output = tmp_path / "prepared.jsonl"
    output.write_text("previous result\n")
    with pytest.raises(ValueError, match="line 2"):
        prepare_sft(source, output)
    assert output.read_text() == "previous result\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["original.jsonl", "prepared.jsonl"]


def test_does_not_overwrite_source_or_accept_empty_data(tmp_path):
    source = tmp_path / "original.jsonl"
    source.write_text("\n")
    with pytest.raises(ValueError, match="separate output"):
        prepare_sft(source, source)
    with pytest.raises(ValueError, match="no SFT records"):
        prepare_sft(source, tmp_path / "prepared.jsonl")
    assert not (tmp_path / "prepared.jsonl").exists()
