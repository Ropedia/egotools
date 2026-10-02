"""Exercise the shell/CLI boundary without importing the GPU training stack."""

import json
import os
import subprocess
import sys
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "train.sh"


def launcher_env(**overrides):
    env = os.environ.copy()
    for key in (
        "DATASET", "MODEL", "OUTPUT_DIR", "SWIFT_BIN", "CUDA_VISIBLE_DEVICES",
        "NPROC_PER_NODE", "FPS", "FPS_MIN_FRAMES", "FPS_MAX_FRAMES",
        "VIDEO_MIN_TOKEN_NUM", "VIDEO_MAX_TOKEN_NUM", "IMAGE_MAX_TOKEN_NUM", "FORCE_QWENVL_VIDEO_READER",
        "SWIFT_PYTHON",
    ):
        env.pop(key, None)
    env.update(overrides)
    return env


def run_launcher(*args, env, cwd):
    return subprocess.run(
        ["bash", str(LAUNCHER), *map(str, args)],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=20,
    )


def make_stub(tmp_path):
    stub = tmp_path / "stub swift"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "keys = ['NPROC_PER_NODE', 'FPS', 'FPS_MIN_FRAMES', 'FPS_MAX_FRAMES', "
        "'VIDEO_MIN_TOKEN_NUM', 'VIDEO_MAX_TOKEN_NUM', 'IMAGE_MAX_TOKEN_NUM', 'FORCE_QWENVL_VIDEO_READER']\n"
        "print(json.dumps({'argv': sys.argv[1:], 'env': {k: os.environ[k] for k in keys}}))\n"
        "print('stub training stderr', file=sys.stderr)\n"
        "sys.exit(int(os.environ.get('STUB_EXIT', '0')))\n"
    )
    stub.chmod(0o755)
    return stub


def test_dry_run_needs_no_data_or_swift_and_round_trips_quoting(tmp_path):
    # Shell metacharacters in filenames must remain literal, including when the
    # printed dry-run command is pasted into a shell.
    dataset = tmp_path / "train $(touch injected).jsonl"
    output = tmp_path / "output with spaces"
    stub = make_stub(tmp_path)
    env = launcher_env(SWIFT_BIN=str(stub))
    result = run_launcher(
        "--dataset", dataset, "--model", "model with spaces", "--output-dir", output,
        "--dry-run", env=env, cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert not output.exists()
    assert not dataset.exists()
    replay = subprocess.run(
        ["bash", "-c", result.stdout], env=env, cwd=tmp_path,
        capture_output=True, text=True, timeout=20,
    )
    assert replay.returncode == 0, replay.stderr
    captured = json.loads(replay.stdout)
    assert captured["argv"][captured["argv"].index("--dataset") + 1] == str(dataset)
    assert captured["argv"][captured["argv"].index("--model") + 1] == "model with spaces"
    assert not (tmp_path / "injected").exists()
    assert captured["env"]["FPS_MIN_FRAMES"] == "64"
    assert captured["env"]["FPS_MAX_FRAMES"] == "64"
    assert captured["env"]["VIDEO_MIN_TOKEN_NUM"] == "128"
    assert captured["env"]["VIDEO_MAX_TOKEN_NUM"] == "128"

    missing_swift = run_launcher(
        "--dataset", dataset, "--output-dir", output, "--dry-run",
        env=launcher_env(SWIFT_BIN=str(tmp_path / "not installed")), cwd=tmp_path,
    )
    assert missing_swift.returncode == 0
    assert not output.exists()


def test_real_dispatch_preserves_overrides_and_failed_training_status(tmp_path):
    dataset = tmp_path / "local data.jsonl"
    dataset.write_text('{"messages": []}\n')
    output = tmp_path / "output directory"
    env = launcher_env(
        DATASET=str(tmp_path / "wrong.jsonl"), MODEL="environment-model",
        OUTPUT_DIR=str(output), SWIFT_BIN=str(make_stub(tmp_path)),
        NPROC_PER_NODE="1", FPS_MIN_FRAMES="16", FPS_MAX_FRAMES="16", STUB_EXIT="17",
    )
    result = run_launcher(
        "--dataset", dataset, "--model", "command-line-model", "--",
        "--max_steps", "1", "--gradient_accumulation_steps", "1",
        env=env, cwd=tmp_path,
    )
    assert result.returncode == 17, result.stdout + result.stderr
    log = (output / "train.log").read_text()
    captured = json.loads(next(line for line in log.splitlines() if line.startswith('{"argv"')))
    args = captured["argv"]
    assert args[0] == "sft"
    assert args[args.index("--model") + 1] == "command-line-model"
    assert args[args.index("--dataset") + 1] == str(dataset)
    assert args[-4:] == ["--max_steps", "1", "--gradient_accumulation_steps", "1"]
    assert captured["env"]["NPROC_PER_NODE"] == "1"
    assert captured["env"]["FPS_MIN_FRAMES"] == "16"
    assert "stub training stderr" in log


def test_missing_local_dataset_does_not_start_process_or_create_output(tmp_path):
    output = tmp_path / "output"
    result = run_launcher(
        "--dataset", tmp_path / "missing.jsonl", "--output-dir", output,
        env=launcher_env(SWIFT_BIN=str(make_stub(tmp_path))), cwd=tmp_path,
    )
    assert result.returncode == 2
    assert "dataset does not exist" in result.stderr
    assert not output.exists()


def test_help_and_argument_errors_do_not_require_training_dependencies(tmp_path):
    env = launcher_env(SWIFT_BIN=str(tmp_path / "not installed"))
    result = run_launcher("--help", env=env, cwd=tmp_path)
    assert result.returncode == 0
    assert "--dry-run" in result.stdout
    result = run_launcher("--dataset", "--dry-run", env=env, cwd=tmp_path)
    assert result.returncode == 2
    assert "--dataset requires a value" in result.stderr


def make_patch_check(tmp_path, exit_code):
    # Stands in for the interpreter that inspects the installed MS-Swift template.
    check = tmp_path / f"patch-check-{exit_code}"
    check.write_text(f"#!/bin/sh\ncat >/dev/null\nexit {exit_code}\n")
    check.chmod(0o755)
    return check


def test_dict_video_entries_require_patched_ms_swift(tmp_path):
    dataset = tmp_path / "windowed.jsonl"
    dataset.write_text(
        '{"messages": [{"role": "user", "content": "<video>Q"}, {"role": "assistant", "content": "A"}], '
        '"videos": [{"video": "clips/a.mp4", "video_start": 0, "video_end": 4}]}\n'
    )
    output = tmp_path / "output"
    stub = str(make_stub(tmp_path))
    unpatched = launcher_env(SWIFT_BIN=stub, SWIFT_PYTHON=str(make_patch_check(tmp_path, 1)))
    result = run_launcher("--dataset", dataset, "--output-dir", output, env=unpatched, cwd=tmp_path)
    assert result.returncode == 2
    assert "video-entry patch" in result.stderr
    assert not output.exists()

    patched = launcher_env(SWIFT_BIN=stub, SWIFT_PYTHON=str(make_patch_check(tmp_path, 0)))
    result = run_launcher("--dataset", dataset, "--output-dir", output, env=patched, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
