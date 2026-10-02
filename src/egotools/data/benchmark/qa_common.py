"""Shared annotation, frame sampling, and Gemini helpers for QA normalization.

Inputs are read-only. Credentials come from the environment or an explicitly
selected local env file. No annotation service or private workspace is required.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────
# Data safety
# ─────────────────────────────────────────────────────────────────────

def assert_output_safe(input_dir: Path, output_dir: Path) -> None:
    """Refuse to run if --output-dir is the input dir or nested inside it.

    Resolves both sides through symlinks first. Exits the process via
    SystemExit on violation — never raises a catchable exception.
    """
    in_resolved = Path(input_dir).resolve(strict=False)
    out_resolved = Path(output_dir).resolve(strict=False)
    if out_resolved == in_resolved:
        raise SystemExit(
            f"DATA SAFETY: --output-dir ({out_resolved}) equals --input-dir. "
            "Originals must not be overwritten."
        )
    try:
        out_resolved.relative_to(in_resolved)
    except ValueError:
        return
    raise SystemExit(
        f"DATA SAFETY: --output-dir ({out_resolved}) is inside --input-dir "
        f"({in_resolved}). Originals must not be overwritten."
    )


# ─────────────────────────────────────────────────────────────────────
# Env / secrets
# ─────────────────────────────────────────────────────────────────────

def load_env(env_file: Path | None = None) -> None:
    """Load KEY=VALUE pairs from env_file into os.environ.

    Lightweight reader so we don't depend on python-dotenv being installed
    for this module. Existing env vars are NOT overridden — that lets
    a user override per-shell. Comments and blank lines are skipped. Quotes
    around values are stripped. The file is opened read-only; the key is
    never echoed.
    """
    if env_file is None:
        return
    p = Path(env_file)
    if not p.is_file():
        return
    with p.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def get_gemini_api_key() -> str:
    """Return GEMINI_API_KEY from env. Raise if missing. Never echo it."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Either export it, or ensure it's "
            "present in an explicitly supplied --env-file."
        )
    return key


# ─────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────

# Model aliases used by the optional Gemini quality checks.
MODEL_IDS: dict[str, str] = {
    "gemini-flash":         "gemini-3-flash-preview",
    "gemini-31-flash-lite": "gemini-3.1-flash-lite-preview",
    # Legacy aliases retained for backward compat — point at 2.5 lite which
    # is still live on v1beta if anyone explicitly requests it.
    "gemini-flash-lite":    "gemini-2.5-flash-lite",
    "gemini-25-flash":      "gemini-2.5-flash",
}

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


# ─────────────────────────────────────────────────────────────────────
# Video frame extraction
# ─────────────────────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    start: float,
    end: float,
    fps: float = 1.0,
    max_frames: int = 12,
    rotation: int = 0,
    resolution: int = 384,
) -> list[str]:
    """Extract evenly-spaced frames between [start, end] as base64 JPEG.

    Returns [] if the video doesn't exist or ffmpeg fails. Caller can pass
    the result list to GeminiClient.generate(frames=...) which inlines them
    as inline_data parts. Defaults trade quality for speed since QC
    distractor reasoning needs scene-level evidence, not pixel-level.
    """
    if not video_path:
        return []
    vp = Path(video_path)
    if not vp.exists():
        return []
    duration = max(0.0, float(end or 0) - float(start or 0))
    if duration <= 0:
        return []
    rotation = int(rotation or 0) % 360
    if rotation not in (0, 90, 180, 270):
        rotation = 0
    num_frames = min(int(duration * fps) + 1, max_frames)
    if num_frames < 1:
        num_frames = 1

    filters: list[str] = [f"fps={fps}"]
    if rotation == 90:
        filters.append("transpose=1")
    elif rotation == 180:
        filters.append("transpose=1,transpose=1")
    elif rotation == 270:
        filters.append("transpose=2")
    filters.append(f"scale={resolution}:-1")
    vf = ",".join(filters)

    tmp_dir = Path(tempfile.mkdtemp(prefix="qc_frames_"))
    try:
        out_pattern = str(tmp_dir / "frame_%04d.jpg")
        r = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{float(start):.3f}",
                "-t",  f"{duration:.3f}",
                "-i",  str(vp),
                "-vf", vf,
                "-q:v", "5",
                "-frames:v", str(num_frames),
                out_pattern,
            ],
            capture_output=True,
            timeout=max(30, int(duration) + 30),
        )
        if r.returncode != 0:
            return []
        frames: list[str] = []
        for fp in sorted(tmp_dir.glob("frame_*.jpg")):
            with fp.open("rb") as f:
                frames.append(base64.b64encode(f.read()).decode("ascii"))
        return frames
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────
# Gemini client (text and optional video frames)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class GeminiResponse:
    text: str
    raw: dict[str, Any]
    error: str | None = None


class GeminiClient:
    """Minimal Gemini REST client for QC stages.

    Defaults to temperature=0 to reduce sampling variability. Retries
    transient 429/500/502/503 with exponential backoff up to max_retries.
    Never logs the API key — passes it as a query param only.
    """

    def __init__(
        self,
        model: str = "gemini-flash",
        temperature: float = 0.0,
        max_output_tokens: int = 8192,
        # Enable Gemini's built-in thinking. "low" gives a lightweight
        # chain-of-thought without ballooning latency; thoughts are
        # discarded by .generate() so callers only see the final answer.
        thinking_level: str = "low",
        max_retries: int = 3,
        connect_timeout: int = 30,
        read_timeout: int = 600,
    ):
        if model not in MODEL_IDS:
            raise ValueError(
                f"Unknown model {model!r}. Choices: {sorted(MODEL_IDS)}"
            )
        self.model = model
        self.model_id = MODEL_IDS[model]
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.thinking_level = thinking_level
        self.max_retries = max_retries
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self._key = get_gemini_api_key()
        try:
            import requests  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "GeminiClient needs requests; install the optional egotools[data] dependencies."
            ) from e
        self._requests = requests

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        frames: list[str] | None = None,
    ) -> GeminiResponse:
        parts: list[dict[str, Any]] = []
        if system:
            parts.append({"text": f"System: {system}\n\nUser: {prompt}"})
        else:
            parts.append({"text": prompt})
        # Inline base64-jpeg video frames per Gemini v1beta multimodal spec
        for b64 in frames or []:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})

        gen_cfg: dict[str, Any] = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_output_tokens,
        }
        if self.thinking_level:
            # Return only final answer parts to the caller.
            gen_cfg["thinkingConfig"] = {
                "thinkingLevel": self.thinking_level,
                "includeThoughts": True,
            }
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": gen_cfg,
        }

        url = GEMINI_URL.format(model=self.model_id)
        last_err: str | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._requests.post(
                    url,
                    params={"key": self._key},
                    json=payload,
                    timeout=(self.connect_timeout, self.read_timeout),
                )
                if resp.status_code in (429, 500, 502, 503):
                    if attempt < self.max_retries - 1:
                        wait = min(int(resp.headers.get("Retry-After", 2 ** (attempt + 1))), 60)
                        time.sleep(wait)
                        continue
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    return GeminiResponse(text="", raw=data, error=str(data["error"].get("message")))
                text = ""
                for cand in data.get("candidates", []):
                    for p in cand.get("content", {}).get("parts", []):
                        if not p.get("thought"):
                            text += p.get("text", "")
                return GeminiResponse(text=text, raw=data)
            except self._requests.exceptions.ReadTimeout:
                last_err = "read_timeout"
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                    continue
                break
            except self._requests.exceptions.ConnectionError:
                last_err = "connection_error"
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                    continue
                break
            except Exception as e:  # noqa: BLE001 — final fallback
                last_err = type(e).__name__
                break
        return GeminiResponse(text="", raw={}, error=last_err or "unknown")


# ─────────────────────────────────────────────────────────────────────
# QA records
# ─────────────────────────────────────────────────────────────────────

@dataclass
class QARecord:
    qa_id: str
    source_file: str
    source_id: str
    canonical_video_id: str
    display_name: str
    annotation_id: str
    question: str
    answer: str
    distractors: list[str]
    annotator_id: Any
    evals: list[dict[str, Any]]
    raw: dict[str, Any]


# ─────────────────────────────────────────────────────────────────────
# Eval cache lookup
# ─────────────────────────────────────────────────────────────────────

def _normq(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def latest_matching_eval(record: QARecord) -> dict[str, Any] | None:
    """Return the most recent eval whose snapshot matches the current
    question, answer, and option texts. None if no match.

    Used during normalization to skip Gemini calls when a stored eval is still
    authoritative for the current QA shape.
    """
    cur = (_normq(record.question), _normq(record.answer), len(record.distractors) + 1)
    matches: list[tuple[str, dict[str, Any]]] = []
    for ev in record.evals:
        snap = (
            _normq(str(ev.get("question") or "")),
            _normq(str(ev.get("answer") or "")),
            int(ev.get("num_choices") or len(ev.get("options") or [])),
        )
        previous_options = sorted(_normq(str(option.get("text") or "")) for option in ev.get("options", []))
        current_options = sorted(_normq(text) for text in [record.answer, *record.distractors])
        if snap == cur and previous_options == current_options:
            matches.append((str(ev.get("evaluated_at") or ""), ev))
    if not matches:
        return None
    matches.sort(key=lambda kv: kv[0])
    return matches[-1][1]


# ─────────────────────────────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────────────────────────────

def write_normalization_report(
    output_dir: Path,
    args: dict[str, Any],
    counts: dict[str, int],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write normalization_report.json with args, counts, git sha, timestamps.

    Never includes secrets — caller is responsible for not passing keys
    in `args`.
    """
    from datetime import datetime, timezone
    git_sha = ""
    try:
        r = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            git_sha = r.stdout.strip()
    except Exception:  # noqa: BLE001 — git missing is non-fatal
        pass
    # Coerce non-JSON types (e.g. PosixPath, set) to strings for portability.
    def _coerce(v: Any) -> Any:
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if isinstance(v, dict):
            return {str(k): _coerce(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_coerce(x) for x in v]
        if isinstance(v, set):
            return sorted(_coerce(x) for x in v)
        return str(v)

    out = {
        "args": _coerce(args),
        "counts": _coerce(counts),
        "git_sha": git_sha,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "extra": _coerce(extra or {}),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "normalization_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    return report_path
