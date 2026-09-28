"""EgoTools video MCQ adapter for an externally installed VLMEvalKit.

A dataset root contains manifest.tsv and relative video paths. Empty trailing
A-H option cells are omitted from the prompt. The full/clip choice selects the
corresponding manifest column; clip mode falls back to full video when absent.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from egotools.answers import extract_letter as extract_letter_ah

EGOTOOLS_VERSIONS = ("custom", "v1_0", "v1_20260428", "v2_20260430", "v3_20260503", "v4_20260504", "v5_20260505")
_OPTION_LETTERS = list("ABCDEFGH")


def resolve_dataset_root(dataset_root: str | Path | None, version: str = "custom") -> Path:
    """Resolve an explicit root or a parent containing a named version."""
    value = dataset_root or os.environ.get("EGOTOOLS_BENCH_ROOT")
    if not value:
        raise ValueError("set --dataset-root or EGOTOOLS_BENCH_ROOT to a directory containing manifest.tsv")
    root = Path(value).expanduser().resolve()
    if not (root / "manifest.tsv").is_file() and (root / version / "manifest.tsv").is_file():
        root /= version
    if not (root / "manifest.tsv").is_file():
        raise FileNotFoundError(f"manifest.tsv not found under {root}; materialize your downloaded manifest first")
    return root


# Prompt template: A..H multiple-choice over a video.
_MCQ_PROMPT_TEMPLATE = (
    "You are watching an egocentric (first-person) video of a user using "
    "tools or performing manual tasks.\n"
    "Answer the following multiple-choice question about the video.\n"
    "Respond with ONLY the single uppercase letter of the best option "
    "(one of: {letters}). Do not include any explanation.\n\n"
    "Question: {question}\n"
    "{options}\n"
    "Answer: "
)


def _is_empty_option(value: Any) -> bool:
    """An option cell is empty if it's NaN, empty string, or whitespace."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    s = str(value).strip()
    return s == "" or s.lower() in ("nan", "none")


def build_mcq_prompt_text(row: dict) -> tuple[str, list[str]]:
    """Build the textual MCQ prompt for a single row.

    Returns (prompt_text, kept_letters) where kept_letters is the list of
    A..H letters that have non-empty options.
    """
    kept_letters: list[str] = []
    option_lines: list[str] = []
    for letter in _OPTION_LETTERS:
        val = row.get(letter)
        if _is_empty_option(val):
            continue
        kept_letters.append(letter)
        option_lines.append(f"{letter}. {str(val).strip()}")

    if not kept_letters:
        # Defensive fallback: a valid row has at least two options; if we
        # somehow get a row with zero options we still produce a coherent
        # prompt rather than crashing. Evaluation will mark wrong.
        kept_letters = ["A"]
        option_lines = ["A. (no options provided)"]

    prompt_text = _MCQ_PROMPT_TEMPLATE.format(
        letters=", ".join(kept_letters),
        question=str(row.get("question", "")).strip(),
        options="\n".join(option_lines),
    )
    return prompt_text, kept_letters


def extract_letter_ah_from_row(row: Any) -> str:
    """Extract a prediction letter using both the response and row options."""
    get = row.get if hasattr(row, "get") else lambda key, default=None: getattr(row, key, default)
    options = {letter: get(letter, "") for letter in _OPTION_LETTERS}
    return extract_letter_ah(get("prediction", ""), options=options)


# ----------------------------------------------------------------------
# Dataset class
# ----------------------------------------------------------------------

# Lazy import VideoBaseDataset only when actually constructing the dataset,
# so this module can be imported without vlmeval installed. We still
# subclass at class-definition time once vlmeval is available; if it
# isn't, we fall back to a thin shim base class that only supports the
# methods needed by the smoke runner.

try:
    from vlmeval.dataset.video_base import VideoBaseDataset as _VLMEvalVideoBase

    _VLMEVAL_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only outside the conda env
    _VLMEVAL_AVAILABLE = False

    class _VLMEvalVideoBase:  # type: ignore[no-redef]
        """Minimal stand-in used only when vlmeval is not importable.

        It deliberately implements *just enough* surface area for the
        smoke runner to construct an EgotoolsBench and iterate rows.
        Real evaluation runs require the actual conda env where
        ``import vlmeval`` works.
        """

        MODALITY = "VIDEO"
        DEFAULT_JUDGE = "exact_matching"

        def __init__(self, dataset="EgotoolsBench", pack=False, nframe=0, fps=-1):
            self.dataset_name = dataset
            self.pack = pack
            self.nframe = nframe
            self.fps = fps
            ret = self.prepare_dataset(dataset)
            assert ret is not None, "prepare_dataset() returned None"
            self.data_root = ret["root"]
            self.data_file = ret["data_file"]
            self.data = pd.read_csv(self.data_file, sep="\t")
            if "index" not in self.data.columns:
                self.data["index"] = list(range(len(self.data)))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return dict(self.data.iloc[idx])


class EgotoolsBench(_VLMEvalVideoBase):
    """VLMEvalKit dataset adapter for the egotools MCQ benchmark.

    Parameters
    ----------
    dataset : str
        VLMEvalKit-facing dataset name. Should be one of the strings in
        :meth:`supported_datasets`. The default is ``"EgotoolsBench"``.
    version : str
        Dataset label, or version subfolder under an explicit dataset root.
    video_mode : {"full", "clip"}
        Selects which column of the manifest is treated as the active
        video path. ``"full"`` (default) uses the ``video`` column (the
        full source recording). ``"clip"`` uses the ``clip_video`` column
        (a per-QA short clip if available).
    nframe, fps, pack : int, float, bool
        Standard VLMEvalKit knobs forwarded to :class:`VideoBaseDataset`.
    """

    TYPE = "Video-MCQ"
    MODALITY = "VIDEO"
    DEFAULT_JUDGE = "exact_matching"

    _DEFAULT_VERSION = "custom"
    _SUPPORTED_VIDEO_MODES = ("full", "clip")

    def __init__(
        self,
        dataset: str = "EgotoolsBench",
        version: str = _DEFAULT_VERSION,
        video_mode: str = "full",
        nframe: int = 0,
        fps: float = -1,
        pack: bool = False,
        dataset_root: str | None = None,
    ):
        if video_mode not in self._SUPPORTED_VIDEO_MODES:
            raise ValueError(f"video_mode must be one of {self._SUPPORTED_VIDEO_MODES}, got {video_mode!r}")
        # Map dataset alias -> version. We allow callers to pass either
        # "EgotoolsBench" + version=..., or a specific alias like
        # "EgotoolsBench_v1_20260428_full" which is what VLMEvalKit's
        # build_dataset() will use.
        alias_meta = self._parse_alias(dataset)
        if alias_meta is not None:
            version = alias_meta.get("version", version)
            video_mode = alias_meta.get("video_mode", video_mode)
            nframe = alias_meta.get("nframe", nframe)
            fps = alias_meta.get("fps", fps)

        self.version = version
        self.video_mode = video_mode
        self._dataset_root_override = Path(dataset_root) if dataset_root is not None else None

        if pack:
            raise ValueError("EgoTools evaluates one question per row; pack mode is unsupported")
        if nframe < 0 or (fps > 0 and nframe > 0):
            raise ValueError("choose either a nonnegative nframe or positive fps")
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)
        if self.data.empty:
            raise ValueError("manifest contains no questions")
        required = {"index", "video", "question", "answer"}
        missing = required.difference(self.data.columns)
        if missing:
            raise ValueError(f"manifest is missing columns: {', '.join(sorted(missing))}")
        if self.data["index"].duplicated().any():
            raise ValueError("manifest contains duplicate index values")

    # ------------------------------------------------------------------
    # VLMEvalKit registry hooks
    # ------------------------------------------------------------------

    @classmethod
    def supported_datasets(cls) -> list[str]:
        names = ["EgotoolsBench"]
        versions = set(EGOTOOLS_VERSIONS) | {os.environ.get("EGOTOOLS_BENCH_VERSION", "custom")}
        frames = sorted({8, 32, 64, int(os.environ.get("EGOTOOLS_NFRAME", "64"))} - {0})
        rates = {1.0, float(os.environ.get("EGOTOOLS_FPS", "1"))}
        for version in sorted(versions):
            for mode in cls._SUPPORTED_VIDEO_MODES:
                base = f"EgotoolsBench_{version}_{mode}"
                names.append(base)
                names.extend(f"{base}_{count}frame" for count in frames)
                names.extend(f"{base}_{rate:g}fps" for rate in sorted(rates) if rate > 0)
        return names

    @classmethod
    def _parse_alias(cls, dataset_name: str) -> dict | None:
        match = re.fullmatch(r"EgotoolsBench_(.+)_(full|clip)(?:_([0-9]+)frame|_([0-9.]+)fps)?", dataset_name)
        if match is None:
            return None
        version, video_mode, count, rate = match.groups()
        result = {"version": version, "video_mode": video_mode}
        if count is not None:
            result["nframe"] = int(count)
        if rate is not None:
            result["fps"] = float(rate)
        return result

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def _resolve_dataset_root(self) -> Path:
        return resolve_dataset_root(self._dataset_root_override, self.version)

    def prepare_dataset(self, dataset_name: str = "EgotoolsBench") -> dict:
        root = self._resolve_dataset_root()
        return {"root": str(root), "data_file": str(root / "manifest.tsv")}

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _row_video_path(self, row: dict) -> str:
        """Return the video path (relative to data_root) for a row."""
        col = "video" if self.video_mode == "full" else "clip_video"
        rel = row.get(col)
        if _is_empty_option(rel):
            # Fall back to the other column if the selected one is missing.
            other = "clip_video" if self.video_mode == "full" else "video"
            rel = row.get(other)
        return str(rel) if rel is not None else ""

    def build_prompt(self, line, video_llm: bool = True):
        """Build a VLMEvalKit message list for a single row.

        Mirrors VideoMME.build_prompt: returns a list of
        ``{"type": ..., "value": ...}`` dicts. When ``video_llm`` is True
        a single ``video`` element is attached. Otherwise the model is
        expected to consume sampled frames; we delegate to the upstream
        ``save_video_frames`` helper provided by VideoBaseDataset.
        """
        if isinstance(line, int):
            assert line < len(self), f"index {line} out of range"
            line = self.data.iloc[line]
        if isinstance(line, pd.Series):
            row = dict(line)
        else:
            row = dict(line)

        prompt_text, _kept = build_mcq_prompt_text(row)

        message: list[dict] = []
        rel_video = self._row_video_path(row)
        abs_video = str(Path(self.data_root) / rel_video) if rel_video else ""

        if video_llm and abs_video:
            message.append({"type": "video", "value": abs_video})
        elif not video_llm and abs_video:
            if not _VLMEVAL_AVAILABLE:
                raise RuntimeError("frame extraction requires VLMEvalKit")
            if not rel_video.lower().endswith(".mp4"):
                raise ValueError("the upstream frame extractor requires .mp4 video paths")
            if self.nframe <= 0 and self.fps <= 0:
                raise ValueError("frame-based models require --nframe or --fps")
            stem = rel_video[:-4]
            for frame_path in self.save_video_frames(stem):
                message.append({"type": "image", "value": frame_path})

        message.append({"type": "text", "value": prompt_text})
        return message

    # ------------------------------------------------------------------
    # Evaluation: exact-match A..H letter -> overall + per-qtype accuracy
    # ------------------------------------------------------------------

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs) -> dict:
        """Score a VLMEvalKit prediction file (xlsx/tsv/json/jsonl).

        Returns a metrics dict with ``overall`` accuracy and a
        ``per_qtype`` mapping. The dict is also written next to the
        prediction file as ``<eval_file>_egotools_score.json`` for
        downstream tooling.
        """
        eval_path = Path(eval_file)
        if not eval_path.exists():
            raise FileNotFoundError(f"eval_file does not exist: {eval_path}")

        ext = eval_path.suffix.lower().lstrip(".")
        if ext == "xlsx":
            data = pd.read_excel(eval_path)
        elif ext == "tsv":
            data = pd.read_csv(eval_path, sep="\t")
        elif ext == "json":
            data = pd.read_json(eval_path)
        elif ext == "jsonl":
            data = pd.read_json(eval_path, lines=True)
        else:
            raise ValueError(f"Unsupported eval_file extension: .{ext}")

        if "prediction" not in data.columns:
            raise ValueError("eval_file is missing the 'prediction' column")
        if "answer" not in data.columns:
            raise ValueError("eval_file is missing the 'answer' column")

        return cls.score_dataframe(data, write_alongside=eval_path)

    @classmethod
    def score_dataframe(cls, df: pd.DataFrame, write_alongside: Path | None = None) -> dict:
        """Compute overall + per-qtype accuracy from a scored dataframe.

        Works on a copy of ``df`` and preserves the caller's dataframe.
        If ``write_alongside`` is given, also writes
        ``<that_path>_egotools_score.json`` with the metrics dict.
        """
        df = df.copy()
        if not {"prediction", "answer"}.issubset(df.columns):
            raise ValueError("predictions must contain prediction and answer columns")
        df["answer"] = df["answer"].astype(str).str.strip().str.upper()
        if not df["answer"].isin(_OPTION_LETTERS).all():
            raise ValueError("predictions contain an invalid gold answer")
        df["predicted_letter"] = df.apply(extract_letter_ah_from_row, axis=1)
        df["score"] = (df["predicted_letter"] == df["answer"].astype(str)).astype(int)

        def grouped(column: str) -> dict:
            result = {}
            for name, group in df.groupby(df[column].fillna("_unknown").replace("", "_unknown")):
                count = len(group)
                correct = int(group["score"].sum())
                result[str(name)] = {"correct": correct, "total": count, "accuracy": correct / count}
            return result

        total = int(len(df))
        correct = int(df["score"].sum())
        overall_acc = correct / total if total else 0.0

        # Coverage: how often did we manage to extract any letter at all?
        extracted = int((df["predicted_letter"] != "").sum())

        metrics = {
            "overall": {
                "correct": correct,
                "total": total,
                "accuracy": overall_acc,
            },
            "extraction_coverage": {
                "extracted": extracted,
                "total": total,
                "rate": (extracted / total) if total else 0.0,
            },
            "per_qtype": grouped("qtype") if "qtype" in df.columns else {},
        }

        for column in ("research_track_id", "research_track", "track"):
            if column in df.columns and df[column].fillna("").astype(str).str.strip().any():
                metrics["per_track"] = grouped(column)
                break

        if write_alongside is not None:
            out = Path(str(write_alongside) + "_egotools_score.json")
            out.write_text(json.dumps(metrics, indent=2))

        return metrics
