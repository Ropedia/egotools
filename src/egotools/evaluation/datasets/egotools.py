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

import pandas as pd

from egotools.answers import extract_letter as extract_letter_ah
from egotools.evaluation.data import _is_empty_option, build_mcq_prompt_text, resolve_dataset_root
from egotools.score import score_rows

__all__ = ["EgotoolsBench", "build_mcq_prompt_text", "extract_letter_ah", "resolve_dataset_root"]

EGOTOOLS_VERSIONS = ("custom", "v1_0", "v1_20260428", "v2_20260430", "v3_20260503", "v4_20260504", "v5_20260505")

# Class-level scoring remains available without the inference dependency.
try:
    from vlmeval.dataset.video_base import VideoBaseDataset as _VLMEvalVideoBase
except ModuleNotFoundError as exc:
    if exc.name != "vlmeval":
        raise

    class _VLMEvalVideoBase:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            raise ModuleNotFoundError("dataset inference requires VLMEvalKit; run scripts/setup_eval.sh")


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
        # Upstream infers TSV types, which changes IDs such as "001" or "NA"
        # and numeric option text before inference writes them into its results.
        self.data = pd.read_csv(self.data_file, sep="\t", dtype=str, keep_default_na=False)
        if self.data.empty:
            raise ValueError("manifest contains no questions")
        required = {"index", "video", "question", "answer"}
        missing = required.difference(self.data.columns)
        if missing:
            raise ValueError(f"manifest is missing columns: {', '.join(sorted(missing))}")
        if self.data["index"].duplicated().any():
            raise ValueError("manifest contains duplicate index values")
        self.videos = sorted(set(self.data["video"]))

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
            line = line.astype(object).where(pd.notna(line), None)
        row = dict(line)

        prompt_text, _kept = build_mcq_prompt_text(row)

        message: list[dict] = []
        rel_video = self._row_video_path(row)
        abs_video = str(Path(self.data_root) / rel_video) if rel_video else ""

        if video_llm and abs_video:
            message.append({"type": "video", "value": abs_video})
        elif not video_llm and abs_video:
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
        if not {"prediction", "answer"}.issubset(df.columns):
            raise ValueError("predictions must contain prediction and answer columns")
        rows = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")
        metrics, _ = score_rows(rows)

        if write_alongside is not None:
            out = Path(str(write_alongside) + "_egotools_score.json")
            out.write_text(json.dumps(metrics, indent=2))

        return metrics
