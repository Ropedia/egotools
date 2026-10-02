"""Score the complete VLMEvalKit prediction table written by an EgoTools run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from egotools.manifest import read_manifest
from egotools.score import _join_manifest, _manifest_indexes, score_rows


def _locate_pred_file(work_dir: Path, model_name: str) -> Path:
    for extension in ("xlsx", "tsv", "jsonl"):
        matches = sorted(work_dir.rglob(f"{model_name}_*EgotoolsBench*.{extension}"))
        # VLMEvalKit also creates model-root symlinks to the prediction tables
        # stored under an evaluation-id directory. They identify the same file.
        matches = sorted(
            {
                path.resolve()
                for path in matches
                if path.is_file() and not path.stem.endswith(("_scored", "_acc", "_rating"))
            }
        )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"multiple prediction files under {work_dir}; score the desired file directly")
    raise FileNotFoundError(f"no EgoTools predictions for {model_name} under {work_dir}")


def aggregate_results(
    work_dir: Path, model_name: str, *, manifest_path: Path | None = None, input_settings: dict | None = None
) -> dict:
    """Score predictions against the selected inference manifest when supplied."""
    prediction_file = _locate_pred_file(work_dir, model_name)
    if prediction_file.suffix.lower() == ".xlsx":
        import pandas as pd

        # Preserve string identities such as "001" instead of inferring numbers.
        frame = pd.read_excel(prediction_file, dtype=object, keep_default_na=False)
        rows = frame.to_dict("records")
    else:
        _, rows = read_manifest(prediction_file)
    if any("prediction" not in row for row in rows):
        raise ValueError("predictions must contain a prediction field")
    if manifest_path is not None:
        _, manifest_rows = read_manifest(manifest_path)
        rows = _join_manifest(rows, manifest_rows)
    else:
        _manifest_indexes(rows)
    metrics, _ = score_rows(rows)
    if manifest_path is not None:
        metrics["coverage"] = {
            "predicted": len(rows),
            "expected": len(manifest_rows),
            "complete": len(rows) == len(manifest_rows),
        }
    score_file = Path(str(prediction_file) + "_egotools_score.json")
    score_file.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    results = {
        "run_kind": "real_model",
        "model": model_name,
        "n_evaluated": metrics["overall"]["total"],
        "metrics": metrics,
        "prediction_file": str(prediction_file),
    }
    if input_settings is not None:
        results["input_settings"] = input_settings
    output = work_dir / "results.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[aggregate] wrote {output}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("model_name")
    parser.add_argument("--manifest", type=Path, help="Require exactly one prediction per row in this manifest.")
    args = parser.parse_args(argv)
    aggregate_results(args.work_dir.expanduser().resolve(), args.model_name, manifest_path=args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
