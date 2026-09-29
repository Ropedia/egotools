"""Score the complete VLMEvalKit prediction table written by an EgoTools run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from egotools.evaluation.datasets.egotools import EgotoolsBench


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


def aggregate_results(work_dir: Path, model_name: str) -> dict:
    prediction_file = _locate_pred_file(work_dir, model_name)
    metrics = EgotoolsBench.evaluate(prediction_file)
    results = {
        "run_kind": "real_model",
        "model": model_name,
        "n_evaluated": metrics["overall"]["total"],
        "metrics": metrics,
        "prediction_file": str(prediction_file),
    }
    output = work_dir / "results.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[aggregate] wrote {output}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("model_name")
    args = parser.parse_args(argv)
    aggregate_results(args.work_dir.expanduser().resolve(), args.model_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
