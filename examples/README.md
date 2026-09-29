# Offline examples

These two synthetic questions demonstrate the manifest schema, mock evaluation,
and scoring. They are not benchmark samples. The video paths are placeholders;
no videos are included or needed for these commands.

From the repository root, after installing `python -m pip install -e '.[eval]'`:

```bash
egotools-validate-manifest examples/benchmark/manifest.tsv --expected-rows 2
egotools-evaluate \
  --dataset-root examples/benchmark --bench-version example \
  --model mock --smoke --mock-mode fixed --mock-letter A \
  --results-dir outputs/example
egotools-score examples/predictions.jsonl \
  --manifest examples/benchmark/manifest.tsv --output-dir outputs/scoring
```

The fixed mock answers `A` for both questions and scores **1/2**. The supplied
predictions contain the two correct answers and score **2/2**. These values
check the format and execution path; they are not model performance results.

For actual videos and model inference, see [the evaluation guide](../docs/EVALUATION.md).
