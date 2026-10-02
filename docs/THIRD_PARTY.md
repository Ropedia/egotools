# Third-party Software

EgoTools uses upstream training and evaluation frameworks as external
dependencies. Their licenses and model terms continue to apply.

| Component | Use | Revision / source |
| --- | --- | --- |
| [MS-Swift](https://github.com/modelscope/ms-swift) | Supervised fine-tuning | `44c92c7cea08bf3b6e9f9b05ab182b6e81b0a7c7` |
| [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) | Model inference and benchmark integration | `e7d64cfa8f6036e1d00e21522aaee0102544ea25` |
| [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Base model | Official upstream model repository |

The training revision is pinned in
[`configs/training/requirements-cu124.txt`](../configs/training/requirements-cu124.txt).
The evaluation setup revision is specified in
[`scripts/setup_eval.sh`](../scripts/setup_eval.sh). The setup script creates
an ignored checkout at `third_party/VLMEvalKit/` or reuses a user-supplied
checkout; the framework itself is not included in the source package.

The EgoTools dataset adapter and answer extractor were adapted from VLMEvalKit's
video-dataset integration and Video-MME parsing, with EgoTools-specific A–H
options, prompts, option-text fallback, and metrics. The upstream copyright and
Apache-2.0 notice are retained in
[`third_party/LICENSE-VLMEvalKit`](../third_party/LICENSE-VLMEvalKit). That notice
covers the adapted portions and does not select a license for the authors'
original project code.
