# Third-party software

This repository does not vendor complete training or evaluation frameworks.

| Component | Use | Upstream |
| --- | --- | --- |
| ms-swift | Qwen3-VL supervised fine-tuning | <https://github.com/modelscope/ms-swift> |
| VLMEvalKit (Apache-2.0) | Distributed model inference and benchmark integration | <https://github.com/open-compass/VLMEvalKit> |
| Qwen3-VL-8B-Instruct | Base model | <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct> |

Installations remain subject to the upstream projects' licenses and model
terms. The revisions used by the released recipes are recorded in the
corresponding configuration and setup scripts.

The EgoTools dataset adapter and answer-extraction helper were adapted from
VLMEvalKit's video-dataset integration and Video-MME answer parsing, with
EgoTools-specific A–H options, prompts, option-text fallback, and metrics.
The retained upstream copyright notice and Apache-2.0 license are in
[`third_party/LICENSE-VLMEvalKit`](../third_party/LICENSE-VLMEvalKit).
That notice applies to the adapted upstream portions; it does not select a
license for the authors' original project code.
