# Third-party sources and licenses

Original RetailAgent-RL code is provided under Apache-2.0. Dependencies, copied source, data and model weights retain their own licenses; the root license does not relicense them.

| Component | Pinned source | License / distribution |
|---|---|---|
| tau2-bench runtime, policies, synthetic domain data | [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench/tree/f8de30c298689cbe0117d76a378e7315a17e5bd8) | MIT; see [vendored LICENSE](third_party/tau2-bench/LICENSE). A runtime subset is vendored without upstream leaderboard/results files. |
| Native community single-action prompt and action-format adaptation | [jbarnes850/Tau2-RL-Pipeline](https://github.com/jbarnes850/Tau2-RL-Pipeline/tree/c0d434b453719e149aac70452325f6e04ac086cf) | Apache-2.0; [retained license](third_party/licenses/Tau2-RL-Pipeline.LICENSE). Local adaptation: `src/retail_agent/community_protocol.py`. |
| Original SFT/RL data and Retail database snapshots | [inclusionAI/AReaL-tau2-data](https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data/tree/86971dc03da6e7c1a7933295e05b84aab8215386) | Apache-2.0 according to fixed source metadata. Derived views preserve original rows and provenance. |
| Original base model | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B/tree/1cfa9a7208912126459214e8b04321603b3df60c) | Apache-2.0; weight downloads remain outside Git. |
| Current community checkpoint | [Jarrodbarnes/Qwen3-4B-tau2-grpo-v1](https://huggingface.co/Jarrodbarnes/Qwen3-4B-tau2-grpo-v1/tree/9902d3e07264157083e3d2c1eaea3735612f49b9) | Source weights retain their license. The MLX copy is a 4-bit conversion, not a new training contribution. Source file hashes are in its conversion manifest. |
| Previous independent model experiment | [MMOPD/Qwen3-4B-OT3-tau2-2ep](https://huggingface.co/MMOPD/Qwen3-4B-OT3-tau2-2ep) | Linked as an upstream model; no weights are included in Git. |
| verl | [volcengine/verl](https://github.com/volcengine/verl) | Apache-2.0. The optional archive retains LICENSE and Notice.txt. Local source revision was unavailable; the runtime source, configuration, pyproject and uv.lock are hash-locked by `configs/verl/source_lock.json`. |

Evaluation examples contain synthetic benchmark identities and orders. They are not production customer records. Do not infer a new model training result or official benchmark endorsement from the reproduction artifacts.
