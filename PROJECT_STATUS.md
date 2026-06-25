# DroneMind-Lite Project Status

## Current State

This repository contains a working UAV-domain small language model stack. It includes model code, tokenizer files, local data, training scripts, inference utilities, API serving, tool-use examples, evaluation scripts, and validation checkpoints.

The basic train-and-infer loop is present:

- Tokenizer files are available in `model/`.
- UAV pretraining and SFT data are available in `data/`.
- Validation checkpoints are available in `out/`.
- Core training scripts are available in `trainer/`.
- CLI inference, OpenAI-compatible API serving, and Streamlit demo scripts are available in `eval_llm.py` and `scripts/`.
- Static readiness checks are available in `scripts/quick_check.py`.
- UAV-domain smoke evaluation lives in `data/eval_drone_smoke.jsonl` and `scripts/eval_drone_smoke.py`.
- Unified config and runner live in `configs/dronemind_64m.json`, `scripts/run_dronemind.py`, and `scripts/dronemind.ps1`.
- UAV tool definitions and mock execution live in `scripts/drone_tools.py`.
- DPO, GRPO, Agent RL, and Tool-Use validation are wired into the unified runner.
- Tool-use validation output is available at `reports/tooluse_eval_report.json`.

Existing weights:

- `out/pretrain_768.pth`
- `out/full_sft_768.pth`
- `out/dpo_768.pth`
- `out/grpo_768.pth`
- `out/agent_768.pth`

## Complete Enough

- Model architecture: decoder-only Transformer with GQA, RoPE/YaRN support, SwiGLU, optional MoE, KV cache generation, and Hugging Face-style config/model classes.
- Pretraining path: `trainer/train_pretrain.py`.
- Full SFT path: `trainer/train_full_sft.py`.
- Dataset loaders: pretraining, SFT, DPO, and RLAIF/agent-related dataset classes in `dataset/lm_dataset.py`.
- Basic local inference: `eval_llm.py`.
- API serving: `scripts/serve_openai_api.py`.
- Automated pipeline skeleton: `run_pipeline.py`.
- CPU-friendly static checks: `scripts/quick_check.py`.
- Lightweight domain smoke evaluation: `scripts/eval_drone_smoke.py`.
- Unified stage runner: `scripts/run_dronemind.py`.
- Windows convenience runner: `scripts/dronemind.ps1`.
- Tool-use validation: `scripts/eval_drone_tooluse.py`.

## Still Needs Work

- Dependency versions should be verified in the target Python environment before training or inference.
- DPO/RLAIF/Agent data currently starts as small sample data. It is enough for pipeline validation; stronger final model quality needs larger alignment data and longer training.
- Evaluation is still light. A smoke evaluation exists, but a formal domain benchmark should be expanded before any public quality claim.
- Some scripts still depend on large GPU memory and are not suitable as quick CPU-only checks.

## Recommended Verification

```powershell
python -m pip install -r requirements.txt
python scripts\quick_check.py
python test_training.py
python scripts\eval_drone_smoke.py --device cuda:0
```
