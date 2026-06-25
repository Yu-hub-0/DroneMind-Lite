# DroneMind Delivery Checklist

Use this checklist before calling the project complete.

## Repository Assets

- [x] DroneMind model implementation: `model/model_dronemind.py`
- [x] Tokenizer files: `model/tokenizer.json`, `model/tokenizer_config.json`
- [x] Pretraining script: `trainer/train_pretrain.py`
- [x] SFT script: `trainer/train_full_sft.py`
- [x] LoRA script: `trainer/train_lora.py`
- [x] DPO script: `trainer/train_dpo.py`
- [x] GRPO/Agent scripts: `trainer/train_grpo.py`, `trainer/train_agent.py`
- [x] Data preparation script: `scripts/prepare_drone_data.py`
- [x] Unified runner: `scripts/run_dronemind.py`
- [x] Windows runner: `scripts/dronemind.ps1`
- [x] Static check: `scripts/quick_check.py`
- [x] Smoke evaluation: `scripts/eval_drone_smoke.py`
- [x] API server: `scripts/serve_openai_api.py`
- [x] API client: `scripts/chat_api.py`
- [x] UAV mock tools: `scripts/drone_tools.py`

## Data Assets

- [x] Pretraining data available
- [x] SFT source data available
- [x] Merged SFT data generated
- [x] DPO sample data generated
- [x] RLAIF sample data generated
- [x] Agent sample data generated
- [x] Smoke evaluation data available

## Model Artifacts

- [x] `out/pretrain_768.pth`
- [x] `out/full_sft_768.pth`
- [ ] Optional `out/lora_drone_768.pth`
- [ ] Optional `out/dpo_768.pth`
- [ ] Optional `out/grpo_768.pth`
- [ ] Optional `out/agent_768.pth`

## Verification

- [x] Python syntax check passes
- [x] `scripts/quick_check.py` passes static checks
- [x] `scripts/dronemind.ps1 all -DryRun` produces commands
- [ ] Python dependencies installed in the active environment
- [ ] `python test_training.py` passes
- [ ] `python scripts\eval_drone_smoke.py --device cuda:0` passes
- [ ] API server starts and answers a chat request

## Documentation

- [x] `README_DRONEMIND.md`
- [x] `TRAINING_PIPELINE.md`
- [x] `PROJECT_STATUS.md`
- [x] `MODEL_CARD_DRONEMIND.md`
- [x] `DELIVERY_CHECKLIST.md`

## Final Completion Criteria

The basic project is complete when:

- dependencies are installed,
- `test_training.py` passes,
- `eval_drone_smoke.py` runs on `out/full_sft_768.pth`,
- API server responds locally,
- all docs reflect the actual trained artifacts.

The advanced project is complete when at least one post-SFT alignment stage
also has a trained checkpoint and evaluation result.
