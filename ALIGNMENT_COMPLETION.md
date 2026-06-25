# DroneMind Alignment and Tool-Use Completion Notes

## What was completed

- Added GRPO and Agent stages to the unified runner:
  - `python scripts/run_dronemind.py grpo --device cuda:0`
  - `python scripts/run_dronemind.py agent --device cuda:0`
- Added `tool_eval` stage for reproducible UAV tool-chain validation:
  - `python scripts/run_dronemind.py tool_eval`
- Connected Agent RL tool execution to the UAV tool implementations in
  `scripts/drone_tools.py`.
- Added backward-compatible tool aliases used by early Agent data:
  - `get_drone_specs -> lookup_drone_specs`
  - `convert_drone_units -> convert_uav_units`
- Added static checks for the new Tool-Use evaluator and optional alignment
  weights.

## Current verified artifacts

- Base weights:
  - `out/pretrain_768.pth`
  - `out/full_sft_768.pth`
- Alignment weights:
  - `out/dpo_768.pth`
  - `out/grpo_768.pth`
  - `out/agent_768.pth`
- Training and alignment data:
  - `data/sft_merged.jsonl`: 2000 records
  - `data/dpo_drone_sample.jsonl`: 4 records
  - `data/rlaif_drone_sample.jsonl`: 256 records
  - `data/agent_drone.jsonl`: 3 records
- Tool-Use validation report:
  - `reports/tooluse_eval_report.json`
  - Tool selection: 1.0
  - Parameter validity: 1.0
  - Semantic GT coverage: 1.0

## Commands to reproduce

Static readiness:

```powershell
python scripts\quick_check.py
```

Tool-Use validation:

```powershell
python scripts\eval_drone_tooluse.py
```

DPO training:

```powershell
python scripts\run_dronemind.py dpo --device cuda:0
```

GRPO training:

```powershell
python scripts\run_dronemind.py grpo --device cuda:0
```

Agent RL training:

```powershell
python scripts\run_dronemind.py agent --device cuda:0
```

## Verified training runs

- DPO: 1 epoch / 2 steps, final `dpo_loss=0.5365`, saved `out/dpo_768.pth`.
- GRPO: 1 quick-validation step with rule-based reward model, `Reward=1.5500`,
  `Actor Loss=0.1199`, saved `out/grpo_768.pth`.
- Agent RL: 1 quick-validation step with UAV tool reward path, `Reward=1.0236`,
  `Loss=0.3940`, saved `out/agent_768.pth`.

The quick-validation GRPO and Agent configurations use `max_steps=1` and
`reward_model_path=rule` so the project can be reproduced locally without
downloading a large external reward model. Increase `max_steps` or replace
`reward_model_path` for longer training.
