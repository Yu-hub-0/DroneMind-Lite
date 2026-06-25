# DroneMind-Lite Model Card

## Model

DroneMind-Lite is a compact UAV-domain causal language model for local learning and prototype assistant workflows.

Default local checkpoint:

- `out/full_sft_768.pth`

Base architecture:

- Decoder-only Transformer
- Dense 64M-class default configuration
- GQA attention
- RoPE/YaRN support
- SwiGLU MLP
- Optional MoE path
- Hugging Face-style config/model classes

## Intended Use

DroneMind-Lite is intended for educational and prototype UAV-domain assistant use:

- Drone preflight checklist explanations
- Battery and maintenance Q&A
- Mapping and survey parameter explanations
- General UAV safety and regulation guidance
- Local demo of tool-calling style interactions

## Out-of-Scope Use

Do not use the model as the sole source of truth for:

- Real-time airspace authorization
- Emergency flight decisions
- Legal compliance decisions
- Commercial flight operation approval
- Safety-critical autonomous control

Always verify airspace, weather, aircraft firmware, and local regulations with official sources before real flight.

## Training Data

Available local data includes:

- `data/pretrain_t2t_mini.jsonl`
- `data/sft_t2t_mini.jsonl`
- `data/sft_merged.jsonl`
- `data/dpo_drone_sample.jsonl`
- `data/rlaif_drone_sample.jsonl`
- `data/agent_drone.jsonl`
- `data/eval_drone_smoke.jsonl`

The DPO/RLAIF/Agent files are small project samples. They validate the pipeline, but they are not enough to claim strong aligned or autonomous agent capability.

## Evaluation

Lightweight smoke evaluation:

```powershell
python scripts\eval_drone_smoke.py --device cuda:0
```

Static project readiness:

```powershell
python scripts\quick_check.py
```

Recommended formal evaluation before release:

- UAV safety QA accuracy
- Maintenance diagnosis QA
- Mapping parameter calculation
- Regulation refusal and uncertainty behavior
- Tool-call JSON validity
- Tool selection accuracy
- Hallucination checks on unsupported model specs or regulations

## Known Limitations

- The current environment may not have Python dependencies installed.
- Current RL/Agent checkpoints are local validation artifacts.
- The smoke benchmark is intentionally small.
- Mock tool outputs are for training/demo only and are not connected to official airspace or weather services.

## Deployment Notes

Local API server:

```powershell
python scripts\serve_openai_api.py --load_from model --weight full_sft --device cuda:0
```

For production deployment, add authentication, rate limiting, request logging, safety filters, official data source integrations, and GPU monitoring.
