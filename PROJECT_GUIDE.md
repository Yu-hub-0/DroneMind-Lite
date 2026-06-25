# DroneMind-Lite 项目讲解

## 1. 项目定位

DroneMind-Lite 是一个面向无人机低空应用的小型语言模型工程。它把模型结构、训练流程、领域数据、工具调用、评测和部署脚本放在同一个仓库中，方便快速学习和二次开发。

项目适合以下场景：

- 无人机领域问答原型
- 本地小模型训练流程学习
- 课程设计或毕业设计展示
- 工具调用与 Agent 训练实验
- OpenAI 兼容 API 服务演示

## 2. 核心模块

### model

`model/model_dronemind.py` 定义了 DroneMind 模型结构，包含 decoder-only Transformer、GQA、RoPE/YaRN、SwiGLU、可选 MoE、KV cache 和 Hugging Face 风格的配置类。

`model/model_lora.py` 提供 LoRA 注入、加载和合并逻辑。

### dataset

`dataset/lm_dataset.py` 负责加载 JSONL 数据，并为预训练、SFT、DPO、RLAIF 和 Agent 数据构造训练样本及 loss mask。

### trainer

`trainer/` 中包含主要训练脚本：

- `train_pretrain.py`：预训练
- `train_full_sft.py`：监督微调
- `train_lora.py`：LoRA 微调
- `train_dpo.py`：偏好优化
- `train_ppo.py`：PPO
- `train_grpo.py`：GRPO/CISPO
- `train_agent.py`：工具调用场景的 Agent RL

### scripts

`scripts/` 提供工程辅助能力：

- `prepare_drone_data.py`：整理领域数据
- `quick_check.py`：项目结构和数据检查
- `run_dronemind.py`：统一阶段 runner
- `serve_openai_api.py`：OpenAI 兼容 API 服务
- `chat_api.py`：本地 API 客户端
- `eval_drone_smoke.py`：轻量领域评测
- `drone_tools.py`：无人机工具调用样例

## 3. 数据流程

1. 准备 UAV 领域文本和对话数据。
2. 使用 `prepare_drone_data.py` 生成 SFT、DPO、RLAIF 和 Agent 样例数据。
3. 通过 `train_pretrain.py` 训练基础语言能力。
4. 通过 `train_full_sft.py` 对齐对话格式和领域回答风格。
5. 通过 DPO、GRPO 或 Agent 脚本做进一步对齐验证。
6. 使用 `eval_llm.py`、API 服务或 Web Demo 做本地推理展示。

## 4. 运行顺序

先检查项目：

```powershell
python scripts\quick_check.py
```

准备数据：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\dronemind.ps1 prepare
```

运行轻量训练测试：

```powershell
python test_training.py
```

已有权重时运行推理：

```powershell
python eval_llm.py --load_from model --weight full_sft --device cuda
```

启动 API：

```powershell
python scripts\serve_openai_api.py --load_from model --weight full_sft --device cuda:0
```

## 5. 仓库瘦身

为了保证 GitHub 可以正常 clone，仓库不上传权重和大规模训练语料：

- `out/*.pth`
- `data/pretrain_t2t_mini.jsonl`
- `data/sft_t2t_mini.jsonl`

这些文件不是运行项目结构检查和轻量训练验证的必需文件。需要完整训练或推理时，可以在本地按相同路径放入自己的数据和权重。

## 6. 注意事项

这个项目只用于学习和原型验证。无人机真实飞行涉及安全、法规、天气、空域和设备状态，必须以官方平台和真实设备检查为准。
