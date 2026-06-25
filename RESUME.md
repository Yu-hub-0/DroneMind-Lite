# 项目二：DroneMind 低空/无人机垂域小模型训练与 Tool-Use Agent

**时间：2026.02 - 2026.05**

## 项目描述

基于 DroneMind 改造轻量级 decoder-only LLM，面向无人机飞行安全、法规合规、航测、设备维护等低空场景，构建领域数据、模型训练、对齐优化、工具调用与推理部署闭环。项目完成预训练、SFT、DPO、GRPO 与 Agent RL 本地验证权重产出，并搭建低空 Tool-Use 工具链、OpenAI 兼容 API 和 Streamlit WebUI，用于低空场景问答、工具调用和本地快速验证。

## 技术栈

PyTorch、Decoder-only Transformer、GQA、RoPE/YaRN、SwiGLU、FlashAttention、LoRA、DPO/GRPO、Tool-Use Agent、FastAPI、Streamlit

## 主要工作

**模型架构搭建：** 参考 LLaMA/DroneMind 实现 HuggingFace 风格轻量 Decoder-only LLM，支持 GQA 分组注意力、RoPE/YaRN 长上下文外推、SwiGLU 前馈网络、KV Cache 推理与可选 MoE 结构，并集成 FlashAttention 加速训练和推理；默认配置为 hidden size 768、8 层结构，Dense 版本约 64M 参数。

**训练管线搭建：** 构建低空三阶段数据集，覆盖预训练语料、2000 条领域 SFT 指令数据、DPO/RLAIF 偏好样例及 Agent 多轮工具调用数据；完成 pretrain -> SFT -> eval -> API 基础闭环，产出 `pretrain_768.pth` 与 `full_sft_768.pth` 权重，支持 bf16、梯度累积、断点续训和统一 runner。

**对齐与 Tool-Use Agent：** 完成 LoRA、DPO、GRPO/CISPO、Agent RL 训练入口与本地验证训练，设计工具选择、参数正确性、任务完成度和 KL 约束等奖励逻辑；接入禁飞区查询、飞行天气评估、续航估算、航线规划、机型参数查询、无人机单位换算 6 类低空工具，并新增 Tool-Use 评测脚本，工具选择、参数有效性和语义覆盖验证均为 1.0。

**工程部署与验证：** 搭建 OpenAI 兼容 Chat Completions API，支持 FastAPI 流式输出；实现命令行推理、Streamlit WebUI、静态自检、无人机领域 smoke evaluation 和 Tool-Use validation report，便于本地演示、模型回归验证和部署联调。

## 可展示指标

- 模型规模：Dense 版本约 64M 参数，默认 hidden size 768、8 layers。
- 数据规模：SFT 合并数据 2000 条，RLAIF 样例 256 条，DPO 样例 4 条，Agent 工具调用样例 3 条。
- 已产出权重：`out/pretrain_768.pth`、`out/full_sft_768.pth`、`out/dpo_768.pth`、`out/grpo_768.pth`、`out/agent_768.pth`。
- 对齐训练：DPO 1 epoch / 2 steps，`dpo_loss=0.5365`；GRPO 快速验证 `Reward=1.5500`、`Actor Loss=0.1199`；Agent RL 快速验证 `Reward=1.0236`、`Loss=0.3940`。
- 工具能力：6 类低空工具，`reports/tooluse_eval_report.json` 中工具选择、参数有效性、语义覆盖均为 1.0。
- 工程入口：统一 runner 支持 `prepare/pretrain/sft/lora/dpo/grpo/agent/eval/tool_eval/serve` 阶段。
