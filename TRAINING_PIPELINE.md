# DroneMind 训练流水线

这份文档描述一个完整的无人机垂域模型训练项目应该如何从数据走到部署。当前仓库已经准备好基础数据、模型代码、训练脚本、推理/API、轻量评测和统一 runner。

## 0. 环境

安装通用依赖：

```powershell
python -m pip install -r requirements.txt
```

PyTorch 按本机 CUDA 版本单独安装。确认：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

静态自检：

```powershell
python scripts\quick_check.py
```

## 1. 数据准备

目标：生成统一的无人机领域训练数据文件。

输入：

- `data/sft_drone_sample.jsonl`
- `data/sft_drone.jsonl`
- `data/sft_drone_batch2.jsonl`
- `data/sft_drone_batch3.jsonl`
- `data/agent_drone_sample.jsonl`

输出：

- `data/sft_merged.jsonl`
- `data/rlaif_drone_sample.jsonl`
- `data/dpo_drone_sample.jsonl`
- `data/agent_drone.jsonl`

命令：

```powershell
python scripts\prepare_drone_data.py
```

或：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\dronemind.ps1 prepare
```

## 2. 预训练

目标：让小模型获得基础语言能力。

输入：

- `data/pretrain_t2t_mini.jsonl`

输出：

- `out/pretrain_768.pth`

命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\dronemind.ps1 pretrain -Device cuda:0
```

## 3. 全量 SFT

目标：让模型学会无人机领域问答、聊天格式和基础工具调用格式。

输入：

- `out/pretrain_768.pth`
- `data/sft_merged.jsonl`

输出：

- `out/full_sft_768.pth`

命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\dronemind.ps1 sft -Device cuda:0
```

## 4. LoRA 可选微调

目标：针对一个更窄的无人机场景做低成本适配，例如航测、植保、巡检、法规问答。

输入：

- `out/full_sft_768.pth`
- `data/sft_merged.jsonl` 或自定义场景 SFT 数据

输出：

- `out/lora_drone_768.pth`

命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\dronemind.ps1 lora -Device cuda:0
```

## 5. DPO 可选偏好对齐

目标：让模型偏向更安全、更专业、更保守的无人机答案，远离危险建议。

输入：

- `out/full_sft_768.pth`
- `data/dpo_drone_sample.jsonl`

输出：

- `out/dpo_768.pth`

命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\dronemind.ps1 dpo -Device cuda:0
```

当前 DPO 数据是小样例，适合验证链路。正式训练应扩充到数百到数千条偏好对。

## 6. GRPO / Agent RL 可选阶段

目标：训练多轮工具调用和自主决策能力。

现有脚本：

- `trainer/train_grpo.py`
- `trainer/train_agent.py`

现有数据：

- `data/rlaif_drone_sample.jsonl`
- `data/agent_drone.jsonl`

注意：

- 这两个阶段依赖 reward model 或规则奖励，显存和时间成本更高。
- 当前仓库提供脚本和样例数据，但没有已训练的 GRPO/Agent 权重。
- 如果把它作为最终交付能力，需要补充大规模 Agent 数据、训练日志和 `out/grpo_768.pth` 或 `out/agent_768.pth`。

## 7. 评测

快速领域评测：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\dronemind.ps1 eval -Device cuda:0
```

或：

```powershell
python scripts\eval_drone_smoke.py --device cuda:0
```

评测集：

- `data/eval_drone_smoke.jsonl`

输出指标：

- 关键词覆盖率
- 每条样例的命中关键词
- 生成文本样例

这个评测是 smoke test，不代表正式 benchmark。正式交付建议增加：

- 飞行安全问答集
- 法规合规问答集
- 航测参数计算集
- 设备维护诊断集
- 工具调用准确率
- 拒答/安全边界测试

## 8. 部署

启动 API：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\dronemind.ps1 serve -Device cuda:0
```

客户端：

```powershell
python scripts\chat_api.py --base_url http://localhost:8998/v1
```

生产部署还需要：

- 鉴权
- 请求日志
- 限流
- 错误追踪
- GPU 资源监控
- 官方空域/天气/机型数据库工具接入

## 推荐交付标准

基础版：

- `pretrain -> sft -> eval -> api` 跑通
- 有 `out/full_sft_768.pth`
- `scripts/quick_check.py` 通过
- `scripts/eval_drone_smoke.py` 有结果

完整版：

- 基础版全部完成
- LoRA 或 DPO 至少一个对齐阶段跑通
- 有训练日志和评测记录
- Agent 工具调用至少在 20 条以上任务上验证
- README、训练路线、项目状态文档一致
