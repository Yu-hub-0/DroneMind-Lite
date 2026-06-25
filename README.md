# DroneMind-Lite

DroneMind-Lite 是一个面向无人机低空场景的小型语言模型项目，包含模型结构、训练脚本、示例数据、工具调用、评测脚本和本地 API 服务。项目目标是让使用者可以快速理解并运行一个完整的 UAV 领域小模型工程。

详细中文讲解见 [PROJECT_GUIDE.md](./PROJECT_GUIDE.md)。

## 项目内容

```text
model/                模型结构、LoRA 支持和 tokenizer 配置
dataset/              JSONL 数据加载和 loss mask 处理
trainer/              预训练、SFT、LoRA、DPO、PPO、GRPO、Agent 训练脚本
scripts/              数据准备、API 服务、Web Demo、评测和统一 runner
configs/              默认训练配置
data/                 小型 UAV 示例数据和评测数据
reports/              评测报告样例
eval_llm.py           本地命令行推理
run_pipeline.py       自动训练流水线
test_training.py      轻量训练 smoke test
```

## 快速开始

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

PyTorch 请根据本机 CUDA/CPU 环境单独安装。安装后可以检查：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

运行项目结构检查：

```powershell
python scripts\quick_check.py
```

准备 UAV 领域数据：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\dronemind.ps1 prepare
```

运行轻量训练验证：

```powershell
python test_training.py
```

## 推理与服务

如果本地已有训练权重，请将 `.pth` 文件放入 `out/`，例如：

```text
out/pretrain_768.pth
out/full_sft_768.pth
```

命令行推理：

```powershell
python eval_llm.py --load_from model --weight full_sft --device cuda
```

启动 OpenAI 兼容 API：

```powershell
python scripts\serve_openai_api.py --load_from model --weight full_sft --device cuda:0
```

默认接口：

```text
http://localhost:8998/v1/chat/completions
```

本地客户端：

```powershell
python scripts\chat_api.py --base_url http://localhost:8998/v1
```

## 数据与权重说明

GitHub 仓库只保留运行演示和轻量验证需要的小型数据。权重文件和大规模训练数据不进入仓库：

```text
out/*.pth
data/pretrain_t2t_mini.jsonl
data/sft_t2t_mini.jsonl
```

默认配置使用仓库内的小型数据：

```text
data/pretrain_merged.jsonl
data/sft_merged.jsonl
```

## 安全边界

DroneMind-Lite 仅用于学习、实验和原型验证。真实飞行前必须以官方空域、天气、法规、设备固件、电池健康和任务审批信息为准。不要把模型输出作为飞行安全、法规合规或商业运行许可的唯一依据。

## License

Apache License 2.0. See [LICENSE](./LICENSE).
