# DroneMind Benchmark

本项目新增了一个无人机领域 benchmark 第一版，用于回答“这个大模型到底要评估什么、输入是什么、输出是什么、怎么判断好坏”。

## 目标

DroneMind 是面向无人机领域的问答与任务辅助模型。Benchmark 评估模型是否能在无人机业务场景下输出更安全、更专业、更可控的回答。

## 输入

输入是自然语言问题，例如：

```text
无人机飞行中 GPS 信号突然丢失，操作者应该怎么处理？
```

也可以是任务型请求，例如：

```text
我有一块 5000mAh 电池，当前电量 80%，无人机重 1.2kg，风速 5m/s，请估算可用飞行时间。
```

## 输出

输出是模型生成的无人机领域回答，包括：

- 飞行安全建议
- 故障处理步骤
- 维护检查建议
- 航测任务规划建议
- 合规与风险提醒
- 工具调用型任务的计算或判断说明

## 评估维度

当前 benchmark 覆盖以下维度：

- `flight_safety`：飞行安全，例如 GPS 丢失、低电量、强风飞行、起飞前检查
- `maintenance`：设备维护，例如电池健康、异响排查
- `mapping`：航测任务，例如重叠率、航线规划
- `compliance`：合规风险，例如机场附近、密集人群
- `emergency`：应急处置，例如失联、坠机
- `tool_use`：工具使用意图，例如天气、禁飞区、飞行时间估算
- `safety_boundary`：安全拒答，例如绕过禁飞限制

## 评分方法

每道题包含：

- `must_include`：必须覆盖的关键点
- `should_include`：建议覆盖的关键点
- `forbidden`：不应该出现的危险或错误表达
- `expected_tools`：如果是工具型任务，期望使用或提及的工具能力
- `reference_answer`：参考答案

脚本会生成模型回答，并计算：

- 必要关键点覆盖率
- 建议关键点覆盖率
- 禁止项命中惩罚
- 工具意图覆盖率
- 单题综合分
- 分类平均分
- 总平均分

## 运行方式

```powershell
python scripts\eval_drone_benchmark.py --device cuda
```

CPU 也可以运行，但速度较慢：

```powershell
python scripts\eval_drone_benchmark.py --device cpu
```

默认读取：

```text
data/drone_benchmark_v1.jsonl
```

默认输出：

```text
reports/drone_benchmark_report.json
reports/drone_benchmark_report.md
```

## 重要说明

这是无人机领域 benchmark 的第一版，适合做项目展示、训练验收和不同权重之间的对比。

它比原来的 smoke test 更正式，但还不是行业级权威 benchmark。后续如果要更严肃地证明模型优劣，应继续扩充题量，并加入人工专家评分、事实核查、安全红线测试和与通用大模型的横向对比。
