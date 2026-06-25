import argparse
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


SFT_SOURCES = [
    "data/sft_drone_sample.jsonl",
    "data/sft_drone.jsonl",
    "data/sft_drone_batch2.jsonl",
    "data/sft_drone_batch3.jsonl",
]


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_sft(output):
    merged = []
    seen = set()
    for rel_path in SFT_SOURCES:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            continue
        for row in read_jsonl(path):
            key = json.dumps(row.get("conversations", []), ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
    write_jsonl(output, merged)
    return len(merged)


def build_rlaif_from_sft(sft_path, output, limit):
    rows = read_jsonl(sft_path)
    selected = rows[:limit]
    write_jsonl(output, selected)
    return len(selected)


def build_dpo(output):
    rows = [
        {
            "chosen": [
                {"role": "user", "content": "无人机起飞前需要做哪些检查？"},
                {"role": "assistant", "content": "起飞前应检查电池电量和电芯压差、桨叶是否破损并安装牢固、GPS/RTK 是否锁定、指南针和 IMU 状态、返航高度、禁飞区和周边障碍物、遥控器和图传信号，并先短暂悬停确认姿态稳定。"}
            ],
            "rejected": [
                {"role": "user", "content": "无人机起飞前需要做哪些检查？"},
                {"role": "assistant", "content": "直接起飞就可以，电量差不多就行，其他问题飞起来再看。"}
            ]
        },
        {
            "chosen": [
                {"role": "user", "content": "无人机 GPS 信号丢失后应该怎么办？"},
                {"role": "assistant", "content": "先保持冷静，确认飞机是否进入姿态模式，降低动作幅度并保持可视范围；必要时升高以改善卫星信号，手动控制返航或选择安全区域降落，避免依赖自动返航。"}
            ],
            "rejected": [
                {"role": "user", "content": "无人机 GPS 信号丢失后应该怎么办？"},
                {"role": "assistant", "content": "继续向前飞，等信号自己恢复；如果看不见就加速返航。"}
            ]
        },
        {
            "chosen": [
                {"role": "user", "content": "电池鼓包的无人机电池还能继续使用吗？"},
                {"role": "assistant", "content": "不建议继续使用。鼓包说明电池内部已经异常，存在发热、漏液甚至起火风险，应停止充放电，放入防火容器并按当地规范回收处理。"}
            ],
            "rejected": [
                {"role": "user", "content": "电池鼓包的无人机电池还能继续使用吗？"},
                {"role": "assistant", "content": "只要还能充电就可以继续飞，鼓包通常不影响安全。"}
            ]
        },
        {
            "chosen": [
                {"role": "user", "content": "航测任务为什么要设置航向和旁向重叠率？"},
                {"role": "assistant", "content": "重叠率保证同一区域被多张照片覆盖，便于空三匹配、三维重建和正射拼接。地形起伏、建筑密集或建模任务通常需要更高重叠率，例如航向 75%-85%、旁向 65%-80%。"}
            ],
            "rejected": [
                {"role": "user", "content": "航测任务为什么要设置航向和旁向重叠率？"},
                {"role": "assistant", "content": "重叠率越低越好，可以少拍照片，后期软件都会自动补齐。"}
            ]
        }
    ]
    write_jsonl(output, rows)
    return len(rows)


def build_agent_data(output):
    src = PROJECT_ROOT / "data/agent_drone_sample.jsonl"
    if not src.exists():
        write_jsonl(output, [])
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, output)
    return sum(1 for line in output.open("r", encoding="utf-8") if line.strip())


def main():
    parser = argparse.ArgumentParser(description="Prepare DroneMind domain training data")
    parser.add_argument("--sft_out", default=str(PROJECT_ROOT / "data/sft_merged.jsonl"))
    parser.add_argument("--rlaif_out", default=str(PROJECT_ROOT / "data/rlaif_drone_sample.jsonl"))
    parser.add_argument("--dpo_out", default=str(PROJECT_ROOT / "data/dpo_drone_sample.jsonl"))
    parser.add_argument("--agent_out", default=str(PROJECT_ROOT / "data/agent_drone.jsonl"))
    parser.add_argument("--rlaif_limit", type=int, default=256)
    args = parser.parse_args()

    sft_out = Path(args.sft_out)
    rlaif_out = Path(args.rlaif_out)
    dpo_out = Path(args.dpo_out)
    agent_out = Path(args.agent_out)

    sft_count = merge_sft(sft_out)
    rlaif_count = build_rlaif_from_sft(sft_out, rlaif_out, args.rlaif_limit)
    dpo_count = build_dpo(dpo_out)
    agent_count = build_agent_data(agent_out)

    print(f"SFT merged: {sft_count} -> {sft_out}")
    print(f"RLAIF sample: {rlaif_count} -> {rlaif_out}")
    print(f"DPO sample: {dpo_count} -> {dpo_out}")
    print(f"Agent sample: {agent_count} -> {agent_out}")


if __name__ == "__main__":
    main()
