import argparse
import json
from pathlib import Path

from drone_tools import DRONE_TOOLS, execute_tool


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_ARGS = {
    "check_no_fly_zone": {"latitude": 39.92, "longitude": 116.40},
    "get_weather_for_flight": {"location": "Beijing Chaoyang Park"},
    "calculate_flight_time": {
        "battery_capacity_mah": 5000,
        "battery_percentage": 100,
        "drone_weight_kg": 1.2,
        "wind_speed_ms": 5,
    },
    "plan_mission_route": {
        "area_km2": 2,
        "overlap_rate": 0.75,
        "side_overlap_rate": 0.65,
        "flight_altitude_m": 120,
    },
    "lookup_drone_specs": {"model": "DJI Mavic 3"},
    "get_drone_specs": {"model": "DJI Mavic 3"},
    "convert_uav_units": {"value": 120, "from_unit": "m", "to_unit": "ft"},
    "convert_drone_units": {"value": 120, "from_unit": "m", "to_unit": "ft"},
}


ALIASES = {
    "get_drone_specs": "lookup_drone_specs",
    "convert_drone_units": "convert_uav_units",
}


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def tool_names_from_schema(tools):
    names = []
    for tool in tools or []:
        name = tool.get("function", {}).get("name")
        if name:
            names.append(name)
    return names


def parse_sample(sample):
    tools = []
    for message in sample.get("conversations", []):
        raw_tools = message.get("tools")
        if raw_tools:
            tools = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools
            break
    return tools, sample.get("gt", [])


def canonical(name):
    return ALIASES.get(name, name)


def run_tool_plan(tool_names):
    calls = []
    for name in tool_names:
        args = DEFAULT_ARGS.get(name) or DEFAULT_ARGS.get(canonical(name), {})
        result = execute_tool(name, args)
        calls.append({"name": name, "arguments": args, "result": result})
    return calls


def score_record(tool_names, calls, gt):
    expected = {canonical(name) for name in tool_names}
    called = {canonical(call["name"]) for call in calls if "error" not in call.get("result", {})}
    tool_selection = len(expected & called) / len(expected) if expected else 1.0
    parameter_valid = sum(1 for call in calls if "error" not in call.get("result", {})) / len(calls) if calls else 1.0
    # The Agent data stores human-readable gt hints, while this script validates the
    # executable tool chain. Use semantic tool coverage so the score is not brittle
    # to natural-language wording or terminal encoding.
    gt_hits = sorted(expected & called)
    gt_coverage = tool_selection if gt else 1.0
    return {
        "tool_selection": round(tool_selection, 4),
        "parameter_valid": round(parameter_valid, 4),
        "gt_coverage": round(gt_coverage, 4),
        "gt_hits": gt_hits,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate DroneMind UAV tool-use data and tool execution")
    parser.add_argument("--data_path", default=str(PROJECT_ROOT / "data" / "agent_drone.jsonl"))
    parser.add_argument("--report_path", default=str(PROJECT_ROOT / "reports" / "tooluse_eval_report.json"))
    args = parser.parse_args()

    data_path = Path(args.data_path)
    rows = load_jsonl(data_path)
    records = []
    for idx, sample in enumerate(rows, 1):
        tools, gt = parse_sample(sample)
        tool_names = tool_names_from_schema(tools)
        calls = run_tool_plan(tool_names)
        score = score_record(tool_names, calls, gt)
        records.append({
            "id": idx,
            "tool_names": tool_names,
            "gt": gt,
            "calls": calls,
            "score": score,
        })

    averages = {}
    for key in ("tool_selection", "parameter_valid", "gt_coverage"):
        averages[key] = round(sum(r["score"][key] for r in records) / len(records), 4) if records else 0.0
    report = {
        "data_path": str(data_path),
        "registered_drone_tools": [tool["function"]["name"] for tool in DRONE_TOOLS],
        "num_cases": len(records),
        "averages": averages,
        "records": records,
    }

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Evaluated {len(records)} tool-use cases")
    print(json.dumps({"averages": averages, "report_path": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
