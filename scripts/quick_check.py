import argparse
import importlib.util
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "model/tokenizer.json",
    "model/tokenizer_config.json",
    "model/model_dronemind.py",
    "dataset/lm_dataset.py",
    "trainer/train_pretrain.py",
    "trainer/train_full_sft.py",
    "eval_llm.py",
    "scripts/serve_openai_api.py",
    "scripts/prepare_drone_data.py",
    "scripts/run_dronemind.py",
    "scripts/drone_tools.py",
    "scripts/eval_drone_tooluse.py",
    "scripts/eval_toolcall.py",
    "configs/dronemind_64m.json",
    "README.md",
    "PROJECT_GUIDE.md",
    "TRAINING_PIPELINE.md",
    "MODEL_CARD_DRONEMIND.md",
    "DELIVERY_CHECKLIST.md",
    "ALIGNMENT_COMPLETION.md",
]

WEIGHT_FILES = [
    "out/pretrain_768.pth",
    "out/full_sft_768.pth",
]

OPTIONAL_ALIGNMENT_WEIGHTS = [
    "out/dpo_768.pth",
    "out/grpo_768.pth",
    "out/agent_768.pth",
]

DATA_FILES = [
    "data/pretrain_drone_sample.jsonl",
    "data/sft_drone_sample.jsonl",
    "data/sft_merged.jsonl",
    "data/dpo_drone_sample.jsonl",
    "data/rlaif_drone_sample.jsonl",
    "data/agent_drone.jsonl",
    "data/eval_drone_smoke.jsonl",
]

PYTHON_PACKAGES = [
    "torch",
    "transformers",
    "datasets",
    "fastapi",
    "uvicorn",
    "openai",
    "streamlit",
]


def ok(label):
    print(f"[OK] {label}")


def warn(label):
    print(f"[WARN] {label}")


def fail(label):
    print(f"[FAIL] {label}")


def check_files(paths):
    missing = []
    for rel_path in paths:
        path = PROJECT_ROOT / rel_path
        if path.exists():
            ok(f"{rel_path}")
        else:
            missing.append(rel_path)
            fail(f"{rel_path} missing")
    return missing


def check_jsonl(rel_path, limit=20):
    path = PROJECT_ROOT / rel_path
    if not path.exists():
        fail(f"{rel_path} missing")
        return 0, 1

    total = 0
    errors = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            if limit and total >= limit:
                break
            total += 1
            if limit:
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    errors += 1
                    fail(f"{rel_path}:{line_no} invalid JSON: {exc}")
            else:
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    errors += 1
                    fail(f"{rel_path}:{line_no} invalid JSON: {exc}")

    if errors == 0:
        ok(f"{rel_path} JSONL sample valid ({total if limit else 'all'} checked)")
    return total, errors


def count_jsonl(rel_path):
    path = PROJECT_ROOT / rel_path
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def check_imports():
    missing = []
    for package in PYTHON_PACKAGES:
        if importlib.util.find_spec(package) is None:
            missing.append(package)
            warn(f"python package not installed: {package}")
        else:
            ok(f"python package available: {package}")
    return missing


def main():
    parser = argparse.ArgumentParser(description="DroneMind project readiness check")
    parser.add_argument("--full-jsonl", action="store_true", help="validate every JSONL line instead of a small sample")
    args = parser.parse_args()

    print(f"Project root: {PROJECT_ROOT}\n")

    print("== Required files ==")
    missing_files = check_files(REQUIRED_FILES)

    print("\n== Weights ==")
    missing_weights = check_files(WEIGHT_FILES)
    print("\n== Optional alignment weights ==")
    for rel_path in OPTIONAL_ALIGNMENT_WEIGHTS:
        path = PROJECT_ROOT / rel_path
        if path.exists():
            ok(f"{rel_path}")
        else:
            warn(f"{rel_path} not found; run the corresponding dpo/grpo/agent stage if needed")

    print("\n== Data ==")
    data_errors = 0
    for rel_path in DATA_FILES:
        _, errors = check_jsonl(rel_path, limit=0 if args.full_jsonl else 20)
        data_errors += errors

    print("\n== Data counts ==")
    for rel_path in sorted(str(p.relative_to(PROJECT_ROOT)).replace("\\", "/") for p in (PROJECT_ROOT / "data").glob("*.jsonl")):
        print(f"{rel_path}: {count_jsonl(rel_path)} records")

    print("\n== Python packages ==")
    missing_packages = check_imports()

    print("\n== Summary ==")
    if missing_files or data_errors:
        fail("project structure/data checks did not pass")
        raise SystemExit(1)
    if missing_weights:
        warn("some trained weights are missing")
    if missing_packages:
        warn("install dependencies before training/inference")
    ok("static project check complete")


if __name__ == "__main__":
    main()
