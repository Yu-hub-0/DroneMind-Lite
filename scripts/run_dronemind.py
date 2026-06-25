import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/dronemind_64m.json"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_command(cmd, cwd=PROJECT_ROOT, dry_run=False):
    printable = " ".join(str(x) for x in cmd)
    print(f"\n$ {printable}")
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(cwd), check=False).returncode


def model_args(config):
    model = config["model"]
    return [
        "--hidden_size", str(model["hidden_size"]),
        "--num_hidden_layers", str(model["num_hidden_layers"]),
        "--use_moe", "1" if model.get("use_moe") else "0",
    ]


def stage_prepare(args, config):
    paths = config["paths"]
    return run_command([
        sys.executable,
        str(PROJECT_ROOT / "scripts/prepare_drone_data.py"),
        "--sft_out", str(PROJECT_ROOT / paths["sft_merged"]),
        "--rlaif_out", str(PROJECT_ROOT / paths["rlaif_data"]),
        "--dpo_out", str(PROJECT_ROOT / paths["dpo_data"]),
        "--agent_out", str(PROJECT_ROOT / paths["agent_data"]),
    ], dry_run=args.dry_run)


def stage_pretrain(args, config):
    paths = config["paths"]
    train = config["training"]["pretrain"]
    model = config["model"]
    cmd = [
        sys.executable, "train_pretrain.py",
        "--data_path", str(PROJECT_ROOT / paths["pretrain_data"]),
        "--epochs", str(train["epochs"]),
        "--batch_size", str(train["batch_size"]),
        "--accumulation_steps", str(train["accumulation_steps"]),
        "--learning_rate", str(train["learning_rate"]),
        "--max_seq_len", str(model["max_seq_len_pretrain"]),
        "--save_dir", str(PROJECT_ROOT / paths["out"]),
        "--from_weight", "none",
        "--device", args.device,
    ] + model_args(config)
    return run_command(cmd, cwd=PROJECT_ROOT / "trainer", dry_run=args.dry_run)


def stage_sft(args, config):
    paths = config["paths"]
    train = config["training"]["sft"]
    model = config["model"]
    cmd = [
        sys.executable, "train_full_sft.py",
        "--data_path", str(PROJECT_ROOT / paths["sft_merged"]),
        "--epochs", str(train["epochs"]),
        "--batch_size", str(train["batch_size"]),
        "--accumulation_steps", str(train["accumulation_steps"]),
        "--learning_rate", str(train["learning_rate"]),
        "--max_seq_len", str(model["max_seq_len_sft"]),
        "--save_dir", str(PROJECT_ROOT / paths["out"]),
        "--from_weight", "pretrain",
        "--device", args.device,
    ] + model_args(config)
    return run_command(cmd, cwd=PROJECT_ROOT / "trainer", dry_run=args.dry_run)


def stage_lora(args, config):
    paths = config["paths"]
    train = config["training"]["lora"]
    model = config["model"]
    cmd = [
        sys.executable, "train_lora.py",
        "--data_path", str(PROJECT_ROOT / paths["sft_merged"]),
        "--epochs", str(train["epochs"]),
        "--batch_size", str(train["batch_size"]),
        "--learning_rate", str(train["learning_rate"]),
        "--lora_name", train["lora_name"],
        "--max_seq_len", str(model["max_seq_len_sft"]),
        "--save_dir", str(PROJECT_ROOT / paths["out"]),
        "--from_weight", "full_sft",
        "--device", args.device,
    ] + model_args(config)
    return run_command(cmd, cwd=PROJECT_ROOT / "trainer", dry_run=args.dry_run)


def stage_dpo(args, config):
    paths = config["paths"]
    train = config["training"]["dpo"]
    model = config["model"]
    cmd = [
        sys.executable, "train_dpo.py",
        "--data_path", str(PROJECT_ROOT / paths["dpo_data"]),
        "--epochs", str(train["epochs"]),
        "--batch_size", str(train["batch_size"]),
        "--learning_rate", str(train["learning_rate"]),
        "--beta", str(train["beta"]),
        "--max_seq_len", str(model["max_seq_len_rl"]),
        "--save_dir", str(PROJECT_ROOT / paths["out"]),
        "--from_weight", "full_sft",
        "--device", args.device,
    ] + model_args(config)
    return run_command(cmd, cwd=PROJECT_ROOT / "trainer", dry_run=args.dry_run)


def stage_grpo(args, config):
    paths = config["paths"]
    train = config["training"]["grpo"]
    model = config["model"]
    cmd = [
        sys.executable, "train_grpo.py",
        "--data_path", str(PROJECT_ROOT / paths["rlaif_data"]),
        "--epochs", str(train["epochs"]),
        "--batch_size", str(train["batch_size"]),
        "--learning_rate", str(train["learning_rate"]),
        "--beta", str(train["beta"]),
        "--num_generations", str(train["num_generations"]),
        "--max_gen_len", str(train["max_gen_len"]),
        "--max_steps", str(train.get("max_steps", 0)),
        "--num_workers", str(train.get("num_workers", 0)),
        "--max_seq_len", str(model["max_seq_len_rl"]),
        "--reward_model_path", str(train["reward_model_path"]),
        "--save_dir", str(PROJECT_ROOT / paths["out"]),
        "--from_weight", "full_sft",
        "--device", args.device,
    ] + model_args(config)
    return run_command(cmd, cwd=PROJECT_ROOT / "trainer", dry_run=args.dry_run)


def stage_agent(args, config):
    paths = config["paths"]
    train = config["training"]["agent"]
    model = config["model"]
    cmd = [
        sys.executable, "train_agent.py",
        "--data_path", str(PROJECT_ROOT / paths["agent_data"]),
        "--epochs", str(train["epochs"]),
        "--batch_size", str(train["batch_size"]),
        "--learning_rate", str(train["learning_rate"]),
        "--beta", str(train["beta"]),
        "--num_generations", str(train["num_generations"]),
        "--max_gen_len", str(train["max_gen_len"]),
        "--max_steps", str(train.get("max_steps", 0)),
        "--num_workers", str(train.get("num_workers", 0)),
        "--max_seq_len", str(model["max_seq_len_rl"]),
        "--reward_model_path", str(train["reward_model_path"]),
        "--save_dir", str(PROJECT_ROOT / paths["out"]),
        "--from_weight", "full_sft",
        "--device", args.device,
    ] + model_args(config)
    return run_command(cmd, cwd=PROJECT_ROOT / "trainer", dry_run=args.dry_run)


def stage_eval(args, config):
    paths = config["paths"]
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/eval_drone_smoke.py"),
        "--eval_path", str(PROJECT_ROOT / paths["eval_data"]),
        "--save_dir", paths["out"],
        "--weight", args.weight,
        "--device", args.device,
    ] + model_args(config)
    return run_command(cmd, dry_run=args.dry_run)


def stage_tool_eval(args, config):
    paths = config["paths"]
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/eval_drone_tooluse.py"),
        "--data_path", str(PROJECT_ROOT / paths["agent_data"]),
        "--report_path", str(PROJECT_ROOT / paths["tooluse_report"]),
    ]
    return run_command(cmd, dry_run=args.dry_run)


def stage_benchmark(args, config):
    paths = config["paths"]
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/eval_drone_benchmark.py"),
        "--benchmark_path", str(PROJECT_ROOT / paths["benchmark_data"]),
        "--report_path", str(PROJECT_ROOT / paths["benchmark_report"]),
        "--save_dir", paths["out"],
        "--weight", args.weight,
        "--device", args.device,
    ] + model_args(config)
    return run_command(cmd, dry_run=args.dry_run)


def stage_serve(args, config):
    paths = config["paths"]
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/serve_openai_api.py"),
        "--load_from", paths["tokenizer"],
        "--save_dir", paths["out"],
        "--weight", args.weight,
        "--device", args.device,
    ] + model_args(config)
    return run_command(cmd, dry_run=args.dry_run)


STAGES = {
    "prepare": stage_prepare,
    "pretrain": stage_pretrain,
    "sft": stage_sft,
    "lora": stage_lora,
    "dpo": stage_dpo,
    "grpo": stage_grpo,
    "agent": stage_agent,
    "eval": stage_eval,
    "benchmark": stage_benchmark,
    "tool_eval": stage_tool_eval,
    "serve": stage_serve,
}


def main():
    parser = argparse.ArgumentParser(description="DroneMind end-to-end project runner")
    parser.add_argument("stage", choices=list(STAGES) + ["all"], help="Pipeline stage to run")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weight", default="full_sft")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    sequence = ["prepare", "pretrain", "sft", "eval", "benchmark", "tool_eval"] if args.stage == "all" else [args.stage]
    for stage in sequence:
        code = STAGES[stage](args, config)
        if code != 0:
            print(f"Stage failed: {stage} (exit={code})")
            raise SystemExit(code)


if __name__ == "__main__":
    main()
