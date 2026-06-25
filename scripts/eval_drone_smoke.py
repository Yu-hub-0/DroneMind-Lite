import argparse
import json
import os
import random
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_eval_set(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def keyword_score(answer, keywords):
    if not keywords:
        return 0.0, []
    hits = [kw for kw in keywords if kw.lower() in answer.lower()]
    return len(hits) / len(keywords), hits


def init_model(args):
    try:
        import torch
        from transformers import AutoTokenizer
        from model.model_dronemind import DroneMindConfig, DroneMindForCausalLM
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}")
        print("Install dependencies first, and install PyTorch for your CUDA/CPU runtime.")
        raise SystemExit(1) from exc

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(PROJECT_ROOT / "model")
    config = DroneMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        inference_rope_scaling=args.inference_rope_scaling,
    )
    model = DroneMindForCausalLM(config)
    suffix = "_moe" if args.use_moe else ""
    weight_path = PROJECT_ROOT / args.save_dir / f"{args.weight}_{args.hidden_size}{suffix}.pth"
    if not weight_path.exists():
        raise FileNotFoundError(f"weight not found: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=True)
    model = model.half().eval().to(device) if "cuda" in device else model.float().eval().to(device)
    return model, tokenizer, torch, device


def generate_answer(model, tokenizer, torch, device, question, args):
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-5),
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Small DroneMind UAV-domain smoke evaluation")
    parser.add_argument("--eval_path", default=str(PROJECT_ROOT / "data" / "eval_drone_smoke.jsonl"))
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--weight", default="full_sft")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--inference_rope_scaling", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_eval_set(args.eval_path)
    model, tokenizer, torch, device = init_model(args)
    print(f"Loaded {len(rows)} cases on {device}")

    scores = []
    started = time.time()
    for row in rows:
        answer = generate_answer(model, tokenizer, torch, device, row["question"], args)
        score, hits = keyword_score(answer, row.get("keywords", []))
        scores.append(score)
        print("\n" + "=" * 80)
        print(f"[{row['id']}] {row['question']}")
        print(f"score={score:.2f} hits={hits}")
        print(answer[:800])

    avg = sum(scores) / len(scores) if scores else 0.0
    print("\n" + "=" * 80)
    print(f"Average keyword coverage: {avg:.2f}")
    print(f"Elapsed: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
