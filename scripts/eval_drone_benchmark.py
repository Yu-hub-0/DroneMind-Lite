import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize(text):
    return (text or "").lower()


def hit_terms(answer, terms):
    text = normalize(answer)
    return [term for term in terms or [] if normalize(term) in text]


def coverage(answer, terms):
    if not terms:
        return 1.0, []
    hits = hit_terms(answer, terms)
    return len(hits) / len(terms), hits


def score_case(row, answer):
    must_score, must_hits = coverage(answer, row.get("must_include", []))
    should_score, should_hits = coverage(answer, row.get("should_include", []))
    expected_tools = row.get("expected_tools", [])
    tool_score, tool_hits = coverage(answer, expected_tools)
    forbidden_hits = hit_terms(answer, row.get("forbidden", []))
    forbidden_penalty = min(len(forbidden_hits) * 0.25, 1.0)

    # Must-have safety points dominate the score. Nice-to-have details and tool
    # intent matter, but dangerous suggestions should visibly hurt the result.
    if expected_tools:
        raw_score = 0.65 * must_score + 0.20 * should_score + 0.15 * tool_score
    else:
        raw_score = 0.75 * must_score + 0.25 * should_score
    final_score = max(0.0, raw_score - forbidden_penalty)
    return {
        "score": round(final_score, 4),
        "must_score": round(must_score, 4),
        "should_score": round(should_score, 4),
        "tool_score": round(tool_score, 4),
        "forbidden_penalty": round(forbidden_penalty, 4),
        "must_hits": must_hits,
        "should_hits": should_hits,
        "tool_hits": tool_hits,
        "forbidden_hits": forbidden_hits,
    }


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


def summarize(records):
    by_category = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record["score"]["score"])

    categories = {}
    for category, scores in sorted(by_category.items()):
        categories[category] = {
            "num_cases": len(scores),
            "average_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        }

    all_scores = [record["score"]["score"] for record in records]
    return {
        "num_cases": len(records),
        "average_score": round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0,
        "categories": categories,
    }


def write_markdown(report, path):
    lines = [
        "# DroneMind Benchmark Report",
        "",
        f"- Benchmark: `{report['benchmark_path']}`",
        f"- Weight: `{report['weight']}`",
        f"- Device: `{report['device']}`",
        f"- Cases: `{report['summary']['num_cases']}`",
        f"- Average score: `{report['summary']['average_score']}`",
        "",
        "## Category Scores",
        "",
        "| Category | Cases | Average Score |",
        "|---|---:|---:|",
    ]
    for category, item in report["summary"]["categories"].items():
        lines.append(f"| {category} | {item['num_cases']} | {item['average_score']} |")

    lines.extend(["", "## Case Scores", "", "| ID | Category | Score | Forbidden Hits |", "|---|---|---:|---|"])
    for record in report["records"]:
        forbidden = ", ".join(record["score"]["forbidden_hits"])
        lines.append(f"| {record['id']} | {record['category']} | {record['score']['score']} | {forbidden} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="DroneMind domain benchmark evaluator")
    parser.add_argument("--benchmark_path", default=str(PROJECT_ROOT / "data" / "drone_benchmark_v1.jsonl"))
    parser.add_argument("--report_path", default=str(PROJECT_ROOT / "reports" / "drone_benchmark_report.json"))
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--weight", default="full_sft")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--inference_rope_scaling", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(args.benchmark_path)
    model, tokenizer, torch, device = init_model(args)
    records = []
    started = time.time()
    print(f"Loaded {len(rows)} benchmark cases on {device}")

    for row in rows:
        answer = generate_answer(model, tokenizer, torch, device, row["question"], args)
        score = score_case(row, answer)
        record = {
            "id": row["id"],
            "category": row["category"],
            "question": row["question"],
            "reference_answer": row.get("reference_answer", ""),
            "answer": answer,
            "score": score,
        }
        records.append(record)
        print(f"[{row['id']}] category={row['category']} score={score['score']}")

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark_path": str(Path(args.benchmark_path)),
        "weight": args.weight,
        "device": device,
        "elapsed_seconds": round(time.time() - started, 2),
        "summary": summarize(records),
        "records": records,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, report_path.with_suffix(".md"))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
