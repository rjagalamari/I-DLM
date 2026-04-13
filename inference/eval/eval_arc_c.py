"""
ARC-Challenge evaluation across multiple sglang servers.

Uses allenai/ai2_arc ARC-Challenge split, 4-choice (A/B/C/D).

Usage:
  python scripts/eval_arc_c.py                          # all ~1172 problems, 8 GPUs
  python scripts/eval_arc_c.py --num-problems 200       # quick
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset


def strip_thinking(text):
    if not text:
        return ""
    return re.sub(r"^.*?</think>\s*", "", text, count=1, flags=re.DOTALL)


def extract_choice(text):
    """Extract single letter choice (A-D) from model output."""
    if "</think>" in text:
        after = re.sub(r"^.*?</think>\s*", "", text, count=1, flags=re.DOTALL)
    else:
        after = text[-500:]
    m = re.search(r"[Aa]nswer is:?\s*\*{0,2}\(?([A-Da-d])\)?\*{0,2}", after)
    if m:
        return m.group(1).upper()
    m = re.search(r"\\boxed\{[^}]*?([A-Da-d])[^A-Da-d}]*\}", after)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?:\*\*\(?([A-Da-d])\)?\*\*|\(([A-Da-d])\))\s*[.\s]*$", after.strip())
    if m:
        return (m.group(1) or m.group(2)).upper()
    return "?"


def normalize_label(label):
    """Convert numeric labels (1,2,3,4) to letters (A,B,C,D)."""
    num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    return num_to_letter.get(label, label)


def format_choices(choices):
    labels = choices["label"]
    texts = choices["text"]
    lines = []
    for l, t in zip(labels, texts):
        lines.append(f"{normalize_label(l)}. {t}")
    return "\n".join(lines)


def run_one(args):
    idx, question, choices_str, gold, port, max_tokens, timeout, temperature, top_p, top_k = args
    prompt = (
        f"{question}\n\n{choices_str}\n\n"
        "Think step by step, then give your answer as \"The answer is (X)\"."
    )
    try:
        r = requests.post(f"http://localhost:{port}/v1/chat/completions", json={
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }, timeout=timeout).json()
        content = r["choices"][0]["message"]["content"]
        finish = r["choices"][0].get("finish_reason", "")
        comp = r["usage"]["completion_tokens"]
        pred = extract_choice(content)
        return idx, pred, gold, comp, finish, content, None
    except Exception as e:
        return idx, "?", gold, 0, "", "", str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=0, help="0 = all")
    parser.add_argument("--ports", type=int, nargs="+", default=[30000 + i for i in range(8)])
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-workers", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    if args.num_problems > 0 and args.num_problems < len(ds):
        import random
        random.seed(42)
        indices = random.sample(range(len(ds)), args.num_problems)
        ds = ds.select(indices)
    N = len(ds)
    ports = args.ports

    problems = []
    for item in ds:
        q = item["question"]
        choices_str = format_choices(item["choices"])
        gold = normalize_label(item["answerKey"])
        problems.append((q, choices_str, gold))

    print(f"ARC-Challenge eval: {N} problems, {len(ports)} servers")

    tasks = [
        (i, q, c, g, ports[i % len(ports)], args.max_tokens, args.timeout,
         args.temperature, args.top_p, args.top_k)
        for i, (q, c, g) in enumerate(problems)
    ]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(run_one, t) for t in tasks]
        results = [f.result() for f in futures]

    elapsed = time.time() - t0
    correct = total_tok = errors = truncated = no_extract = wrong = 0
    wrong_examples = []
    for idx, pred, gold, comp, finish, content, err in results:
        if err:
            errors += 1
            continue
        total_tok += comp
        if pred == gold:
            correct += 1
        else:
            if finish == "length":
                truncated += 1
            elif pred == "?":
                no_extract += 1
            else:
                wrong += 1
            if len(wrong_examples) < 10:
                wrong_examples.append({
                    "idx": idx, "pred": pred, "gold": gold,
                    "finish": finish, "output": content[:200]
                })

    acc = correct / N * 100
    print(f"\n{'=' * 60}")
    print(f"ARC-Challenge {N} problems, {len(ports)} GPUs")
    print(f"{'=' * 60}")
    print(f"Accuracy:       {correct}/{N} ({acc:.1f}%)")
    print(f"Truncated:      {truncated}")
    print(f"No extraction:  {no_extract}")
    print(f"Wrong answer:   {wrong}")
    print(f"Total tokens:   {total_tok:,}")
    print(f"Wall time:      {elapsed:.1f}s")
    print(f"Throughput:     {total_tok / elapsed:.1f} tok/s")
    if errors:
        print(f"Errors:         {errors}")

    if wrong_examples:
        print(f"\n--- Sample wrong answers ---")
        for ex in wrong_examples[:5]:
            print(f"  [{ex['idx']}] pred={ex['pred']} gold={ex['gold']} finish={ex['finish']}")
            print(f"    {ex['output'][:150]}...")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "arc_c_summary.json"), "w") as f:
            json.dump({"accuracy": acc, "correct": correct, "total": N,
                       "tokens": total_tok, "truncated": truncated,
                       "no_extract": no_extract, "wrong": wrong}, f, indent=2)


if __name__ == "__main__":
    main()
