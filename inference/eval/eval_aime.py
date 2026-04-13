"""
AIME 2024 / 2025 evaluation across multiple sglang servers.

Answers are integers 0-999. Extracts from \\boxed{} then falls back to last integer.

Usage:
  python scripts/eval_aime.py                          # AIME 2024, all 30 problems, 8 GPUs
  python scripts/eval_aime.py --year 2025              # AIME 2025
  python scripts/eval_aime.py --ports 30000            # single GPU
  python scripts/eval_aime.py --num-problems 5         # quick test
  python scripts/eval_aime.py --num-samples 4          # majority vote (pass@1 via sampling)
  python scripts/eval_aime.py --output-dir /tmp/aime   # save details
"""
import argparse
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset

SYSTEM_PROMPT = "You are a mathematics expert. Please reason step by step, and put your final answer within \\boxed{}."


def strip_thinking(text):
    if not text:
        return ""
    return re.sub(r"^.*?</think>\s*", "", text, count=1, flags=re.DOTALL)


def extract_answer(text):
    """Extract integer answer from model output.
    1. Find last \\boxed{...} and parse as integer
    2. Fall back to last integer in the text
    """
    text = strip_thinking(text)
    # Find all \boxed{...} occurrences
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        raw = boxed[-1].strip()
        # Remove commas, spaces, leading zeros edge cases
        raw = raw.replace(",", "").replace(" ", "")
        try:
            return int(raw)
        except ValueError:
            # Try to extract integer from inside boxed
            m = re.search(r"-?\d+", raw)
            if m:
                return int(m.group())
    # Fall back: last integer in the response
    integers = re.findall(r"\b(\d{1,3})\b", text)
    if integers:
        return int(integers[-1])
    return None


def run_one(args):
    idx, problem, answer, port, max_tokens, timeout, sample_idx, temperature, top_p, top_k = args
    payload = {
        "model": "default",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }
    try:
        t0 = time.time()
        resp = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        ).json()
        elapsed = time.time() - t0
        raw = resp["choices"][0]["message"]["content"]
        pred = extract_answer(raw)
        correct = pred == answer
        tokens = resp.get("usage", {}).get("completion_tokens", 0)
        return {
            "idx": idx,
            "sample_idx": sample_idx,
            "problem": problem,
            "answer": answer,
            "pred": pred,
            "correct": correct,
            "raw": raw,
            "tokens": tokens,
            "elapsed": elapsed,
        }
    except Exception as e:
        return {
            "idx": idx,
            "sample_idx": sample_idx,
            "problem": problem,
            "answer": answer,
            "pred": None,
            "correct": False,
            "raw": str(e),
            "tokens": 0,
            "elapsed": 0,
            "error": True,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024, choices=[2024, 2025])
    parser.add_argument("--ports", type=int, nargs="+", default=[30000 + i for i in range(8)])
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--num-problems", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=1, help="samples per problem for majority vote")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    # Load dataset
    if args.year == 2024:
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        problems = [(row["problem"], int(row["answer"])) for row in ds]
    else:
        ds = load_dataset("MathArena/aime_2025", split="train")
        problems = [(row["problem"], int(row["answer"])) for row in ds]

    N = min(args.num_problems, len(problems)) if args.num_problems else len(problems)
    problems = problems[:N]
    S = args.num_samples

    print(f"AIME {args.year}: {N} problems x {S} samples, {len(args.ports)} servers, max_tokens={args.max_tokens}")

    # Build tasks: (idx, problem, answer, port, max_tokens, timeout, sample_idx, temperature, top_p, top_k)
    tasks = [
        (i, prob, ans, args.ports[(i * S + s) % len(args.ports)], args.max_tokens, args.timeout, s, args.temperature, args.top_p, args.top_k)
        for i, (prob, ans) in enumerate(problems)
        for s in range(S)
    ]

    print(f"Sampling: temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    t0 = time.time()
    results = []
    errors = 0
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {pool.submit(run_one, t): t for t in tasks}
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            if r.get("error"):
                errors += 1
            status = "✓" if r["correct"] else "✗"
            print(f"  [{status}] Problem {r['idx']+1} sample {r['sample_idx']}: pred={r['pred']} ans={r['answer']} ({r['tokens']} tok)")

    elapsed = time.time() - t0
    total_tokens = sum(r["tokens"] for r in results)

    # Aggregate by problem (majority vote if num_samples > 1)
    per_problem = {}
    for r in results:
        per_problem.setdefault(r["idx"], []).append(r)

    correct_count = 0
    details = []
    wrong = []
    for i in range(N):
        samples = per_problem.get(i, [])
        preds = [r["pred"] for r in samples if r["pred"] is not None]
        if preds:
            # Majority vote
            voted = Counter(preds).most_common(1)[0][0]
        else:
            voted = None
        ans = problems[i][1]
        correct = voted == ans
        if correct:
            correct_count += 1
        entry = {
            "idx": i,
            "problem": problems[i][0][:120] + "...",
            "answer": ans,
            "pred": voted,
            "all_preds": preds,
            "correct": correct,
        }
        details.append(entry)
        if not correct:
            wrong.append(entry)

    score = correct_count / N * 100
    print(f"\n{'='*50}")
    print(f"AIME {args.year} Results ({S} sample{'s' if S>1 else ''}, majority vote)")
    print(f"  Score:       {score:.1f}% ({correct_count}/{N})")
    print(f"  Errors:      {errors}")
    print(f"  Total tokens:{total_tokens:,}")
    print(f"  Wall time:   {elapsed:.1f}s")
    print(f"  Throughput:  {total_tokens/elapsed:.1f} tok/s")

    if wrong:
        print(f"\nWrong answers ({len(wrong)}):")
        for w in wrong:
            print(f"  Problem {w['idx']+1}: pred={w['pred']} (all={w['all_preds']}) correct={w['answer']}")
            print(f"    {w['problem']}")

    if args.output_dir:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        tag = f"_{args.tag}" if args.tag else ""
        with open(os.path.join(args.output_dir, f"aime{args.year}_summary{tag}.json"), "w") as f:
            json.dump({"score": score, "correct": correct_count, "total": N, "samples": S, "errors": errors}, f, indent=2)
        with open(os.path.join(args.output_dir, f"aime{args.year}_details{tag}.json"), "w") as f:
            # Save full raw outputs
            full_details = []
            for i in range(N):
                samples = per_problem.get(i, [])
                full_details.append({
                    "idx": i,
                    "problem": problems[i][0],
                    "answer": problems[i][1],
                    "pred": details[i]["pred"],
                    "correct": details[i]["correct"],
                    "samples": [{"pred": r["pred"], "raw": r["raw"], "tokens": r["tokens"]} for r in samples],
                })
            json.dump(full_details, f, indent=2, ensure_ascii=False)
        print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
