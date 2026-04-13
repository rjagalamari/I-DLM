"""
GSM8K evaluation across multiple sglang servers.

Usage:
  python scripts/eval_gsm8k.py                     # 200 problems, 8 GPUs
  python scripts/eval_gsm8k.py --num-problems 1319  # full test set
  python scripts/eval_gsm8k.py --ports 30000 30001   # custom ports
"""
import argparse
import json
import os
import requests
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset


def run_one(args):
    i, q, gold, port, max_tokens, temperature, top_p, top_k, timeout = args
    prompt = f"{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    try:
        r = requests.post(f"http://localhost:{port}/v1/chat/completions", json={
            "model": "sdar",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }, timeout=timeout).json()
        c = r["choices"][0]["message"]["content"]
        comp = r["usage"]["completion_tokens"]
        finish = r["choices"][0]["finish_reason"]
        boxed = re.findall(r'\\boxed\{([^}]+)\}', c)
        if boxed:
            pred = boxed[-1].replace(",", "").replace("$", "").replace("\\", "").strip()
        else:
            after = c.split("</think>")[-1] if "</think>" in c else c
            nums = re.findall(r'[\d,]+', after)
            pred = nums[-1].replace(",", "") if nums else "?"
        return i, pred, gold, comp, finish, None
    except Exception as e:
        return i, "?", gold, 0, "error", str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=0, help="0=full dataset (1319)")
    parser.add_argument("--ports", type=int, nargs="+", default=[30000 + i for i in range(8)])
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    ds = load_dataset("gsm8k", "main", split="test")
    N = len(ds) if args.num_problems == 0 else min(args.num_problems, len(ds))
    problems = [(item["question"], item["answer"].split("####")[-1].strip().replace(",", ""))
                for item in ds.select(range(N))]
    ports = args.ports
    max_workers = args.max_workers or N

    print(f"GSM8K eval: {N} problems, {len(ports)} servers, {max_workers} workers")
    print(f"Config: max_tokens={args.max_tokens}, temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")

    tasks = [
        (i, q, g, ports[i % len(ports)], args.max_tokens, args.temperature, args.top_p, args.top_k, args.timeout)
        for i, (q, g) in enumerate(problems)
    ]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_one, t) for t in tasks]
        done = 0
        for f in as_completed(futures):
            done += 1
            if done % max(N // 5, 1) == 0:
                print(f"  {done}/{N} done ({time.time() - t0:.0f}s)")
        results = [f.result() for f in futures]

    elapsed = time.time() - t0

    correct = total_tok = errors = length_limited = 0
    for i, pred, gold, comp, finish, err in sorted(results):
        correct += (pred == gold)
        total_tok += comp
        if err:
            errors += 1
        if finish == "length":
            length_limited += 1

    print(f"\n{'=' * 60}")
    print(f"GSM8K {N} problems, {len(ports)} GPUs")
    print(f"{'=' * 60}")
    print(f"Accuracy:          {correct}/{N} ({correct / N * 100:.1f}%)")
    if errors > 0 and N - errors > 0:
        print(f"Accuracy (no err): {correct}/{N - errors} ({correct / (N - errors) * 100:.1f}%)")
    print(f"Total tokens:      {total_tok:,}")
    print(f"Wall time:         {elapsed:.1f}s")
    print(f"Total throughput:  {total_tok / elapsed:.1f} tok/s")
    print(f"Per-GPU avg:       {total_tok / elapsed / len(ports):.1f} tok/s")
    print(f"Avg tok/problem:   {total_tok / N:.0f}")
    print(f"Hit max_tokens:    {length_limited}")
    print(f"Errors (timeout):  {errors}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        tag = f"_{args.tag}" if args.tag else ""
        with open(os.path.join(args.output_dir, f"gsm8k_summary{tag}.json"), "w") as f:
            json.dump({"accuracy": correct / N * 100, "correct": correct, "total": N,
                       "errors": errors, "total_tok": total_tok}, f, indent=2)
        print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
