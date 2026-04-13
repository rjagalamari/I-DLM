"""
GPQA Diamond evaluation across multiple sglang servers.

198 graduate-level science questions (4-choice). Uses Idavidrein/gpqa.

Usage:
  python scripts/eval_gpqa.py                          # all 198, 8 GPUs
  python scripts/eval_gpqa.py --num-problems 50
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
    """Extract A-D from model output. OC format: last line 'ANSWER: X'."""
    if "</think>" in text:
        after = re.sub(r"^.*?</think>\s*", "", text, count=1, flags=re.DOTALL)
    else:
        after = text[-500:]
    m = re.search(r"ANSWER:\s*([A-Da-d])", after)
    if m:
        return m.group(1).upper()
    # "answer is D", "Answer: D", "**Answer: D**", "**Answer:** A) ...", "answer: B"
    # Require "is" or ":" after "answer" to avoid matching "answer choices"
    m = re.search(r"[Aa]nswer\s*(?:is\s*:?\s*|:\s*)\*{0,2}\s*\(?([A-Da-d])\)?\*{0,2}", after)
    if m:
        return m.group(1).upper()
    m = re.search(r"\\boxed\{[^}]*?([A-Da-d])[^A-Da-d}]*\}", after)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?:\*\*\(?([A-Da-d])\)?\*\*|\(([A-Da-d])\))\s*[.\s]*$", after.strip())
    if m:
        return (m.group(1) or m.group(2)).upper()
    return "?"


def run_one(args):
    idx, prompt, gold, port, max_tokens, timeout, temperature, top_p, top_k = args
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
        comp = r["usage"]["completion_tokens"]
        pred = extract_choice(content)
        return idx, pred, gold, comp, content, None
    except Exception as e:
        return idx, "?", gold, 0, "", str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=0, help="0 = all")
    parser.add_argument("--subset", type=str, default="diamond",
                        choices=["diamond", "main", "extended"],
                        help="GPQA subset: diamond(198), main(448), extended(546)")
    parser.add_argument("--ports", type=int, nargs="+", default=[30000 + i for i in range(8)])
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-workers", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    subset_map = {"diamond": "gpqa_diamond", "main": "gpqa_main", "extended": "gpqa_extended"}
    hf_subset = subset_map[args.subset]
    ds = load_dataset("Idavidrein/gpqa", hf_subset, split="train")
    N = min(args.num_problems, len(ds)) if args.num_problems > 0 else len(ds)
    ports = args.ports

    problems = []
    for item in ds.select(range(N)):
        q = item["Question"]
        choices = [
            item["Correct Answer"],
            item["Incorrect Answer 1"],
            item["Incorrect Answer 2"],
            item["Incorrect Answer 3"],
        ]
        # Shuffle choices deterministically, gold is always at original position
        import hashlib
        seed = int(hashlib.md5(q.encode()).hexdigest()[:8], 16)
        import random
        rng = random.Random(seed)
        indices = list(range(4))
        rng.shuffle(indices)
        shuffled = [choices[i] for i in indices]
        gold_idx = indices.index(0)  # Correct Answer was at index 0
        gold_letter = "ABCD"[gold_idx]

        # OC format: "A) choice\nB) choice..." with "ANSWER: $LETTER" instruction
        choices_str = "\n".join(f"{'ABCD'[i]}) {shuffled[i]}" for i in range(4))
        prompt = (
            "Answer the following multiple choice question. "
            "The last line of your response should be of the following format: "
            "'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. "
            "Think step by step before answering.\n\n"
            f"{q}\n\n{choices_str}"
        )
        problems.append((prompt, gold_letter))

    print(f"GPQA {args.subset} eval: {N} problems, {len(ports)} servers")

    tasks = [
        (i, p, g, ports[i % len(ports)], args.max_tokens, args.timeout,
         args.temperature, args.top_p, args.top_k)
        for i, (p, g) in enumerate(problems)
    ]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(run_one, t) for t in tasks]
        results = [f.result() for f in futures]

    elapsed = time.time() - t0
    correct = total_tok = errors = 0
    wrong_examples = []
    for idx, pred, gold, comp, content, err in results:
        if err:
            errors += 1
            continue
        total_tok += comp
        if pred == gold:
            correct += 1
        else:
            wrong_examples.append((idx, pred, gold, content))

    acc = correct / N * 100
    print(f"\n{'=' * 60}")
    print(f"GPQA {args.subset} {N} problems, {len(ports)} GPUs")
    print(f"{'=' * 60}")
    print(f"Accuracy:       {correct}/{N} ({acc:.1f}%)")
    print(f"Total tokens:   {total_tok:,}")
    print(f"Wall time:      {elapsed:.1f}s")
    print(f"Throughput:     {total_tok / elapsed:.1f} tok/s")
    if errors:
        print(f"Errors:         {errors}")

    # Show some wrong examples for debugging
    if wrong_examples:
        print(f"\n--- Sample wrong answers (first 10) ---")
        for idx, pred, gold, content in wrong_examples[:10]:
            # Show last 200 chars of output
            tail = content[-200:] if len(content) > 200 else content
            print(f"  #{idx}: pred={pred} gold={gold}  ...{tail}\n")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f"gpqa_{args.subset}_summary.json"), "w") as f:
            json.dump({"accuracy": acc, "correct": correct, "total": N,
                       "tokens": total_tok}, f, indent=2)
        # Save per-problem details
        all_details = []
        for idx, pred, gold, comp, content, err in results:
            all_details.append({
                "idx": idx, "pred": pred, "gold": gold,
                "correct": pred == gold, "tokens": comp,
                "output": content, "error": err
            })
        with open(os.path.join(args.output_dir, f"gpqa_{args.subset}_details.json"), "w") as f:
            json.dump(all_details, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
