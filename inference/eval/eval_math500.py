"""
MATH-500 (PRM800K subset) evaluation across multiple sglang servers.

Uses math_verify for answer verification (same as OpenCompass MATHVerifyEvaluator).
Strips <think>...</think> before extracting \boxed{} answers.

Usage:
  python scripts/eval_math500.py                        # all 500, 8 GPUs
  python scripts/eval_math500.py --num-problems 5       # quick test
"""
import argparse
import json
import re
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from latex2sympy2_extended import NormalizationConfig
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify


MATH_DATA = "/data/cxu/dllm-distillation/evaluation/opencompass/.cache/opencompass/data/math/test_prm800k_500.json"
MATH_HF_DATASET = "HuggingFaceH4/MATH-500"


def strip_thinking(text):
    if not text:
        return ""
    return re.sub(r'^.*</think>\s*', '', text, count=1, flags=re.DOTALL)


def verify_answer(prediction, reference):
    """Verify prediction against reference using math_verify (OpenCompass-compatible)."""
    prediction = strip_thinking(prediction)

    ref_with_env = f'${reference}$'
    gold_parsed = parse(
        ref_with_env,
        extraction_mode='first_match',
        extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
    )

    if len(gold_parsed) == 0:
        return False

    pred_parsed = parse(
        prediction,
        extraction_config=[
            LatexExtractionConfig(
                boxed_match_priority=0,
            ),
            ExprExtractionConfig(),
        ],
    )

    if len(pred_parsed) == 0:
        return False

    try:
        return verify(gold_parsed, pred_parsed)
    except Exception:
        return False


def run_one(args):
    i, problem, port, max_tokens, timeout = args
    prompt = f"{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    try:
        r = requests.post(f"http://localhost:{port}/v1/chat/completions", json={
            "model": "sdar",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 50,
        }, timeout=timeout).json()
        pred = r["choices"][0]["message"]["content"]
        comp = r["usage"]["completion_tokens"]
        return i, pred, comp, None
    except Exception as e:
        return i, "", 0, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=None)
    parser.add_argument("--ports", type=int, nargs="+", default=[30000 + i for i in range(8)])
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    import os
    if os.path.exists(MATH_DATA):
        with open(MATH_DATA) as f:
            data = json.load(f)
        problems = [(data[str(i)]["problem"], data[str(i)]["solution"]) for i in range(len(data))]
    else:
        from datasets import load_dataset
        ds = load_dataset(MATH_HF_DATASET, split="test")
        problems = [(item["problem"], item["answer"]) for item in ds]
    N = min(args.num_problems, len(problems)) if args.num_problems else len(problems)
    problems = problems[:N]
    ports = args.ports
    max_workers = args.max_workers or N

    print(f"MATH-500: {N} problems, {len(ports)} servers, max_tokens={args.max_tokens}")

    tasks = [
        (i, prob, ports[i % len(ports)], args.max_tokens, args.timeout)
        for i, (prob, _) in enumerate(problems)
    ]

    print(f"Running {N} problems, {max_workers} concurrent workers...")
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

    # Verify
    predictions = [""] * N
    total_tok = 0
    errors = 0
    for i, pred, comp, err in results:
        predictions[i] = pred
        total_tok += comp
        if err:
            errors += 1

    correct = 0
    details = []
    for i, (pred, (prob, sol)) in enumerate(zip(predictions, problems)):
        is_correct = verify_answer(pred, sol)
        correct += is_correct
        details.append({
            "idx": i,
            "problem": prob[:100],
            "solution": sol,
            "prediction": pred,
            "correct": is_correct,
        })

    score = correct / N * 100
    print(f"\n{'=' * 55}")
    print(f"MATH-500 {N} problems, {len(ports)} GPUs")
    print(f"{'=' * 55}")
    print(f"Accuracy:       {score:.1f}% ({correct}/{N})")
    print(f"Total tokens:   {total_tok:,}")
    print(f"Wall time:      {elapsed:.1f}s")
    print(f"Throughput:     {total_tok / elapsed:.1f} tok/s")
    print(f"Errors:         {errors}")

    if args.output_dir:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        tag = f"_{args.tag}" if args.tag else ""
        with open(os.path.join(args.output_dir, f"math500_summary{tag}.json"), "w") as f:
            json.dump({"score": score, "correct": correct, "total": N, "errors": errors}, f, indent=2)
        with open(os.path.join(args.output_dir, f"math500_details{tag}.json"), "w") as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
