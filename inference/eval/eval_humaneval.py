"""
HumanEval evaluation across multiple sglang servers.

Uses the official human_eval package for pass@1 evaluation.
Matches OpenCompass prompt template and postprocessing.

Usage:
  python scripts/eval_humaneval.py                        # all 164, 8 GPUs
  python scripts/eval_humaneval.py --num-problems 5       # quick test
"""
import argparse
import json
import os
import re
import requests
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset

from human_eval.data import HUMAN_EVAL, read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness


def strip_thinking(text):
    if not text:
        return ""
    return re.sub(r'^.*</think>\s*', '', text, count=1, flags=re.DOTALL)


def postprocess(text):
    """Extract code from model output (matches OpenCompass humaneval_postprocess_v2)."""
    text = re.sub(r'^[^\x00-\x7F]+', '', text)
    text = strip_thinking(text)
    # Find code blocks
    blocks = re.findall(r'```\w*\n(.*?)```', text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r'```\w*\s(.*?)```', text, re.DOTALL)
    if len(blocks) >= 1:
        text = blocks[0]  # Match OpenCompass: take first block
    return text.lstrip()


def run_one(args):
    i, prompt_code, port, max_tokens, timeout, temperature, top_p, top_k = args
    prompt = (
        "Read the following function signature and docstring, and fully implement "
        "the function described. Your response should only contain the code for "
        f"this function.\n{prompt_code}"
    )
    try:
        r = requests.post(f"http://localhost:{port}/v1/chat/completions", json={
            "model": "sdar",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
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
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    ds = load_dataset("openai/openai_humaneval", split="test")
    N = min(args.num_problems, len(ds)) if args.num_problems else len(ds)
    ds_sub = ds.select(range(N))
    ports = args.ports
    max_workers = args.max_workers or N

    print(f"HumanEval: {N} problems, {len(ports)} servers, max_tokens={args.max_tokens}")

    tasks = [
        (i, ds_sub[i]["prompt"], ports[i % len(ports)], args.max_tokens, args.timeout,
         args.temperature, args.top_p, args.top_k)
        for i in range(N)
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

    # Postprocess and prepare for evaluation
    predictions = [""] * N
    total_tok = 0
    errors = 0
    for i, pred, comp, err in results:
        predictions[i] = pred
        total_tok += comp
        if err:
            errors += 1

    # Build human_eval format
    humaneval_preds = []
    for i in range(N):
        code = postprocess(predictions[i])
        task_id = ds_sub[i]["task_id"]
        humaneval_preds.append({"task_id": task_id, "completion": code})

    # Evaluate using official human_eval
    # For subset eval, create a filtered problem file
    with tempfile.TemporaryDirectory() as tmp_dir:
        pred_file = os.path.join(tmp_dir, "humaneval_preds.jsonl")
        write_jsonl(pred_file, humaneval_preds)

        if N < 164:
            # Create subset problem file for partial eval
            all_problems = read_problems()
            subset_ids = {ds_sub[i]["task_id"] for i in range(N)}
            subset = [v for k, v in all_problems.items() if k in subset_ids]
            problem_file = os.path.join(tmp_dir, "problems_subset.jsonl")
            write_jsonl(problem_file, subset)
            score = evaluate_functional_correctness(
                pred_file, k=[1], n_workers=4, timeout=10.0, problem_file=problem_file
            )
        else:
            score = evaluate_functional_correctness(
                pred_file, k=[1], n_workers=4, timeout=10.0
            )

    pass_at_1 = score["pass@1"] * 100

    print(f"\n{'=' * 55}")
    print(f"HumanEval {N} problems, {len(ports)} GPUs")
    print(f"{'=' * 55}")
    print(f"pass@1:         {pass_at_1:.1f}%")
    print(f"Total tokens:   {total_tok:,}")
    print(f"Wall time:      {elapsed:.1f}s")
    print(f"Throughput:     {total_tok / elapsed:.1f} tok/s")
    print(f"Errors:         {errors}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        tag = f"_{args.tag}" if args.tag else ""
        with open(os.path.join(args.output_dir, f"humaneval_summary{tag}.json"), "w") as f:
            json.dump({"pass_at_1": pass_at_1, "scores": score, "total_tok": total_tok, "errors": errors}, f, indent=2)
        # Save raw predictions
        details = [{"idx": i, "task_id": ds_sub[i]["task_id"], "prediction_raw": predictions[i],
                     "code_extracted": postprocess(predictions[i])} for i in range(N)]
        with open(os.path.join(args.output_dir, f"humaneval_details{tag}.json"), "w") as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
