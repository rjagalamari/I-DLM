"""
IFEval evaluation across multiple sglang servers.

Uses the official Google IFEval dataset and evaluator.
Strips <think>...</think> blocks before scoring (Qwen3 thinking mode).

Usage:
  python scripts/eval_ifeval.py                          # all 541 prompts, 8 GPUs
  python scripts/eval_ifeval.py --num-problems 5         # quick test
  python scripts/eval_ifeval.py --ports 30000             # single GPU

Requires: pip install datasets langdetect immutabledict
"""
import argparse
import json
import re
import requests
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset

# IFEval evaluator — import from local ifeval_lib (Google Research IFEval code)
import os as _os
_IFEVAL_LIB = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "ifeval_lib")
sys.path.insert(0, _os.path.dirname(_IFEVAL_LIB))
# Register ifeval_lib as 'instruction_following_eval' so internal imports resolve
import importlib as _importlib
_ifeval_pkg = _importlib.import_module("ifeval_lib")
sys.modules["instruction_following_eval"] = _ifeval_pkg
from ifeval_lib.evaluation_lib import InputExample
from ifeval_lib.evaluation_lib import test_instruction_following_strict as _test_strict_orig
from ifeval_lib.evaluation_lib import test_instruction_following_loose as _test_loose_orig

def test_instruction_following_strict(inp, response):
    """Wrapper: Google IFEval expects dict {prompt: response}, our script passes response directly."""
    return _test_strict_orig(inp, {inp.prompt: response})

def test_instruction_following_loose(inp, response):
    """Wrapper: Google IFEval expects dict {prompt: response}, our script passes response directly."""
    return _test_loose_orig(inp, {inp.prompt: response})


def strip_thinking(text):
    """Remove everything before and including the last </think> from model output."""
    if not text:
        return ""
    return re.sub(r'^.*</think>\s*', '', text, count=1, flags=re.DOTALL)


def run_one(args):
    i, prompt, ref, port, max_tokens, timeout, temperature, top_p, top_k = args
    try:
        r = requests.post(f"http://localhost:{port}/v1/chat/completions", json={
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }, timeout=timeout).json()
        pred = r["choices"][0]["message"]["content"]
        comp = r["usage"]["completion_tokens"]
        return i, pred, ref, comp, None
    except Exception as e:
        return i, "", ref, 0, str(e)


def evaluate(predictions, references):
    """Score predictions using IFEval strict/loose metrics."""
    prompt_strict_correct = prompt_strict_total = 0
    inst_strict_correct = inst_strict_total = 0
    prompt_loose_correct = prompt_loose_total = 0
    inst_loose_correct = inst_loose_total = 0

    for pred, ref in zip(predictions, references):
        pred = strip_thinking(pred)
        inp = InputExample(
            key=ref["key"],
            instruction_id_list=ref["instruction_id_list"],
            prompt=ref["prompt"],
            kwargs=[{k: v for k, v in kw.items() if v is not None} for kw in ref["kwargs"]],
        )

        # Strict
        ex = test_instruction_following_strict(inp, pred)
        prompt_strict_total += 1
        prompt_strict_correct += all(ex.follow_instruction_list)
        inst_strict_total += len(ex.instruction_id_list)
        inst_strict_correct += sum(ex.follow_instruction_list)

        # Loose
        ex = test_instruction_following_loose(inp, pred)
        prompt_loose_total += 1
        prompt_loose_correct += all(ex.follow_instruction_list)
        inst_loose_total += len(ex.instruction_id_list)
        inst_loose_correct += sum(ex.follow_instruction_list)

    return {
        "Prompt-level-strict": prompt_strict_correct / max(prompt_strict_total, 1) * 100,
        "Inst-level-strict": inst_strict_correct / max(inst_strict_total, 1) * 100,
        "Prompt-level-loose": prompt_loose_correct / max(prompt_loose_total, 1) * 100,
        "Inst-level-loose": inst_loose_correct / max(inst_loose_total, 1) * 100,
        "n_prompts": prompt_strict_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-problems", type=int, default=None, help="Limit number of problems (default: all 541)")
    parser.add_argument("--ports", type=int, nargs="+", default=[30000 + i for i in range(8)])
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="Save raw results to this dir")
    parser.add_argument("--tag", type=str, default="", help="Tag for output filename")
    args = parser.parse_args()

    ds = load_dataset("google/IFEval", split="train")
    N = min(args.num_problems, len(ds)) if args.num_problems else len(ds)
    ds = ds.select(range(N))
    ports = args.ports
    max_workers = args.max_workers or N

    print(f"IFEval: {N} prompts, {len(ports)} servers, max_tokens={args.max_tokens}")

    # Build references
    references = []
    for item in ds:
        references.append({
            "key": item["key"],
            "prompt": item["prompt"],
            "instruction_id_list": item["instruction_id_list"],
            "kwargs": json.loads(item["kwargs"]) if isinstance(item["kwargs"], str) else item["kwargs"],
        })

    # Build tasks
    tasks = [
        (i, item["prompt"], references[i], ports[i % len(ports)], args.max_tokens, args.timeout, args.temperature, args.top_p, args.top_k)
        for i, item in enumerate(ds)
    ]

    print(f"Running {N} prompts, {max_workers} concurrent workers...")
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

    # Collect predictions
    predictions = [""] * N
    total_tok = 0
    errors = 0
    for i, pred, ref, comp, err in results:
        predictions[i] = pred
        total_tok += comp
        if err:
            errors += 1

    # Detailed per-prompt evaluation
    detail_results = []
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        pred_stripped = strip_thinking(pred)
        inp = InputExample(
            key=ref["key"],
            instruction_id_list=ref["instruction_id_list"],
            prompt=ref["prompt"],
            kwargs=[{k: v for k, v in kw.items() if v is not None} for kw in ref["kwargs"]],
        )
        ex_strict = test_instruction_following_strict(inp, pred_stripped)
        ex_loose = test_instruction_following_loose(inp, pred_stripped)
        strict_pass = all(ex_strict.follow_instruction_list)
        loose_pass = all(ex_loose.follow_instruction_list)
        failed_strict = [iid for iid, ok in zip(ex_strict.instruction_id_list, ex_strict.follow_instruction_list) if not ok]
        detail_results.append({
            "idx": i,
            "key": ref["key"],
            "prompt": ref["prompt"],
            "instruction_ids": ref["instruction_id_list"],
            "prediction_raw": pred,
            "prediction_stripped": pred_stripped,
            "strict_pass": strict_pass,
            "loose_pass": loose_pass,
            "failed_strict_ids": failed_strict,
            "strict_follow_list": ex_strict.follow_instruction_list,
            "loose_follow_list": ex_loose.follow_instruction_list,
        })

    # Compute scores
    scores = evaluate(predictions, references)

    print(f"\n{'=' * 60}")
    print(f"IFEval {N} prompts, {len(ports)} GPUs")
    print(f"{'=' * 60}")
    print(f"Prompt-level strict accuracy: {scores['Prompt-level-strict']:.2f}%")
    print(f"Inst-level strict accuracy:   {scores['Inst-level-strict']:.2f}%")
    print(f"Prompt-level loose accuracy:  {scores['Prompt-level-loose']:.2f}%")
    print(f"Inst-level loose accuracy:    {scores['Inst-level-loose']:.2f}%")
    print(f"")
    print(f"Total tokens:     {total_tok:,}")
    print(f"Wall time:        {elapsed:.1f}s")
    print(f"Throughput:       {total_tok / elapsed:.1f} tok/s")
    print(f"Errors:           {errors}")

    # Save raw results
    output_dir = args.output_dir or "output_ifeval"
    import os
    os.makedirs(output_dir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    # Save detailed results
    detail_path = os.path.join(output_dir, f"ifeval_details{tag}.json")
    with open(detail_path, "w") as f:
        json.dump(detail_results, f, indent=2, ensure_ascii=False)

    # Save summary
    summary = {
        "scores": scores,
        "total_tokens": total_tok,
        "wall_time": elapsed,
        "throughput": total_tok / elapsed,
        "errors": errors,
        "n_prompts": N,
        "n_failed_strict": sum(1 for d in detail_results if not d["strict_pass"]),
        "failed_indices": [d["idx"] for d in detail_results if not d["strict_pass"]],
    }
    summary_path = os.path.join(output_dir, f"ifeval_summary{tag}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Failed indices ({summary['n_failed_strict']}): {summary['failed_indices']}")


if __name__ == "__main__":
    main()
