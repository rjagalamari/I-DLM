"""
Sanitized MBPP evaluation across multiple sglang servers.

Sends coding prompts, extracts Python code from responses (strips thinking),
executes against test cases, reports pass rate.

Usage:
  python scripts/eval_mbpp.py                        # all 257 problems, 8 GPUs
  python scripts/eval_mbpp.py --num-problems 5       # quick test
  python scripts/eval_mbpp.py --ports 30000           # single GPU
"""
import argparse
import json
import re
import requests
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datasets import load_dataset


def strip_thinking(text):
    if not text:
        return ""
    return re.sub(r'^.*</think>\s*', '', text, count=1, flags=re.DOTALL)


def extract_code(text):
    """Extract Python code from model output.

    Strategy: find ALL ```python blocks, pick the last one containing 'def '.
    This handles models that show test cases first, then the solution.
    """
    text = strip_thinking(text)
    text = re.sub(r'^[^\x00-\x7F]+', '', text)  # strip leading non-ASCII

    # Find all ```python...``` blocks
    blocks = re.findall(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r'```\s*(.*?)\s*```', text, re.DOTALL)

    if blocks:
        # Prefer the last block containing 'def ' (the final solution)
        def_blocks = [b for b in blocks if 'def ' in b]
        if def_blocks:
            return def_blocks[-1].strip()
        # Otherwise take the last block
        return blocks[-1].strip()

    # Fallback: unclosed ```python block
    match = re.search(r'```python\s*(.*)', text, re.DOTALL)
    if match:
        return match.group(1).split('```')[0].strip()

    return text.strip()


def execute_code(program, timeout=10):
    """Execute code and return 'pass', 'failed', 'timeout', or 'wrong_answer'."""
    import signal

    def handler(signum, frame):
        raise TimeoutError()

    try:
        old = signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)
        exec_globals = {}
        exec(program, exec_globals)
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        return 'pass'
    except TimeoutError:
        return 'timeout'
    except AssertionError:
        return 'wrong_answer'
    except Exception:
        return 'failed'


def run_one(args):
    i, text, test_list, port, max_tokens, timeout = args
    prompt = (
        f"You are an expert Python programmer, and here is your task:\n{text}\n"
        f"Your code should pass these tests:\n\n{test_list}\n"
        f" You should submit your final solution in the following format: "
        f"```python\n\n```"
    )
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
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    N = min(args.num_problems, len(ds)) if args.num_problems else len(ds)
    ds = ds.select(range(N))
    ports = args.ports
    max_workers = args.max_workers or N

    print(f"MBPP: {N} problems, {len(ports)} servers, max_tokens={args.max_tokens}")

    # Build tasks
    tasks = []
    test_cases = []
    for i, item in enumerate(ds):
        text = item["prompt"]
        test_list_str = "\n".join(item["test_list"])
        test_cases.append("\n".join(item["test_list"]))
        tasks.append((i, text, test_list_str, ports[i % len(ports)], args.max_tokens, args.timeout))

    # Generate predictions
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

    gen_time = time.time() - t0

    # Extract code and execute
    print("Executing test cases...")
    predictions = [""] * N
    total_tok = 0
    errors = 0
    for i, pred, comp, err in results:
        predictions[i] = pred
        total_tok += comp
        if err:
            errors += 1

    codes = [extract_code(p) for p in predictions]

    # Execute with process pool (sandboxed)
    result_counts = {'pass': 0, 'timeout': 0, 'failed': 0, 'wrong_answer': 0}
    details = []

    for i in range(N):
        code = codes[i]
        program = code + "\n" + test_cases[i]
        ret = execute_code(program)
        result_counts[ret] += 1
        details.append({
            "idx": i,
            "prompt": ds[i]["prompt"],
            "prediction_raw": predictions[i],
            "code_extracted": code,
            "result": ret,
        })

    elapsed = time.time() - t0
    score = result_counts['pass'] / N * 100

    print(f"\n{'=' * 55}")
    print(f"MBPP {N} problems, {len(ports)} GPUs")
    print(f"{'=' * 55}")
    print(f"Score:          {score:.1f}% ({result_counts['pass']}/{N} pass)")
    print(f"  pass:         {result_counts['pass']}")
    print(f"  wrong_answer: {result_counts['wrong_answer']}")
    print(f"  failed:       {result_counts['failed']}")
    print(f"  timeout:      {result_counts['timeout']}")
    print(f"Total tokens:   {total_tok:,}")
    print(f"Gen time:       {gen_time:.1f}s")
    print(f"Throughput:     {total_tok / gen_time:.1f} tok/s")
    print(f"Errors:         {errors}")

    # Save
    if args.output_dir:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        tag = f"_{args.tag}" if args.tag else ""
        with open(os.path.join(args.output_dir, f"mbpp_details{tag}.json"), "w") as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        with open(os.path.join(args.output_dir, f"mbpp_summary{tag}.json"), "w") as f:
            json.dump({"score": score, "counts": result_counts, "total_tok": total_tok, "errors": errors}, f, indent=2)
        print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
