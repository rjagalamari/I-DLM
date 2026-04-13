"""
LiveCodeBench (LCB) evaluation across multiple sglang servers.
Uses official OpenCompass prompt format, code extraction, and test execution.

Usage:
  python scripts/eval_lcb.py                          # v6, all problems, 8 GPUs
  python scripts/eval_lcb.py --version 5              # v5
  python scripts/eval_lcb.py --num-problems 10        # quick test
"""
import argparse
import base64
import importlib.util
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, TimeoutError as MPTimeoutError

import requests
from huggingface_hub import hf_hub_download

# --- Import official LCB utils directly (local copy from OpenCompass) ---
_OC_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lcb_lib")

def _import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_extract_utils = _import_file("lcb_extract_utils", f"{_OC_BASE}/extract_utils.py")
extract_code = _extract_utils.extract_code_generation_v2

# Mock pyext so testing_util can be imported without it
import types as _types
_pyext_mock = _types.ModuleType("pyext")
class _RuntimeModuleMock:
    @staticmethod
    def from_string(name, path, code):
        mod = _types.ModuleType(name)
        exec(code, mod.__dict__)
        return mod
_pyext_mock.RuntimeModule = _RuntimeModuleMock
sys.modules["pyext"] = _pyext_mock

_testing_util = _import_file("lcb_testing_util", f"{_OC_BASE}/testing_util.py")
_run_test_oc = _testing_util.run_test


def load_private_test_cases(raw):
    """Decode compressed private test cases (json.loads(pickle.loads(zlib+base64)))."""
    if not raw:
        return []
    try:
        # Official OpenCompass format: json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw))))
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8")))))
    except Exception:
        try:
            return json.loads(raw)
        except Exception:
            return []


def build_prompt(item):
    """Build prompt matching OpenCompass format."""
    question = item["question_content"]
    starter = item.get("starter_code", "").strip()
    meta = json.loads(item["metadata"]) if isinstance(item["metadata"], str) else item["metadata"]
    fn_name = meta.get("func_name", None)

    system = (
        "You are an expert Python programmer. You will be given a question "
        "(problem specification) and will generate a correct Python program "
        "that matches the specification and passes all tests. "
        "You will NOT return anything except for the program."
    )

    if starter:
        fmt = (
            "You will use the following starter code to write the solution to "
            "the problem and enclose your code within delimiters."
        )
        user = (
            f"### Question:\n{question}\n\n"
            f"### Format: {fmt}\n"
            f"```python\n{starter}\n```\n\n"
            "### Answer: (use the provided format with backticks)\n\n"
        )
    else:
        fmt = (
            "Read the inputs from stdin solve the problem and write the answer "
            "to stdout (do not directly test on the sample inputs). "
            "Enclose your code within delimiters as follows."
        )
        user = (
            f"### Question:\n{question}\n\n"
            f"### Format: {fmt}\n"
            "```python\n# YOUR CODE HERE\n```\n\n"
            "### Answer: (use the provided format with backticks)\n\n"
        )

    return system, user


def run_test_stdin(code, inp, expected, timeout=10):
    """Run stdin/stdout test."""
    try:
        r = subprocess.run(
            ["python3", "-c", code],
            input=inp, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode == 0 and r.stdout.strip() == expected.strip()
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


PREAMBLE = (
    "from typing import *\nfrom string import *\nfrom re import *\n"
    "from datetime import *\nfrom collections import *\nfrom heapq import *\n"
    "from bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\n"
    "from statistics import *\nfrom itertools import *\nfrom functools import *\n"
    "from operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\n"
    "from builtins import *\nimport string, re, datetime, collections, heapq, bisect\n"
    "import copy, math, random, statistics, itertools, functools, operator, io, sys, json\n"
    "sys.setrecursionlimit(600000)\n"
)


def run_test_call_based(code, fn_name, test_cases, timeout=10):
    """Run call-based tests (LeetCode style).
    test_cases: list of {'input': '<json lines>', 'output': '<json>'}
    input is multi-line JSON: one line per function argument.
    """
    results = []
    for tc in test_cases:
        inp_str = tc.get("input", "")
        out_str = tc.get("output", "")

        try:
            # Parse inputs: each line is one JSON argument
            inputs = [json.loads(line) for line in inp_str.strip().split('\n') if line.strip()]
            expected = json.loads(out_str)
        except Exception:
            results.append(False)
            continue

        try:
            full_code = (
                f"{PREAMBLE}\n{code}\n"
                f"import json as _json_\n"
                f"_inputs_ = _json_.loads({repr(inp_str.strip())} if {repr(inp_str.strip()).count(chr(10)) == 0} else '[]')\n"
            )
            # Build test runner
            run_code = (
                f"{PREAMBLE}\n{code}\n"
                f"_inputs_ = [{', '.join(repr(json.loads(l)) for l in inp_str.strip().split(chr(10)) if l.strip())}]\n"
            )
            if "class Solution" in code:
                run_code += f"_sol_ = Solution()\n_out_ = _sol_.{fn_name}(*_inputs_)\n"
            else:
                run_code += f"_out_ = {fn_name}(*_inputs_)\n"
            run_code += "import json as _j; print(_j.dumps(_out_))\n"

            r = subprocess.run(
                ["python3", "-c", run_code],
                capture_output=True, text=True, timeout=timeout
            )
            if r.returncode != 0:
                results.append(False)
                continue

            actual = json.loads(r.stdout.strip())
            # Compare with flexibility (tuple vs list)
            if isinstance(actual, tuple):
                actual = list(actual)
            exp = expected[0] if isinstance(expected, list) and len(expected) == 1 else expected
            match = actual == exp
            if not match and isinstance(actual, list) and isinstance(exp, list):
                match = sorted(str(x) for x in actual) == sorted(str(x) for x in exp)
            results.append(match)
        except subprocess.TimeoutExpired:
            results.append(False)
        except Exception:
            results.append(False)
    return results


def compare_output(actual_lines, expected):
    """Compare output matching OC's custom_compare_ logic."""
    # actual_lines: list of output lines (from Capturing)
    # expected: string
    actual = "\n".join(actual_lines).strip()
    exp = expected.strip()
    if actual == exp:
        return True
    # Try line-by-line stripped comparison
    act_lines = [l.strip() for l in actual.splitlines() if l.strip()]
    exp_lines = [l.strip() for l in exp.splitlines() if l.strip()]
    return act_lines == exp_lines


def run_stdin_test(code, inp, expected, timeout=10):
    """Run stdin/stdout test matching OC's approach."""
    try:
        r = subprocess.run(
            ["python3", "-c", code],
            input=inp, capture_output=True, text=True, timeout=timeout
        )
        if r.returncode != 0:
            return False, "re"
        actual = r.stdout.strip()
        exp = expected.strip()
        if actual == exp:
            return True, None
        # Try split-line comparison (OC fallback)
        act_lines = [l.strip() for l in actual.splitlines() if l.strip()]
        exp_lines = [l.strip() for l in exp.splitlines() if l.strip()]
        if act_lines == exp_lines:
            return True, None
        return False, "wa"
    except subprocess.TimeoutExpired:
        return False, "tle"
    except Exception:
        return False, "re"


def run_call_test(code, fn_name, inp_str, expected_str, timeout=10):
    """Run call-based test matching OC's approach."""
    try:
        inputs = [json.loads(line) for line in inp_str.strip().split('\n') if line.strip()]
        expected = json.loads(expected_str)
    except Exception:
        return False, "parse_error"

    try:
        inputs_repr = repr(inputs)
        if "class Solution" in code:
            run_code = (
                f"{PREAMBLE}\n{code}\n"
                f"_sol_ = Solution()\n"
                f"_out_ = _sol_.{fn_name}(*{inputs_repr})\n"
                f"import json as _j; print(_j.dumps(_out_))\n"
            )
        else:
            run_code = (
                f"{PREAMBLE}\n{code}\n"
                f"_out_ = {fn_name}(*{inputs_repr})\n"
                f"import json as _j; print(_j.dumps(_out_))\n"
            )
        r = subprocess.run(["python3", "-c", run_code], capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return False, "re"
        actual = json.loads(r.stdout.strip())
        if isinstance(actual, tuple):
            actual = list(actual)
        # Flexible comparison matching OC
        exp = expected[0] if isinstance(expected, list) and len(expected) == 1 else expected
        if actual == exp:
            return True, None
        if isinstance(actual, list) and isinstance(exp, list):
            if sorted(str(x) for x in actual) == sorted(str(x) for x in exp):
                return True, None
        return False, "wa"
    except subprocess.TimeoutExpired:
        return False, "tle"
    except Exception:
        return False, "re"


def evaluate_solution(code, item):
    """Evaluate code against private test cases using subprocess."""
    if not code or not code.strip():
        return False, "empty"

    meta = json.loads(item["metadata"]) if isinstance(item["metadata"], str) else item["metadata"]
    fn_name = meta.get("func_name", None)

    # Load private test cases
    tcs = load_private_test_cases(item["private_test_cases"])
    if not tcs:
        tcs = json.loads(item["public_test_cases"]) if isinstance(item["public_test_cases"], str) else item["public_test_cases"]
    if not tcs:
        return False, "no_tests"

    for tc in tcs:
        inp = tc.get("input", "")
        exp = tc.get("output", "")
        if fn_name:
            ok, reason = run_call_test(code, fn_name, inp, exp)
        else:
            ok, reason = run_stdin_test(code, inp, exp)
        if not ok:
            return False, reason

    return True, None


_OC_EVAL_SCRIPT = """
import sys, types, importlib.util, json, base64, zlib, pickle

_OC_BASE = "__OC_BASE_PLACEHOLDER__"
def _import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_pyext_mock = types.ModuleType("pyext")
class _RM:
    @staticmethod
    def from_string(name, path, c):
        mod = types.ModuleType(name)
        exec(c, mod.__dict__)
        return mod
_pyext_mock.RuntimeModule = _RM
sys.modules["pyext"] = _pyext_mock
tu = _import_file("tu", f"{_OC_BASE}/testing_util.py")

data = json.loads(sys.stdin.read())
code = data["code"]
item = data["item"]

meta = json.loads(item["metadata"]) if isinstance(item["metadata"], str) else item["metadata"]
fn_name = meta.get("func_name", None)

pub_tcs = json.loads(item["public_test_cases"]) if isinstance(item["public_test_cases"], str) else item["public_test_cases"]
try:
    priv_tcs = json.loads(pickle.loads(zlib.decompress(base64.b64decode(item["private_test_cases"].encode("utf-8")))))
except:
    priv_tcs = []
all_tcs = pub_tcs + priv_tcs

if not all_tcs:
    print(json.dumps({"ok": False, "reason": "no_tests"}))
    sys.exit(0)

input_output = json.dumps({
    "inputs": [tc["input"] for tc in all_tcs],
    "outputs": [tc["output"] for tc in all_tcs],
    "fn_name": fn_name,
})
sample = {"input_output": input_output}

try:
    results, _ = tu.run_test(sample, test=code, debug=False, timeout=6)
    ok = bool(results and all(r is True for r in results))
    print(json.dumps({"ok": ok, "reason": None if ok else "wa"}))
except Exception as e:
    print(json.dumps({"ok": False, "reason": f"error:{e}"}))
"""


def evaluate_solution_oc(code, item):
    """Run OC evaluator in a fresh subprocess (avoids reliability_guard pollution)."""
    if not code or not code.strip():
        return False, "empty"
    try:
        payload = json.dumps({"code": code, "item": item})
        r = subprocess.run(
            ["python3", "-c", _OC_EVAL_SCRIPT.replace("__OC_BASE_PLACEHOLDER__", _OC_BASE)],
            input=payload, capture_output=True, text=True, timeout=300
        )
        if r.returncode != 0 or not r.stdout.strip():
            return False, f"re:{r.stderr[:100]}"
        result = json.loads(r.stdout.strip())
        return result["ok"], result["reason"]
    except subprocess.TimeoutExpired:
        return False, "tle"
    except Exception as e:
        return False, f"error:{e}"


def evaluate_all(gen_results):
    """Evaluate all solutions in parallel, each in a fresh subprocess."""
    sorted_results = sorted(gen_results, key=lambda x: x[0])

    def eval_one(args):
        idx, code, item, comp, finish, err = args
        if err or not code or not code.strip():
            return (False, "empty")
        return evaluate_solution_oc(code, item)

    with ThreadPoolExecutor(max_workers=16) as pool:
        evals = list(pool.map(eval_one, sorted_results))

    return evals


def run_one(args):
    idx, system, user, item, port, max_tokens, timeout, temperature, top_p, top_k = args
    try:
        r = requests.post(f"http://localhost:{port}/v1/chat/completions", json={
            "model": "default",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }, timeout=timeout).json()
        content = r["choices"][0]["message"]["content"]
        comp = r["usage"]["completion_tokens"]
        finish = r["choices"][0]["finish_reason"]
        code = extract_code(content, model_type="chat")
        return idx, code, item, comp, finish, None
    except Exception as e:
        return idx, "", item, 0, "error", str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=6)
    parser.add_argument("--num-problems", type=int, default=0)
    parser.add_argument("--ports", type=int, nargs="+", default=[30000 + i for i in range(8)])
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    filename = f"test{args.version}.jsonl" if args.version > 1 else "test.jsonl"
    path = hf_hub_download("livecodebench/code_generation_lite", filename, repo_type="dataset")
    data = [json.loads(l) for l in open(path)]
    N = min(args.num_problems, len(data)) if args.num_problems > 0 else len(data)
    data = data[:N]
    ports = args.ports

    n_call = sum(1 for item in data if (json.loads(item["metadata"]) if isinstance(item["metadata"], str) else item["metadata"]).get("func_name"))
    n_stdin = N - n_call
    print(f"LiveCodeBench v{args.version}: {N} problems ({n_call} call-based, {n_stdin} stdin)")

    tasks = []
    for i, item in enumerate(data):
        system, user = build_prompt(item)
        tasks.append((i, system, user, item, ports[i % len(ports)],
                      args.max_tokens, args.timeout, args.temperature, args.top_p, args.top_k))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(run_one, t) for t in tasks]
        gen_results = [f.result() for f in futures]
    gen_time = time.time() - t0
    print(f"Generation done in {gen_time:.1f}s. Evaluating...")

    eval_results = evaluate_all(gen_results)

    passed = 0
    by_type = {"call": {"pass": 0, "total": 0}, "stdin": {"pass": 0, "total": 0}}
    truncated = empty = wa = 0
    total_tok = 0
    samples = []

    for (idx, code, item, comp, finish, err), (ok, reason) in zip(
        sorted(gen_results, key=lambda x: x[0]), eval_results
    ):
        total_tok += comp
        meta = json.loads(item["metadata"]) if isinstance(item["metadata"], str) else item["metadata"]
        fn_name = meta.get("func_name", None)
        t = "call" if fn_name else "stdin"
        by_type[t]["total"] += 1

        if finish == "length":
            truncated += 1

        if ok:
            passed += 1
            by_type[t]["pass"] += 1
        elif reason in ("empty", "no_tests") or err:
            empty += 1
        else:
            wa += 1

        samples.append({"question_id": item["question_id"], "completion": code or ""})

    elapsed = time.time() - t0
    acc = passed / N * 100

    print(f"\n{'=' * 60}")
    print(f"LiveCodeBench v{args.version}: {N} problems")
    print(f"{'=' * 60}")
    print(f"Pass@1:         {passed}/{N} ({acc:.1f}%)")
    print(f"  call-based:   {by_type['call']['pass']}/{by_type['call']['total']}")
    print(f"  stdin/stdout: {by_type['stdin']['pass']}/{by_type['stdin']['total']}")
    print(f"Truncated:      {truncated}")
    print(f"Wrong answer:   {wa}")
    print(f"Empty/error:    {empty}")
    print(f"Total tokens:   {total_tok:,}")
    print(f"Wall time:      {elapsed:.1f}s")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f"lcb_v{args.version}_samples.jsonl"), "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        with open(os.path.join(args.output_dir, f"lcb_v{args.version}_summary.json"), "w") as f:
            json.dump({"accuracy": acc, "passed": passed, "total": N,
                       "truncated": truncated, "tokens": total_tok}, f, indent=2)
        print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
