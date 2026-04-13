"""
MathBench evaluation matching OpenCompass protocol.

Data: /data/cxu/dllm-distillation/evaluation/opencompass/.cache/opencompass/data/mathbench_v1/
Splits:
  Single-choice (A/B/C/D): college, high, middle, college_knowledge, high_knowledge,
                             middle_knowledge, primary_knowledge  (CN + EN each)
  Cloze (fill-in-blank):   primary (CN+EN), arithmetic (EN only)

Circular eval for single-choice: test with 4 option permutations (ABCD/BCDA/CDAB/DABC).
A problem is correct only if ALL 4 permutations are answered correctly.

Usage:
  python scripts/eval_mathbench.py
  python scripts/eval_mathbench.py --ports 30010 30011 30012 30013 30014 30015 30016 30017
  python scripts/eval_mathbench.py --no-circular   # faster, less strict
"""
import argparse
import copy
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

DATA_ROOT = "/data/cxu/dllm-distillation/evaluation/opencompass/.cache/opencompass/data/mathbench_v1"

SINGLE_CHOICE_SPLITS = [
    "college", "high", "middle",
    "college_knowledge", "high_knowledge", "middle_knowledge", "primary_knowledge",
]
CLOZE_SPLITS = ["primary", "arithmetic"]

CIRCULAR_PATTERNS = ["ABCD", "BCDA", "CDAB", "DABC"]


def strip_thinking(text):
    if not text:
        return ""
    return re.sub(r"^.*?</think>\s*", "", text, count=1, flags=re.DOTALL)


def get_number(options):
    return "\n".join(f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options))


def make_circular_examples(entry):
    examples = []
    for pattern in CIRCULAR_PATTERNS:
        e = copy.deepcopy(entry)
        e["options"] = [e["options"][ord(c) - ord("A")] for c in pattern]
        e["answer"] = {pattern[i]: chr(ord("A") + i) for i in range(4)}[e["answer"]]
        e["_pattern"] = pattern
        examples.append(e)
    return examples


def extract_single_choice(text):
    text = strip_thinking(text)
    patterns = [
        r"所以答案为选项\s*([ABCD])",
        r"答案[为是]?\s*[：:]\s*([ABCD])",
        r"答案[为是]?\s*([ABCD])[^a-zA-Z]",
        r"[Tt]herefore.*?correct.*?option.*?[Ii]s\s*\(?([ABCD])\)?",
        r"[Tt]he correct answer is\s*\(?([ABCD])\)?",
        r"[Tt]he answer is\s*\(?([ABCD])\)?",
        r"故选\s*([ABCD])",
        r"选\s*([ABCD])\s*[。.]",
        r"^([ABCD])[^a-zA-Z\d]",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m and m.group(1) in "ABCD":
            return m.group(1)
    found = re.findall(r"\b([ABCD])\b", text)
    return found[-1] if found else "?"


def extract_cloze(text, is_cn=True):
    text = strip_thinking(text)
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        raw = boxed[-1].replace(",", "").strip()
        try:
            v = float(raw)
            return str(int(v)) if v == int(v) else raw
        except Exception:
            return raw
    sep = "答案是" if is_cn else "The answer is"
    parts = text.split(sep)
    if len(parts) > 1:
        nums = re.findall(r"-?\d+\.?\d*", parts[1])
        if nums:
            return nums[0]
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else "?"


def build_single_choice_prompt(entry, is_cn):
    q = entry["question"].strip() + "\n" + get_number(entry["options"])
    if is_cn:
        prefix = '以下是一道关于数学的单项选择题，请你一步一步推理，并在最后用\u201c所以答案为选项X\u201d给出答案，其中\u201cX\u201d为选项A，B，C，D中你认为正确的选项。下面是你要回答的问题'
        return f"{prefix}\n{q}\n让我们一步一步思考：\n"
    else:
        return (
            "Here is a multiple-choice question about mathematics. Please reason through it "
            "step by step, and at the end, provide your answer option with "
            "'Therefore, the correct answer is option X', where 'X' is the correct option "
            f"from A, B, C, D.\n{q}\nLet's think step by step:\n"
        )


def build_cloze_prompt(entry, is_cn):
    q = entry["question"].strip()
    if is_cn:
        return f"{q}\n请一步一步推理，并在最后用\\boxed{{}}给出你的答案。"
    else:
        return f"{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def run_one(args):
    idx, prompt, gold, kind, is_cn, port, max_tokens, timeout, temperature, top_p, top_k = args
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
        finish = r["choices"][0]["finish_reason"]
        comp = r["usage"]["completion_tokens"]
        pred = extract_single_choice(content) if kind == "single" else extract_cloze(content, is_cn)
        return idx, pred, gold, finish, comp, None
    except Exception as e:
        return idx, "?", gold, "error", 0, str(e)


def eval_file(filepath, kind, is_cn, ports, args, task_offset):
    data = [json.loads(l) for l in open(filepath)]
    if kind == "single" and args.circular:
        all_entries = [circ for entry in data for circ in make_circular_examples(entry)]
    else:
        all_entries = data

    tasks = []
    for i, entry in enumerate(all_entries):
        if kind == "single":
            prompt = build_single_choice_prompt(entry, is_cn)
            gold = entry["answer"]
        else:
            prompt = build_cloze_prompt(entry, is_cn)
            gold = str(entry["answer"]).replace(",", "").strip()
        port = ports[(task_offset + i) % len(ports)]
        tasks.append((i, prompt, gold, kind, is_cn, port,
                      args.max_tokens, args.timeout, args.temperature, args.top_p, args.top_k))

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(run_one, t) for t in tasks]
        results = [f.result() for f in futures]

    if kind == "single" and args.circular:
        n = len(data)
        correct = sum(
            1 for i in range(n)
            if all(results[i*4+c][1] == results[i*4+c][2] for c in range(4))
        )
        total = n
    else:
        correct = sum(1 for _, pred, gold, _, _, _ in results if pred == gold)
        total = len(results)

    truncated = sum(1 for r in results if r[3] == "length")
    errors = sum(1 for r in results if r[5])
    return correct, total, truncated, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", type=int, nargs="+", default=[30000 + i for i in range(8)])
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-workers", type=int, default=256)
    parser.add_argument("--no-circular", dest="circular", action="store_false", default=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    print(f"MathBench: circular={args.circular}, max_tokens={args.max_tokens}, {len(args.ports)} servers")

    total_correct = total_total = 0
    results_by_split = {}
    task_offset = 0
    t0 = time.time()

    for split in SINGLE_CHOICE_SPLITS:
        for lang in ["cn", "en"]:
            fpath = os.path.join(DATA_ROOT, split, f"single_choice_{lang}.jsonl")
            if not os.path.exists(fpath):
                continue
            is_cn = (lang == "cn")
            c, t, trunc, err = eval_file(fpath, "single", is_cn, args.ports, args, task_offset)
            task_offset += t * (4 if args.circular else 1)
            acc = c / t * 100 if t > 0 else 0
            results_by_split[f"{split}/single_choice_{lang}"] = {"acc": acc, "correct": c, "total": t}
            circ_tag = "(circ)" if args.circular else ""
            print(f"  {split:22s} {lang}  {c:3d}/{t:3d} = {acc:5.1f}% {circ_tag}  trunc={trunc} err={err}")
            total_correct += c
            total_total += t

    for split in CLOZE_SPLITS:
        for lang in ["cn", "en"]:
            fpath = os.path.join(DATA_ROOT, split, f"cloze_{lang}.jsonl")
            if not os.path.exists(fpath):
                continue
            is_cn = (lang == "cn")
            c, t, trunc, err = eval_file(fpath, "cloze", is_cn, args.ports, args, task_offset)
            task_offset += t
            acc = c / t * 100 if t > 0 else 0
            results_by_split[f"{split}/cloze_{lang}"] = {"acc": acc, "correct": c, "total": t}
            print(f"  {split:22s} {lang}  {c:3d}/{t:3d} = {acc:5.1f}%         trunc={trunc} err={err}")
            total_correct += c
            total_total += t

    elapsed = time.time() - t0
    overall = total_correct / total_total * 100 if total_total > 0 else 0
    print(f"\n{'='*60}")
    print(f"MathBench Overall: {total_correct}/{total_total} = {overall:.1f}%")
    print(f"Wall time: {elapsed:.1f}s")
    print(f"{'='*60}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "mathbench_summary.json"), "w") as f:
            json.dump({"overall_acc": overall, "correct": total_correct,
                       "total": total_total, "splits": results_by_split}, f, indent=2)
        print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
