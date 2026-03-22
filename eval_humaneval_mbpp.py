"""
HumanEval & MBPP Code Generation Benchmark
============================================
Evaluates code generation ability using pass@k on two standard benchmarks.

Benchmarks:
  1. HumanEval  — 164 problems (OpenAI, function completion)
  2. MBPP       — 257 problems (Google, sanitized split, full function generation)

Uses Gemini API for code generation. Executes generated code in sandboxed
subprocess with timeout to verify correctness.

Usage:
  python eval_humaneval_mbpp.py --benchmark humaneval --quick-n 20
  python eval_humaneval_mbpp.py --benchmark mbpp --quick-n 20
  python eval_humaneval_mbpp.py --benchmark both --quick-n 0   # full run
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Auto-load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gemini-key", default=os.getenv("GEMINI_API_KEY"))
    p.add_argument("--benchmark", default="both",
                   choices=["humaneval", "mbpp", "both"])
    p.add_argument("--gen-model", default="gemini-2.5-flash")
    p.add_argument("--quick-n", type=int, default=0, help="0 = full benchmark")
    p.add_argument("--num-samples", type=int, default=1,
                   help="Samples per problem for pass@k (k=1 uses 1 sample)")
    p.add_argument("--timeout", type=int, default=10,
                   help="Seconds per test execution")
    p.add_argument("--delay", type=float, default=1.0,
                   help="Seconds between API calls")
    p.add_argument("--output-dir", default=".")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not args.gemini_key:
        p.error("Gemini key required: set GEMINI_API_KEY in .env")
    return args


# ─────────────────────────────────────────────
# Gemini client
# ─────────────────────────────────────────────
def make_gemini_client(api_key):
    from google import genai as _genai

    class GeminiClient:
        def __init__(self, key):
            self._client = _genai.Client(api_key=key)

        def generate(self, model, prompt, max_tokens=2048, temperature=0):
            import threading
            from google import genai as _g
            result = [None]
            error = [None]
            def _call():
                try:
                    result[0] = self._client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=_g.types.GenerateContentConfig(
                            max_output_tokens=max_tokens,
                            temperature=temperature,
                        ),
                    )
                except Exception as e:
                    error[0] = e
            t = threading.Thread(target=_call)
            t.start()
            t.join(timeout=60)
            if t.is_alive():
                raise TimeoutError("Gemini API timeout")
            if error[0]:
                raise error[0]
            return result[0].text if result[0] and result[0].text else ""

    return GeminiClient(api_key)


# ─────────────────────────────────────────────
# Code generation
# ─────────────────────────────────────────────
def generate_humaneval(client, model, prompt, delay):
    """Generate complete function for HumanEval prompt."""
    full_prompt = (
        "Complete the following Python function. "
        "Return the COMPLETE function including the signature and body. "
        "Do NOT include the docstring — only the def line and the implementation. "
        "Do NOT include any explanation or markdown code fences. "
        "Return ONLY valid Python code.\n\n"
        f"{prompt}"
    )
    for attempt in range(3):
        time.sleep(delay)
        try:
            raw = client.generate(model, full_prompt)
            return _clean_code(raw)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(15 * (attempt + 1))
            else:
                print(f"    gen error: {e}")
                return "def f(): pass"
    return "def f(): pass"


def generate_mbpp(client, model, prompt, test_list, delay):
    """Generate a complete Python function for the MBPP task."""
    # Include test cases so the model knows the expected function name/signature
    test_examples = "\n".join(test_list[:3])
    full_prompt = (
        "Write a Python function that solves the following task. "
        "Return ONLY the Python function code. "
        "Do NOT include test cases, explanations, or markdown code fences.\n\n"
        f"Task: {prompt}\n\n"
        f"The function must pass these tests:\n{test_examples}"
    )
    for attempt in range(3):
        time.sleep(delay)
        try:
            raw = client.generate(model, full_prompt)
            return _clean_code(raw)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(15 * (attempt + 1))
            else:
                print(f"    gen error: {e}")
                return "pass"
    return "pass"


def _clean_completion(raw, prompt):
    """Clean HumanEval completion: remove markdown fences, ensure indentation."""
    raw = raw.strip()
    # Remove markdown fences
    if "```" in raw:
        lines = raw.split("\n")
        cleaned = []
        for line in lines:
            if line.strip().startswith("```"):
                continue
            cleaned.append(line)
        raw = "\n".join(cleaned).strip()

    # If the model repeated the full function, extract just the body
    lines = raw.split("\n")
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            body_start = i + 1
            # Skip docstring if present
            if body_start < len(lines):
                stripped = lines[body_start].strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    quote = stripped[:3]
                    if stripped.count(quote) >= 2:
                        body_start += 1  # single-line docstring
                    else:
                        for j in range(body_start + 1, len(lines)):
                            if quote in lines[j]:
                                body_start = j + 1
                                break
            break
    if body_start > 0:
        lines = lines[body_start:]
        raw = "\n".join(lines)

    # Normalize indentation: find the indentation of the first non-empty line
    # and ensure everything is at least 4-space indented (function body level)
    out_lines = []
    for line in raw.split("\n"):
        if not line.strip():
            out_lines.append("")
            continue
        # Strip any existing indentation, then add exactly 4 spaces
        stripped = line.lstrip()
        # Count original indent to preserve relative indentation
        orig_indent = len(line) - len(stripped)
        out_lines.append(line)

    # Check if first code line has proper indentation
    first_code = next((l for l in out_lines if l.strip()), "")
    if first_code and not first_code.startswith("    "):
        # No indentation at all — add 4 spaces to every non-empty line
        out_lines = [("    " + l if l.strip() else l) for l in out_lines]
    elif first_code and first_code.startswith("        "):
        # Too much indentation (8+ spaces) — likely double-indented
        # Dedent by finding minimum indent and normalizing to 4 spaces
        non_empty = [l for l in out_lines if l.strip()]
        if non_empty:
            min_indent = min(len(l) - len(l.lstrip()) for l in non_empty)
            if min_indent > 4:
                strip_n = min_indent - 4
                out_lines = [
                    l[strip_n:] if (l.strip() and len(l) - len(l.lstrip()) >= strip_n) else l
                    for l in out_lines
                ]

    return "\n".join(out_lines)


def _clean_code(raw):
    """Clean MBPP code: remove markdown fences."""
    raw = raw.strip()
    if "```" in raw:
        lines = raw.split("\n")
        cleaned = []
        inside = False
        for line in lines:
            if line.strip().startswith("```"):
                inside = not inside
                continue
            cleaned.append(line)
        raw = "\n".join(cleaned).strip()
    return raw


# ─────────────────────────────────────────────
# Execution sandbox
# ─────────────────────────────────────────────
def execute_code(code: str, timeout: int = 10) -> tuple[bool, str]:
    """Execute code in a subprocess, return (passed, error_msg)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                      delete=False, encoding="utf-8") as f:
        f.write(code)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return True, ""
        else:
            err = (result.stderr or result.stdout)[-500:]
            return False, err
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)[:200]
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ─────────────────────────────────────────────
# HumanEval evaluation
# ─────────────────────────────────────────────
def eval_humaneval(client, model, problems, delay, timeout, num_samples):
    """Run HumanEval: generate completion, combine with prompt+tests, execute."""
    results = []
    for prob in tqdm(problems, desc="HumanEval"):
        task_id = prob["task_id"]
        prompt = prob["prompt"]
        test_code = prob["test"]
        entry_point = prob["entry_point"]

        passed_any = False
        n_passed = 0
        for s in range(num_samples):
            completion = generate_humaneval(client, model, prompt, delay)

            # Model returns complete function — use it directly with tests
            # Extract imports from the prompt (e.g. "from typing import List")
            import_lines = [l for l in prompt.split("\n")
                            if l.strip().startswith(("from ", "import "))]
            imports = "\n".join(import_lines) + "\n" if import_lines else ""

            full_code = (
                f"{imports}"
                f"{completion}\n\n"
                f"{test_code}\n"
                f"check({entry_point})\n"
            )

            passed, err = execute_code(full_code, timeout)
            if passed:
                n_passed += 1
                passed_any = True

        results.append({
            "task_id": task_id,
            "n_samples": num_samples,
            "n_passed": n_passed,
            "passed": passed_any,
            "completion": completion[:500],  # save last sample for inspection
            "error": err[:200] if not passed_any else "",
        })
        status = "PASS" if passed_any else "FAIL"
        print(f"  {task_id}: {status}  ({n_passed}/{num_samples})")

    return results


# ─────────────────────────────────────────────
# MBPP evaluation
# ─────────────────────────────────────────────
def eval_mbpp(client, model, problems, delay, timeout, num_samples):
    """Run MBPP: generate function, combine with assertions, execute."""
    results = []
    for prob in tqdm(problems, desc="MBPP"):
        task_id = prob["task_id"]
        prompt_text = prob["prompt"]
        test_list = prob["test_list"]
        test_imports = prob.get("test_imports", [])

        passed_any = False
        n_passed = 0
        for s in range(num_samples):
            code = generate_mbpp(client, model, prompt_text, test_list, delay)

            # Assemble: imports + generated code + test assertions
            import_block = "\n".join(test_imports) + "\n" if test_imports else ""
            test_block = "\n".join(test_list)
            full_code = f"{import_block}{code}\n\n{test_block}\n"

            passed, err = execute_code(full_code, timeout)
            if passed:
                n_passed += 1
                passed_any = True

        results.append({
            "task_id": task_id,
            "n_samples": num_samples,
            "n_passed": n_passed,
            "passed": passed_any,
            "code": code[:500],
            "error": err[:200] if not passed_any else "",
        })
        status = "PASS" if passed_any else "FAIL"
        print(f"  MBPP-{task_id}: {status}  ({n_passed}/{num_samples})")

    return results


# ─────────────────────────────────────────────
# pass@k estimator (unbiased, from Chen et al. 2021)
# ─────────────────────────────────────────────
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator: 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(range(n - c, n - c - k, -1)) / math.prod(range(n, n - k, -1))


def compute_pass_at_k(results, k_values=(1,)):
    """Compute pass@k for each k from per-problem results."""
    scores = {}
    for k in k_values:
        per_problem = []
        for r in results:
            n = r["n_samples"]
            c = r["n_passed"]
            if n >= k:
                per_problem.append(pass_at_k(n, c, k))
        scores[f"pass@{k}"] = sum(per_problem) / len(per_problem) if per_problem else 0.0
    return scores


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    import random
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = make_gemini_client(args.gemini_key)
    print(f"Model: {args.gen_model}")
    print(f"Samples per problem: {args.num_samples}")

    random.seed(args.seed)

    benchmarks = (
        ["humaneval", "mbpp"] if args.benchmark == "both"
        else [args.benchmark]
    )

    all_summary = {}

    for bench in benchmarks:
        print(f"\n{'='*60}")
        print(f"  Benchmark: {bench.upper()}")
        print(f"{'='*60}")

        if bench == "humaneval":
            from datasets import load_dataset
            ds = load_dataset("openai/openai_humaneval", split="test")
            problems = list(ds)
            if args.quick_n > 0:
                problems = random.sample(problems, min(args.quick_n, len(problems)))
            print(f"  Problems: {len(problems)}")

            results = eval_humaneval(
                client, args.gen_model, problems,
                args.delay, args.timeout, args.num_samples,
            )

        else:  # mbpp
            from datasets import load_dataset
            ds = load_dataset("google-research-datasets/mbpp", "sanitized",
                              split="test")
            problems = list(ds)
            if args.quick_n > 0:
                problems = random.sample(problems, min(args.quick_n, len(problems)))
            print(f"  Problems: {len(problems)}")

            results = eval_mbpp(
                client, args.gen_model, problems,
                args.delay, args.timeout, args.num_samples,
            )

        # Compute pass@k
        k_values = [1] if args.num_samples == 1 else [1, 5, 10]
        k_values = [k for k in k_values if k <= args.num_samples]
        scores = compute_pass_at_k(results, k_values)

        n_passed = sum(1 for r in results if r["passed"])
        n_total = len(results)
        print(f"\n  {bench.upper()} Results:")
        print(f"  Passed: {n_passed}/{n_total} ({n_passed/n_total*100:.1f}%)")
        for metric, val in scores.items():
            print(f"  {metric}: {val:.3f}")

        all_summary[bench] = {
            "n_problems": n_total,
            "n_passed": n_passed,
            **scores,
        }

        # Save per-problem results
        df = pd.DataFrame(results)
        csv_path = output_dir / f"{bench}_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    # Save summary
    summary_df = pd.DataFrame(all_summary).T
    summary_path = output_dir / "code_benchmark_summary.csv"
    summary_df.to_csv(summary_path)
    print(f"\n{'='*60}")
    print(summary_df.to_string())
    print(f"{'='*60}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
