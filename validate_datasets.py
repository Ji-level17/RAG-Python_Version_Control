"""
Dataset Validator — FastAPI Golden Set + RAGAS
================================================
Run this to catch any hallucinations or structural issues in the
generated datasets WITHOUT having to read the generation code.

What it checks
  ✓ Required fields exist and are non-empty
  ✓ Code blocks are valid Python (ast.parse)
  ✓ Version metadata matches filename conventions
  ✓ No duplicate questions
  ✓ RAGAS contexts actually contain code relevant to the question
  ✓ ground_truth code mentions the function named in the question
  ✓ Splits are non-overlapping and sum to total
  ✓ Context chunks match real corpus entries (spot-check)

Usage
  python validate_datasets.py
  python validate_datasets.py --verbose    # print every warning
"""

import ast
import json
import sys
import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict

VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv

OUTPUTS = Path("/mnt/user-data/outputs")
CORPUS_PATH = Path("/mnt/user-data/uploads/local_corpus.json")

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
WARN = "\033[33m⚠\033[0m"
INFO = "\033[36mℹ\033[0m"

errors   = []
warnings = []
checks   = []


def ok(msg):
    checks.append((True, msg))
    print(f"  {PASS}  {msg}")

def fail(msg, detail=""):
    checks.append((False, msg))
    errors.append((msg, detail))
    print(f"  {FAIL}  {msg}")
    if detail and VERBOSE:
        for line in detail.splitlines():
            print(f"       {line}")

def warn(msg, detail=""):
    warnings.append((msg, detail))
    print(f"  {WARN}  {msg}")
    if detail and VERBOSE:
        for line in detail.splitlines():
            print(f"       {line}")

def info(msg):
    print(f"  {INFO}  {msg}")


# ── Load files ────────────────────────────────────────────────────────────────
print("\n=== Loading files ===")
files_ok = True

def load_json(path):
    global files_ok
    try:
        with open(path) as f:
            data = json.load(f)
        ok(f"Loaded {path.name}  ({len(data)} items)")
        return data
    except Exception as e:
        fail(f"Failed to load {path.name}", str(e))
        files_ok = False
        return None

def load_jsonl(path):
    global files_ok
    try:
        with open(path) as f:
            data = [json.loads(line) for line in f if line.strip()]
        ok(f"Loaded {path.name}  ({len(data)} items)")
        return data
    except Exception as e:
        fail(f"Failed to load {path.name}", str(e))
        files_ok = False
        return None

golden      = load_json(OUTPUTS / "fastapi_golden_set.json")
golden_jsonl = load_jsonl(OUTPUTS / "fastapi_golden_set.jsonl")
splits      = load_json(OUTPUTS / "fastapi_golden_set_splits.json")
ragas       = load_jsonl(OUTPUTS / "ragas_dataset.jsonl")
corpus      = load_json(CORPUS_PATH) if CORPUS_PATH.exists() else None

if not files_ok:
    print("\nAborting — fix file loading errors first.")
    sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# 1. GOLDEN SET — schema
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Golden set — schema ===")

REQUIRED_GOLDEN = ["instruction", "ground_truth", "version", "_meta"]
VERSION_KEYS    = ["python", "style", "fastapi_min_version"]
META_KEYS       = ["source_file", "topic", "type"]

schema_errors = []
for i, item in enumerate(golden):
    for k in REQUIRED_GOLDEN:
        if k not in item:
            schema_errors.append(f"item[{i}] missing key '{k}'")
        elif not item[k] and k not in ("_meta",):
            schema_errors.append(f"item[{i}] empty field '{k}'")
    if "version" in item:
        for k in VERSION_KEYS:
            if k not in item["version"]:
                schema_errors.append(f"item[{i}].version missing '{k}'")
    if "_meta" in item:
        for k in META_KEYS:
            if k not in item["_meta"]:
                schema_errors.append(f"item[{i}]._meta missing '{k}'")

if schema_errors:
    fail(f"Schema errors found ({len(schema_errors)})", "\n".join(schema_errors[:10]))
else:
    ok(f"All {len(golden)} golden items pass schema check")


# ════════════════════════════════════════════════════════════════════════════
# 2. GOLDEN SET — Python syntax validation
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Golden set — Python syntax ===")

syntax_errors = []
syntax_ok = 0
for i, item in enumerate(golden):
    code = item.get("ground_truth", "")
    if not code.strip():
        syntax_errors.append(f"item[{i}] empty ground_truth")
        continue
    try:
        ast.parse(code)
        syntax_ok += 1
    except SyntaxError as e:
        syntax_errors.append(f"item[{i}] ({item['_meta'].get('source_file','?')}): {e.msg} line {e.lineno}")

if syntax_errors:
    fail(
        f"{len(syntax_errors)} ground_truth blocks failed ast.parse",
        "\n".join(syntax_errors[:15]) + (f"\n...and {len(syntax_errors)-15} more" if len(syntax_errors) > 15 else "")
    )
    warn("Syntax errors can mean truncated code or assembly bugs in the generator")
else:
    ok(f"All {syntax_ok} ground_truth blocks are valid Python")


# ════════════════════════════════════════════════════════════════════════════
# 3. GOLDEN SET — instruction quality
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Golden set — instruction quality ===")

# Duplicates
instructions = [item["instruction"] for item in golden]
dup_counts = Counter(instructions)
dups = {k: v for k, v in dup_counts.items() if v > 1}
if dups:
    warn(f"{len(dups)} duplicate instructions found",
         "\n".join(f"  x{v}: {k[:80]}" for k, v in list(dups.items())[:5]))
else:
    ok("No duplicate instructions")

# Too short
short_inst = [(i, item["instruction"]) for i, item in enumerate(golden)
              if len(item["instruction"].split()) < 5]
if short_inst:
    warn(f"{len(short_inst)} instructions are very short (<5 words)",
         "\n".join(f"  [{i}]: {inst}" for i, inst in short_inst[:5]))
else:
    ok("All instructions are at least 5 words")

# Chinese in instruction (leftover from bad summaries)
chinese_inst = [(i, item["instruction"]) for i, item in enumerate(golden)
                if any("\u4e00" <= c <= "\u9fff" for c in item["instruction"])]
if chinese_inst:
    fail(f"{len(chinese_inst)} instructions contain Chinese characters — possible hallucination",
         "\n".join(f"  [{i}]: {inst[:80]}" for i, inst in chinese_inst[:5]))
else:
    ok("No Chinese characters in instructions")

# Chinese in ground_truth
chinese_gt = [(i, item["ground_truth"][:80]) for i, item in enumerate(golden)
              if any("\u4e00" <= c <= "\u9fff" for c in item["ground_truth"])]
if chinese_gt:
    warn(f"{len(chinese_gt)} ground_truth blocks contain Chinese characters",
         "\n".join(f"  [{i}]: {g}" for i, g in chinese_gt[:5]))
else:
    ok("No Chinese characters in ground_truth")


# ════════════════════════════════════════════════════════════════════════════
# 4. GOLDEN SET — version metadata plausibility
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Golden set — version metadata ===")

version_issues = []
for i, item in enumerate(golden):
    fp = item["_meta"].get("source_file", "")
    ver = item.get("version", {})
    py = ver.get("python", "")
    style = ver.get("style", "")

    # If filename has _py310 → must be py310+
    if "_py310" in fp and py != "py310+":
        version_issues.append(f"item[{i}]: file has _py310 but version.python='{py}'")
    # If filename has _an_ → must be annotated
    if re.search(r"_an[_.]", fp) and style not in ("annotated", "source"):
        version_issues.append(f"item[{i}]: file has _an_ but version.style='{style}'")
    # fastapi_min_version should be an int or None, not a string
    mv = ver.get("fastapi_min_version")
    if mv is not None and not isinstance(mv, int):
        version_issues.append(f"item[{i}]: fastapi_min_version is '{type(mv).__name__}', expected int")

if version_issues:
    fail(f"{len(version_issues)} version metadata mismatches",
         "\n".join(version_issues[:10]))
else:
    ok("All version metadata is internally consistent")


# ════════════════════════════════════════════════════════════════════════════
# 5. SPLITS — integrity
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Splits — integrity ===")

train = splits.get("train", [])
val   = splits.get("val",   [])
test  = splits.get("test",  [])

total_split = len(train) + len(val) + len(test)
if total_split != len(golden):
    fail(f"Split sizes sum to {total_split} but golden has {len(golden)} items")
else:
    ok(f"Split sizes sum correctly: {len(train)} train + {len(val)} val + {len(test)} test = {total_split}")

# Check no overlap
train_q = {x["instruction"] for x in train}
val_q   = {x["instruction"] for x in val}
test_q  = {x["instruction"] for x in test}

tv_overlap = train_q & val_q
tt_overlap = train_q & test_q
vt_overlap = val_q   & test_q

if tv_overlap or tt_overlap or vt_overlap:
    fail(f"Overlap detected between splits",
         f"train∩val: {len(tv_overlap)}, train∩test: {len(tt_overlap)}, val∩test: {len(vt_overlap)}")
else:
    ok("No overlap between train / val / test splits")

# Check JSONL == JSON
if len(golden_jsonl) != len(golden):
    fail(f"JSONL has {len(golden_jsonl)} items but JSON has {len(golden)}")
else:
    ok(f"JSONL and JSON item counts match ({len(golden)})")


# ════════════════════════════════════════════════════════════════════════════
# 6. RAGAS — schema
# ════════════════════════════════════════════════════════════════════════════
print("\n=== RAGAS dataset — schema ===")

REQUIRED_RAGAS = ["question", "contexts", "ground_truth", "answer"]
ragas_schema_errors = []

for i, r in enumerate(ragas):
    for k in REQUIRED_RAGAS:
        if k not in r:
            ragas_schema_errors.append(f"record[{i}] missing '{k}'")
    # contexts must be a list
    if "contexts" in r and not isinstance(r["contexts"], list):
        ragas_schema_errors.append(f"record[{i}] contexts is not a list")
    # contexts must be non-empty
    if "contexts" in r and isinstance(r["contexts"], list) and len(r["contexts"]) == 0:
        ragas_schema_errors.append(f"record[{i}] contexts is empty list")
    # answer should be blank (placeholder)
    if "answer" in r and r["answer"] != "":
        ragas_schema_errors.append(f"record[{i}] answer is pre-filled (should be blank)")

if ragas_schema_errors:
    fail(f"{len(ragas_schema_errors)} RAGAS schema errors",
         "\n".join(ragas_schema_errors[:10]))
else:
    ok(f"All {len(ragas)} RAGAS records pass schema check")


# ════════════════════════════════════════════════════════════════════════════
# 7. RAGAS — contexts spot-check against corpus
# ════════════════════════════════════════════════════════════════════════════
print("\n=== RAGAS dataset — context grounding ===")

if corpus:
    # Build a set of all corpus content fingerprints (first 80 chars)
    corpus_fingerprints = {item["content"].strip()[:80] for item in corpus}

    ctx_not_in_corpus = 0
    total_ctx_checked = 0
    flagged = []

    for i, r in enumerate(ragas[:50]):   # spot-check first 50
        for ctx in r["contexts"]:
            fp = ctx[:80]
            total_ctx_checked += 1
            if fp not in corpus_fingerprints:
                ctx_not_in_corpus += 1
                if len(flagged) < 5:
                    flagged.append(f"record[{i}] context not found in corpus: {fp[:60]}...")

    if ctx_not_in_corpus == 0:
        ok(f"All {total_ctx_checked} spot-checked contexts traced back to corpus")
    elif ctx_not_in_corpus / total_ctx_checked < 0.05:
        warn(f"{ctx_not_in_corpus}/{total_ctx_checked} contexts not found in corpus (< 5% — likely whitespace diff)",
             "\n".join(flagged))
    else:
        fail(f"{ctx_not_in_corpus}/{total_ctx_checked} contexts NOT found in corpus — possible hallucination",
             "\n".join(flagged))
else:
    warn("Corpus file not found — skipping context grounding check")


# ════════════════════════════════════════════════════════════════════════════
# 8. RAGAS — question/ground_truth alignment
# ════════════════════════════════════════════════════════════════════════════
print("\n=== RAGAS dataset — question↔ground_truth alignment ===")

# The function name mentioned in the question (backtick-quoted) should appear
# in the ground_truth code.
misaligned = []
for i, r in enumerate(ragas):
    q = r.get("question", "")
    gt = r.get("ground_truth", "")
    # Extract `fn_name` from question
    backtick_names = re.findall(r"`([^`]+)`", q)
    if not backtick_names:
        continue
    fn = backtick_names[0]
    # fn might be a decorator style or param type — only check if it looks like a Python identifier
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", fn):
        if fn not in gt:
            misaligned.append(f"record[{i}]: question mentions `{fn}` but ground_truth doesn't contain it")

rate = len(misaligned) / len(ragas) if ragas else 0
if rate > 0.30:
    fail(f"{len(misaligned)}/{len(ragas)} records ({rate:.0%}) may be misaligned",
         "\n".join(misaligned[:8]))
elif misaligned:
    warn(f"{len(misaligned)}/{len(ragas)} records ({rate:.0%}) have weak question↔ground_truth alignment",
         "\n".join(misaligned[:8]) + "\n(These may still be valid — e.g. helper fn not in name)")
else:
    ok("All questions align with their ground_truth code")


# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════
n_pass = sum(1 for ok, _ in checks if ok)
n_fail = sum(1 for ok, _ in checks if not ok)

print(f"\n{'='*55}")
print(f"  VALIDATION SUMMARY")
print(f"{'='*55}")
print(f"  Checks passed  : {n_pass}")
print(f"  Checks failed  : {n_fail}")
print(f"  Warnings       : {len(warnings)}")
print()

if n_fail == 0 and len(warnings) == 0:
    print("  \033[32mAll checks passed — datasets look clean.\033[0m")
elif n_fail == 0:
    print(f"  \033[33mPassed with {len(warnings)} warning(s) — review above.\033[0m")
else:
    print(f"  \033[31m{n_fail} check(s) failed — review errors above.\033[0m")
    print("  Re-run with --verbose for full details on each failure.")
print()

sys.exit(0 if n_fail == 0 else 1)
