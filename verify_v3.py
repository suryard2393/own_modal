"""
verify_v3.py
============
Checks that the v3 training pipeline is correctly wired before the first
training run.  Does NOT train, download data, or overwrite any files.

Run from the project root:
    python verify_v3.py

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed.
"""

import sys, os, ast, importlib, hashlib, json, types

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

results = []

def ok(name):
    results.append(("PASS", name, ""))
    print("  PASS  " + name)

def fail(name, reason):
    results.append(("FAIL", name, reason))
    print("  FAIL  " + name)
    print("        " + str(reason))

def section(title):
    print("\n=== " + title + " ===")


# ── 1. Imports ─────────────────────────────────────────────────────────────────
section("1. Imports")

for mod_name in ("model", "train_v3", "chat", "calculator", "rag"):
    try:
        mod = importlib.import_module(mod_name)
        ok("import " + mod_name)
    except Exception as e:
        fail("import " + mod_name, e)


# ── 2. Protected-path constants in train_v3 ────────────────────────────────────
section("2. train_v3 path constants")

import train_v3

v3_tok  = os.path.abspath(train_v3.TOKENIZER_V3_PATH)
v3_ckpt = os.path.abspath(train_v3.CHECKPOINT_V3_PATH)
protected = {os.path.abspath(p) for p in train_v3._PROTECTED}

if "my_bpe_tokenizer_v3.json" in v3_tok:
    ok("TOKENIZER_V3_PATH ends with my_bpe_tokenizer_v3.json")
else:
    fail("TOKENIZER_V3_PATH", "expected ...my_bpe_tokenizer_v3.json, got " + v3_tok)

if "my_minigpt_v3.pth" in v3_ckpt:
    ok("CHECKPOINT_V3_PATH ends with my_minigpt_v3.pth")
else:
    fail("CHECKPOINT_V3_PATH", "expected ...my_minigpt_v3.pth, got " + v3_ckpt)

for name, path in [
    ("my_bpe_tokenizer.json",   os.path.join(ROOT, "tokenizers",  "my_bpe_tokenizer.json")),
    ("my_minigpt_v2.pth",       os.path.join(ROOT, "checkpoints", "my_minigpt_v2.pth")),
    ("my_minigpt_fineweb.pth",  os.path.join(ROOT, "checkpoints", "my_minigpt_fineweb.pth")),
    ("my_minigpt.pth",          os.path.join(ROOT, "checkpoints", "my_minigpt.pth")),
]:
    if os.path.abspath(path) in protected:
        ok(name + " is in _PROTECTED set")
    else:
        fail(name + " protection", name + " is NOT in _PROTECTED — could be overwritten")

if v3_tok not in protected and v3_ckpt not in protected:
    ok("v3 output paths are NOT in _PROTECTED (correct — they are the new files)")
else:
    fail("v3 output paths", "v3 paths are in _PROTECTED — they would block their own write")


# ── 3. train_v3 AST checks (no module-level side effects) ─────────────────────
section("3. train_v3 static analysis")

with open(os.path.join(ROOT, "src", "train_v3.py")) as f:
    src = f.read()
tree = ast.parse(src)

top_prints = [
    n for n in ast.walk(tree)
    if isinstance(n, ast.Expr)
    and isinstance(n.value, ast.Call)
    and isinstance(getattr(n.value.func, "id", None), str)
    and n.value.func.id == "print"
    and n.col_offset == 0
]
if not top_prints:
    ok("No module-level print() calls")
else:
    fail("Module-level print()", "found at lines " + str([n.lineno for n in top_prints]))

fn_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
for expected in ("load_corpus", "build_tokenizer", "encode_and_split",
                 "get_batch", "estimate_loss", "save_checkpoint",
                 "_confirm", "_assert_not_protected", "train"):
    if expected in fn_names:
        ok("function " + expected + "() defined")
    else:
        fail("function " + expected + "()", "not found in train_v3.py")

# Confirm tokenizer is saved BEFORE model is constructed
tok_save_line  = next((n.lineno for n in ast.walk(tree)
                       if isinstance(n, ast.Call)
                       and isinstance(getattr(n.func, "attr", None), str)
                       and n.func.attr == "save"), None)
model_init_line = next((n.lineno for n in ast.walk(tree)
                        if isinstance(n, ast.Call)
                        and isinstance(getattr(n.func, "id", None), str)
                        and n.func.id == "MiniGPTv2"), None)
if tok_save_line and model_init_line and tok_save_line < model_init_line:
    ok("Tokenizer saved before MiniGPTv2 constructed (line "
       + str(tok_save_line) + " < " + str(model_init_line) + ")")
else:
    fail("Tokenizer-before-model order",
         "tok_save=" + str(tok_save_line) + " model_init=" + str(model_init_line))

# Confirm train/val split is present
if "TRAIN_SPLIT" in src and "encode_and_split" in src:
    ok("Train/val split via encode_and_split() and TRAIN_SPLIT constant")
else:
    fail("Train/val split", "TRAIN_SPLIT or encode_and_split not found")

# Confirm tokenizer_file is stored in checkpoint metadata
if '"tokenizer_file"' in src:
    ok("tokenizer_file stored in checkpoint metadata")
else:
    fail("tokenizer_file metadata", "key not found in save_checkpoint payload")

# Confirm optimizer state is NOT claimed to be saved
if "optimizer_state_dict" not in src:
    ok("Optimizer state NOT saved (resumption is weights-only as documented)")
else:
    fail("Optimizer state", "optimizer_state_dict found — update docstring if intentional")

# Confirm post-save verification exists
if "POST-SAVE MISMATCH" in src or ("saved[" in src and "tok2" in src):
    ok("Post-save vocab verification present")
else:
    fail("Post-save verification", "verification block not found")


# ── 4. chat.py checks ─────────────────────────────────────────────────────────
section("4. chat.py checks")

with open(os.path.join(ROOT, "src", "chat.py")) as f:
    chat_src = f.read()

import chat

v3_ckpt_in_list = any("my_minigpt_v3.pth" in p for p in chat.CHECKPOINTS)
v2_ckpt_in_list = any("my_minigpt_v2.pth" in p for p in chat.CHECKPOINTS)
v3_first = chat.CHECKPOINTS[0].endswith("my_minigpt_v3.pth")

if v3_ckpt_in_list:
    ok("my_minigpt_v3.pth in chat.CHECKPOINTS")
else:
    fail("chat.CHECKPOINTS", "my_minigpt_v3.pth not listed")

if v3_first:
    ok("my_minigpt_v3.pth is first in CHECKPOINTS (highest priority)")
else:
    fail("CHECKPOINTS order", "v3 is not first — v2 would be preferred")

if v2_ckpt_in_list:
    ok("my_minigpt_v2.pth retained as fallback")
else:
    fail("chat.CHECKPOINTS fallback", "my_minigpt_v2.pth not listed as fallback")

# v1 files must not appear
v1_in_list = any("fineweb" in p or "my_minigpt.pth" in p for p in chat.CHECKPOINTS)
if not v1_in_list:
    ok("No v1 checkpoints in CHECKPOINTS list")
else:
    fail("v1 checkpoint guard", "v1 checkpoint found in CHECKPOINTS list")

# load_model must return 3 values
import inspect
sig = inspect.signature(chat.load_model)
src_load = inspect.getsource(chat.load_model)
if "tok_file" in src_load and "return model, ckpt_vocab, tok_file" in src_load:
    ok("load_model() returns (model, ckpt_vocab, tok_file)")
else:
    fail("load_model() signature", "does not return tok_file")

# load_tokenizer must accept tokenizer_file param
sig2 = inspect.signature(chat.load_tokenizer)
if "tokenizer_file" in sig2.parameters:
    ok("load_tokenizer() accepts tokenizer_file parameter")
else:
    fail("load_tokenizer() signature", "tokenizer_file parameter missing")

# main() must unpack 3 values from load_model
if "model, ckpt_vocab, tok_file = load_model()" in chat_src:
    ok("main() unpacks 3 values from load_model()")
else:
    fail("main() unpacking", "does not unpack tok_file from load_model()")


# ── 5. Existing file integrity ─────────────────────────────────────────────────
section("5. Existing file integrity (no files overwritten)")

def sha16(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]

existing = {
    "checkpoints/my_minigpt_v2.pth":      "a2343d7d188f14b5",
    "checkpoints/my_minigpt_fineweb.pth": "9571915dfd07bc16",
    "checkpoints/my_minigpt.pth":         "d9fc8f1e1d25e3c6",
    "tokenizers/my_bpe_tokenizer.json":   "d14fa1af7a7be48e",
}
for rel, expected_hash in existing.items():
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        fail(rel, "FILE MISSING")
        continue
    got = sha16(path)
    if got == expected_hash:
        ok(rel + " unchanged (" + got + ")")
    else:
        fail(rel + " integrity", "hash changed: expected " + expected_hash + ", got " + got)

# v3 files: report presence/absence as info, not a failure
for rel in ("checkpoints/my_minigpt_v3.pth", "tokenizers/my_bpe_tokenizer_v3.json"):
    path = os.path.join(ROOT, rel)
    if os.path.exists(path):
        ok(rel + " exists (v3 already trained)")
    else:
        ok(rel + " does not exist yet (not trained yet)")


# ── 6. calculator and rag unchanged ───────────────────────────────────────────
section("6. Calculator and RAG smoke test")

from calculator import calculate, calculate_simple
from rag import retrieve

calc_cases = [("2 + 2", "4"), ("10 divided by 2", "5"), ("what is python", None)]
for q, expected in calc_cases:
    got = calculate(q)
    if got == expected:
        ok("calculate(" + repr(q) + ") == " + repr(expected))
    else:
        fail("calculate(" + repr(q) + ")", "expected " + repr(expected) + " got " + repr(got))

rag_cases = [
    ("What is Python?", "Python is a programming language known for its readable syntax."),
    ("who is tramp?",   "Donald Trump is an American politician and businessman who has served as president of the United States."),
    ("what is your name?", "I am MiniGPT, a small language model built as a local AI project. I am not a commercial product and I do not have a human identity."),
]
for q, expected in rag_cases:
    r = retrieve(q)
    if r["answer"] == expected:
        ok("retrieve(" + repr(q) + ") correct")
    else:
        fail("retrieve(" + repr(q) + ")", "got: " + repr(r["answer"]))


# ── 7. requirements.txt ────────────────────────────────────────────────────────
section("7. requirements.txt")

with open(os.path.join(ROOT, "requirements.txt")) as f:
    reqs = f.read()

if "numpy<2.0.0" in reqs:
    fail("requirements.txt numpy pin", "numpy<2.0.0 conflicts with installed 2.4.6")
else:
    ok("No conflicting numpy<2.0.0 pin")

for pkg in ("torch", "tokenizers", "datasets", "scikit-learn", "numpy"):
    if pkg in reqs:
        ok(pkg + " listed in requirements.txt")
    else:
        fail("requirements.txt", pkg + " missing")


# ── Summary ───────────────────────────────────────────────────────────────────
section("Summary")
passed = sum(1 for r in results if r[0] == "PASS")
failed = sum(1 for r in results if r[0] == "FAIL")
print("PASSED: " + str(passed) + "   FAILED: " + str(failed))
if failed:
    print("\nFailed checks:")
    for r in results:
        if r[0] == "FAIL":
            print("  [FAIL] " + r[1] + ": " + r[2])

sys.exit(0 if failed == 0 else 1)
