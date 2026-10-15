"""
train_v3.py  —  MiniGPT v3 training pipeline
=============================================
Run from the project root:

    python src/train_v3.py

What this script does (in order):
  1. Guards both output files before any work starts.
  2. Streams and cleans a FineWeb-Edu corpus sample.
  3. Trains a BPE tokenizer on that corpus and saves it to
       tokenizers/my_bpe_tokenizer_v3.json
  4. Encodes the corpus, splits 90/10 into train and validation sets.
  5. Trains MiniGPTv2 from scratch (or resumes weights-only from a v3
     checkpoint if one already exists and the user confirms).
  6. Saves a matched checkpoint to
       checkpoints/my_minigpt_v3.pth
     The checkpoint metadata includes vocab_size, tokenizer_file, and all
     architecture hyperparameters so the pair can always be verified.
  7. Verifies that checkpoint vocab_size == tokenizer vocab_size before exit.

Resumption note
---------------
If a v3 checkpoint already exists, this script offers to resume from it
(weights only — optimizer state is NOT saved or restored). Training will
continue from step 0 with the saved weights as the starting point.
Full optimizer-state resumption is not implemented.

Files written
-------------
  tokenizers/my_bpe_tokenizer_v3.json   (new — never overwrites v1/v2 files)
  checkpoints/my_minigpt_v3.pth         (new — never overwrites v1/v2 files)
"""

import os
import re
import sys
import torch
from itertools import islice
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# ── path setup ─────────────────────────────────────────────────────────────────
# Works whether invoked as `python src/train_v3.py` or `python -m src.train_v3`
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)   # so `from model import ...` resolves

from model import MiniGPTv2

# ── v3-specific output paths (never touch v1/v2 files) ────────────────────────
TOKENIZER_V3_PATH  = os.path.join(ROOT, "tokenizers",  "my_bpe_tokenizer_v3.json")
CHECKPOINT_V3_PATH = os.path.join(ROOT, "checkpoints", "my_minigpt_v3.pth")

# Paths that must NEVER be written by this script
_PROTECTED = {
    os.path.join(ROOT, "tokenizers",  "my_bpe_tokenizer.json"),
    os.path.join(ROOT, "checkpoints", "my_minigpt_v2.pth"),
    os.path.join(ROOT, "checkpoints", "my_minigpt_fineweb.pth"),
    os.path.join(ROOT, "checkpoints", "my_minigpt.pth"),
}

# ── hyperparameters ────────────────────────────────────────────────────────────
MAX_DOCUMENTS    = 3000
MAX_CHARS_PER_DOC = 4000
VOCAB_SIZE_TARGET = 2000    # BPE trainer target; actual size may differ slightly
BLOCK_SIZE        = 256
BATCH_SIZE        = 32
N_EMBD            = 128
N_HEAD            = 4
N_LAYER           = 4
DROPOUT           = 0.1
MAX_ITERS         = 3000
EVAL_INTERVAL       = 250
CHECKPOINT_INTERVAL = 500   # save weights every N steps; end-of-training always saves too
LR                  = 3e-4
TRAIN_SPLIT         = 0.9   # fraction used for training; remainder is validation

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── safety helpers ─────────────────────────────────────────────────────────────

def _assert_not_protected(path):
    """Hard stop if we are about to write a protected file."""
    if os.path.abspath(path) in {os.path.abspath(p) for p in _PROTECTED}:
        sys.exit(f"SAFETY ERROR: attempted to write protected file:\n  {path}\nAborted.")


def _confirm(prompt):
    """Return True only when the user explicitly types 'y'."""
    return input(prompt + " [y/N]: ").strip().lower() == "y"


# ── corpus ─────────────────────────────────────────────────────────────────────

def load_corpus():
    """Stream and clean up to MAX_DOCUMENTS documents from FineWeb-Edu."""
    from datasets import load_dataset

    print("Streaming FineWeb-Edu corpus ...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    ).shuffle(seed=42, buffer_size=10_000)

    documents, seen = [], set()
    for example in islice(dataset, MAX_DOCUMENTS):
        raw = example.get("text", "")
        if not isinstance(raw, str):
            continue
        cleaned = raw.encode("ascii", errors="ignore").decode("ascii")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()[:MAX_CHARS_PER_DOC]
        if len(cleaned) < 100 or cleaned in seen:
            continue
        seen.add(cleaned)
        documents.append(cleaned)

    print(f"  Documents loaded : {len(documents)}")
    print(f"  Total characters : {sum(map(len, documents))}")
    return "\n\n".join(documents)


# ── tokenizer ──────────────────────────────────────────────────────────────────

def build_tokenizer(text):
    """Train a BPE tokenizer on `text` and save to TOKENIZER_V3_PATH."""
    _assert_not_protected(TOKENIZER_V3_PATH)

    print(f"\nTraining BPE tokenizer (target vocab={VOCAB_SIZE_TARGET}) ...")
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder       = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE_TARGET,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    )
    tok.train_from_iterator(
        (text[i : i + 100_000] for i in range(0, len(text), 100_000)),
        trainer=trainer,
        length=(len(text) + 99_999) // 100_000,
    )

    tok.save(TOKENIZER_V3_PATH)
    print(f"  Tokenizer saved  : {TOKENIZER_V3_PATH}")
    print(f"  Actual vocab size: {tok.get_vocab_size()}")
    return tok


# ── data split ─────────────────────────────────────────────────────────────────

def encode_and_split(text, tokenizer):
    """Encode corpus with tokenizer and split into train / validation tensors."""
    print("\nEncoding corpus ...")
    all_ids = tokenizer.encode(text).ids
    data    = torch.tensor(all_ids, dtype=torch.long)

    split_idx  = int(TRAIN_SPLIT * len(data))
    train_data = data[:split_idx]
    val_data   = data[split_idx:]

    print(f"  Total tokens     : {len(data)}")
    print(f"  Training tokens  : {len(train_data)}  ({TRAIN_SPLIT*100:.0f}%)")
    print(f"  Validation tokens: {len(val_data)}  ({(1-TRAIN_SPLIT)*100:.0f}%)")

    assert len(train_data) > BLOCK_SIZE + 1, "Training set too small for block_size"
    assert len(val_data)   > BLOCK_SIZE + 1, "Validation set too small for block_size"
    return train_data, val_data


# ── batching ───────────────────────────────────────────────────────────────────

def get_batch(split, train_data, val_data):
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x  = torch.stack([source[i     : i + BLOCK_SIZE    ] for i in ix])
    y  = torch.stack([source[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)


# ── loss estimation ────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model, train_data, val_data, n_batches=20):
    model.eval()
    results = {}
    for split in ("train", "val"):
        losses = [
            model(*get_batch(split, train_data, val_data))[1].item()
            for _ in range(n_batches)
        ]
        results[split] = sum(losses) / len(losses)
    model.train()
    return results


# ── checkpoint save ────────────────────────────────────────────────────────────

def save_checkpoint(model, vocab_size, step):
    """
    Save model weights plus all metadata needed to reconstruct the model
    and verify tokenizer compatibility.

    Metadata stored:
      version        : "v3"
      vocab_size     : int   — must equal tokenizer.get_vocab_size()
      tokenizer_file : str   — basename of the paired tokenizer
      block_size     : int
      n_embd         : int
      n_head         : int
      n_layer        : int
      dropout        : float
      step           : int   — last completed training step
      model_state_dict: dict

    Optimizer state is NOT saved. Resumption restores weights only.
    """
    _assert_not_protected(CHECKPOINT_V3_PATH)

    payload = {
        "version":         "v3",
        "vocab_size":      vocab_size,
        "tokenizer_file":  os.path.basename(TOKENIZER_V3_PATH),
        "block_size":      BLOCK_SIZE,
        "n_embd":          N_EMBD,
        "n_head":          N_HEAD,
        "n_layer":         N_LAYER,
        "dropout":         DROPOUT,
        "step":            step,
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, CHECKPOINT_V3_PATH)
    print(f"  Checkpoint saved : {CHECKPOINT_V3_PATH}  (step={step})")


# ── main training entry point ──────────────────────────────────────────────────

def train():
    print(f"Device: {device}")
    print(f"Output checkpoint : {CHECKPOINT_V3_PATH}")
    print(f"Output tokenizer  : {TOKENIZER_V3_PATH}")

    # ── 1. Guard output files before any work ─────────────────────────────────
    resume_weights = False

    if os.path.exists(TOKENIZER_V3_PATH):
        if not _confirm(f"\nTokenizer already exists:\n  {TOKENIZER_V3_PATH}\nOverwrite?"):
            print("Aborted. No files were changed.")
            return

    if os.path.exists(CHECKPOINT_V3_PATH):
        print(f"\nCheckpoint already exists:\n  {CHECKPOINT_V3_PATH}")
        if _confirm("Resume training from saved weights (optimizer state will NOT be restored)?"):
            resume_weights = True
        elif _confirm("Overwrite checkpoint and start fresh?"):
            resume_weights = False
        else:
            print("Aborted. No files were changed.")
            return

    # ── 2. Corpus ──────────────────────────────────────────────────────────────
    text = load_corpus()

    # ── 3. Tokenizer (always retrained with the corpus) ───────────────────────
    tokenizer  = build_tokenizer(text)
    vocab_size = tokenizer.get_vocab_size()

    # ── 4. Encode + split ─────────────────────────────────────────────────────
    train_data, val_data = encode_and_split(text, tokenizer)

    # ── 5. Model ───────────────────────────────────────────────────────────────
    print("\nInitialising model ...")
    model = MiniGPTv2(
        vocab_size=vocab_size,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT,
    ).to(device)

    if resume_weights:
        print(f"  Loading weights from {CHECKPOINT_V3_PATH} ...")
        ckpt = torch.load(CHECKPOINT_V3_PATH, map_location=device, weights_only=False)
        # Verify the saved vocab matches the freshly trained tokenizer
        if ckpt["vocab_size"] != vocab_size:
            sys.exit(
                f"RESUME ERROR: checkpoint vocab_size={ckpt['vocab_size']} "
                f"!= new tokenizer vocab_size={vocab_size}.\n"
                "The tokenizer was retrained and no longer matches the saved weights.\n"
                "Start fresh instead of resuming."
            )
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Resumed from step {ckpt.get('step', '?')} (weights only)")

    print(f"  Parameters       : {sum(p.numel() for p in model.parameters()):,}")

    # Set seed before model init and training for reproducibility.
    # Exact results may still vary across hardware and non-deterministic ops.
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()

    # ── 6. Training loop ───────────────────────────────────────────────────────
    print(f"\nTraining for {MAX_ITERS} steps ...")
    for step in range(MAX_ITERS):
        xb, yb = get_batch("train", train_data, val_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % EVAL_INTERVAL == 0:
            scores = estimate_loss(model, train_data, val_data)
            print(
                f"  Step {step:4d} | "
                f"train: {scores['train']:.3f} | "
                f"val: {scores['val']:.3f}"
            )

        # Periodic checkpoint — weights only, optimizer state is NOT saved.
        if step > 0 and step % CHECKPOINT_INTERVAL == 0:
            print(f"  [periodic save at step {step}]")
            save_checkpoint(model, vocab_size, step=step)

    # ── 7. Save final checkpoint ───────────────────────────────────────────────
    print("\nSaving ...")
    save_checkpoint(model, vocab_size, step=MAX_ITERS - 1)

    # ── 8. Post-save verification ──────────────────────────────────────────────
    print("\nVerifying matched pair ...")
    saved = torch.load(CHECKPOINT_V3_PATH, map_location="cpu", weights_only=False)
    tok2  = Tokenizer.from_file(TOKENIZER_V3_PATH)

    assert saved["vocab_size"] == tok2.get_vocab_size(), (
        f"POST-SAVE MISMATCH: checkpoint={saved['vocab_size']} "
        f"tokenizer={tok2.get_vocab_size()}"
    )
    assert saved["tokenizer_file"] == os.path.basename(TOKENIZER_V3_PATH), (
        "tokenizer_file metadata does not match actual filename"
    )
    print(f"  vocab_size matches: {saved['vocab_size']}  OK")
    print(f"  tokenizer_file    : {saved['tokenizer_file']}  OK")
    print("\nv3 training complete. Matched pair saved.")


if __name__ == "__main__":
    train()
