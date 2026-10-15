import os
import re
import torch
from itertools import islice
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from model import MiniGPTv2

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_PATH = os.path.join(ROOT, "tokenizers", "my_bpe_tokenizer.json")
CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints", "my_minigpt_fineweb.pth")

# ── hyperparameters ────────────────────────────────────────────────────────────
MAX_DOCUMENTS = 3000
MAX_CHARS_PER_DOC = 4000
BLOCK_SIZE = 256
BATCH_SIZE = 32
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1
MAX_ITERS = 3000
EVAL_INTERVAL = 250
LR = 3e-4

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── data ───────────────────────────────────────────────────────────────────────
def load_corpus():
    from datasets import load_dataset
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT",
        split="train", streaming=True
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

    print(f"Documents: {len(documents)}, chars: {sum(map(len, documents))}")
    return "\n\n".join(documents)


# ── tokenizer ──────────────────────────────────────────────────────────────────
def build_tokenizer(text):
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=2000, min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    )
    tok.train_from_iterator(
        (text[i:i + 100_000] for i in range(0, len(text), 100_000)),
        trainer=trainer,
        length=(len(text) + 99_999) // 100_000,
    )
    tok.save(TOKENIZER_PATH)
    print(f"Tokenizer saved → {TOKENIZER_PATH}  (vocab={tok.get_vocab_size()})")
    return tok


# ── training ───────────────────────────────────────────────────────────────────
def get_batch(data, split, train_data, val_data):
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([source[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([source[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    results = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(20):
            xb, yb = get_batch(None, split, train_data, val_data)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        results[split] = sum(losses) / len(losses)
    model.train()
    return results


def _confirm_overwrite(path, label):
    """Return True only if the user explicitly types y."""
    ans = input(f"{label} already exists:\n  {path}\nOverwrite? [y/N]: ")
    return ans.strip().lower() == "y"


def train():
    print("Device:", device)

    # Guard both output files before doing any work or downloading data.
    blocked = []
    if os.path.exists(CHECKPOINT_PATH):
        blocked.append((CHECKPOINT_PATH, "Checkpoint"))
    if os.path.exists(TOKENIZER_PATH):
        blocked.append((TOKENIZER_PATH, "Tokenizer"))

    for path, label in blocked:
        if not _confirm_overwrite(path, label):
            print("Aborted. No files were changed.")
            return

    text = load_corpus()
    tokenizer = build_tokenizer(text)

    data = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    print(f"Tokens — train: {len(train_data)}, val: {len(val_data)}")

    model = MiniGPTv2(
        vocab_size=tokenizer.get_vocab_size(),
        n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER,
        block_size=BLOCK_SIZE, dropout=DROPOUT,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()

    for step in range(MAX_ITERS):
        xb, yb = get_batch(None, "train", train_data, val_data)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % EVAL_INTERVAL == 0:
            scores = estimate_loss(model, train_data, val_data)
            print(f"Step {step:4d} | train: {scores['train']:.3f} | val: {scores['val']:.3f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.get_vocab_size(),
        "block_size": BLOCK_SIZE,
        "n_embd": N_EMBD, "n_head": N_HEAD, "n_layer": N_LAYER, "dropout": DROPOUT,
    }, CHECKPOINT_PATH)
    print(f"Checkpoint saved → {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
