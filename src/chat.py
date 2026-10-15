import os
import sys
import torch
from tokenizers import Tokenizer

from model import MiniGPTv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Checkpoint priority: v3 (matched pair) > v2 (mismatched tokenizer, backup only).
# v1 checkpoints are never listed here — incompatible architecture.
CHECKPOINTS = [
    os.path.join(ROOT, "checkpoints", "my_minigpt_v3.pth"),
    os.path.join(ROOT, "checkpoints", "my_minigpt_v2.pth"),
]

# Fallback tokenizer used only when a checkpoint has no tokenizer_file metadata
# (i.e. the old v2 checkpoint).
_DEFAULT_TOKENIZER = os.path.join(ROOT, "tokenizers", "my_bpe_tokenizer.json")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Key present in v1 state dicts, absent in v2/v3.
_V1_KEY = "blocks.0.sa.heads.0.tril"


def _is_v2_checkpoint(state_dict):
    return _V1_KEY not in state_dict


def load_model():
    """
    Load the best available checkpoint.

    Returns
    -------
    model      : MiniGPTv2 in eval mode
    ckpt_vocab : int   — vocab_size stored in the checkpoint
    tok_file   : str or None — tokenizer basename from checkpoint metadata
                               (present in v3, absent in v2)
    """
    checkpoint_path = next((p for p in CHECKPOINTS if os.path.exists(p)), None)
    if checkpoint_path is None:
        sys.exit(
            "No compatible checkpoint found.\n"
            "Expected: checkpoints/my_minigpt_v3.pth (or v2 as fallback)\n"
            "Run: python src/train_v3.py"
        )

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if not _is_v2_checkpoint(ckpt["model_state_dict"]):
        sys.exit(
            f"ERROR: {checkpoint_path} uses the v1 architecture and is "
            "incompatible with the current model.py.\n"
            "Use checkpoints/my_minigpt_v3.pth or retrain with train_v3.py."
        )

    ckpt_vocab = ckpt["vocab_size"]
    ckpt_block = ckpt.get("block_size", 256)

    model = MiniGPTv2(
        vocab_size=ckpt_vocab,
        n_embd=ckpt.get("n_embd", 128),
        n_head=ckpt.get("n_head", 4),
        n_layer=ckpt.get("n_layer", 4),
        block_size=ckpt_block,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(
        f"  version={ckpt.get('version', 'v2')}, vocab_size={ckpt_vocab}, "
        f"block_size={ckpt_block}, n_embd={ckpt.get('n_embd', 128)}, "
        f"n_layer={ckpt.get('n_layer', 4)}"
    )

    tok_file = ckpt.get("tokenizer_file", None)
    return model, ckpt_vocab, tok_file


def load_tokenizer(tokenizer_file=None):
    """
    Load the tokenizer paired with the loaded checkpoint.

    Parameters
    ----------
    tokenizer_file : str or None
        Basename of the tokenizer file, as stored in v3 checkpoint metadata.
        When None (v2 checkpoint), falls back to _DEFAULT_TOKENIZER.
    """
    if tokenizer_file:
        path = os.path.join(ROOT, "tokenizers", tokenizer_file)
    else:
        path = _DEFAULT_TOKENIZER

    if not os.path.exists(path):
        sys.exit(
            f"Tokenizer not found: {path}\n"
            "Run: python src/train_v3.py"
        )
    tok = Tokenizer.from_file(path)
    print(f"  tokenizer: {os.path.basename(path)}  (vocab={tok.get_vocab_size()})")
    return tok


def validate_compatibility(ckpt_vocab, tokenizer):
    """
    Compare checkpoint vocab_size against tokenizer vocab_size.
    Prints a warning and returns False on mismatch; returns True when they match.
    """
    tok_vocab = tokenizer.get_vocab_size()
    if tok_vocab != ckpt_vocab:
        print(
            f"\nWARNING: vocab mismatch — checkpoint expects {ckpt_vocab} tokens, "
            f"tokenizer has {tok_vocab}.\n"
            f"Token IDs {tok_vocab}-{ckpt_vocab - 1} exist in the model but "
            f"cannot be produced by the current tokenizer.\n"
            f"Output quality will be degraded. "
            f"Retrain with: python src/train_v3.py\n"
        )
        return False
    return True


def ask(model, tokenizer, question, max_new_tokens=200, temperature=0.8):
    prompt = f"User: {question}\nAssistant:"
    ids = tokenizer.encode(prompt).ids
    context = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(
            context, max_new_tokens=max_new_tokens, temperature=temperature
        )

    # Decode only the newly generated tokens, not the prompt.
    new_ids = output[0][len(ids):].tolist()
    answer = tokenizer.decode(new_ids)

    for marker in ["\n\nUser:", "\nUser:"]:
        if marker in answer:
            answer = answer.split(marker)[0]
    return answer.strip()


def main():
    print(f"Device: {device}")
    model, ckpt_vocab, tok_file = load_model()
    tokenizer = load_tokenizer(tok_file)
    validate_compatibility(ckpt_vocab, tokenizer)
    print("MiniGPT ready. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("MiniGPT: Goodbye.")
            break
        print("MiniGPT:", ask(model, tokenizer, question))
        print()


if __name__ == "__main__":
    main()
