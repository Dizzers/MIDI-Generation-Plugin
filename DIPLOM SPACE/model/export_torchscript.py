import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.transformer import TransformerLM  # noqa: E402


def clean_state_dict(state_dict):
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


class ScriptWrapper(torch.nn.Module):
    def __init__(self, model: TransformerLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, genre_id: torch.Tensor) -> torch.Tensor:
        # (B, T) -> (B, T, vocab)
        return self.model(input_ids, genre_id=genre_id)


def load_vocab(vocab_path: Path):
    with open(vocab_path, "r") as handle:
        vocab = json.load(handle)
    token2id = vocab["token2id"]
    id2token = {int(idx): token for idx, token in vocab["id2token"].items()}
    genre_tokens = sorted(token for token in token2id if token.startswith("<GENRE_"))
    if not genre_tokens:
        genre_tokens = ["<GENRE_NONE>"]
    num_genres = max(1, len(genre_tokens))
    return token2id, id2token, num_genres


def load_model(checkpoint_path: Path, vocab_path: Path, device: str):
    token2id, _, num_genres = load_vocab(vocab_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
    model_config = dict(model_config)
    model_config.pop("vocab_size", None)
    model_config.setdefault("pad_id", token2id.get("<PAD>"))
    model_config.setdefault("num_roles", 0)
    model_config.setdefault("num_genres", num_genres)

    model = TransformerLM(len(token2id), **model_config).to(device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(clean_state_dict(state_dict))
    model.eval()
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Export trained TransformerLM to TorchScript for JUCE/LibTorch")
    p.add_argument("--checkpoint", type=str, default="checkpoints/model_best.pth")
    p.add_argument("--vocab", type=str, default="dataset/processed/vocab.json")
    p.add_argument("--out", type=str, default="checkpoints/model_best.ts.pt")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--copy-to-plugin-bin", action="store_true")
    p.add_argument("--plugin-bin", type=str, default="plugin/juce/bin")
    return p.parse_args()


def main():
    args = parse_args()
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    vocab_path = PROJECT_ROOT / args.vocab
    out_path = PROJECT_ROOT / args.out

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab not found: {vocab_path}")

    device = args.device
    token2id, _, _ = load_vocab(vocab_path)
    model = load_model(checkpoint_path, vocab_path, device=device)
    wrapper = ScriptWrapper(model)

    # Prefer scripting, but fall back to tracing for TorchScript-incompatible ops.
    try:
        scripted = torch.jit.script(wrapper)
    except Exception as exc:
        print(f"[warn] torch.jit.script failed ({type(exc).__name__}: {exc}). Falling back to torch.jit.trace.")
        T = int(min(32, getattr(model, "max_len", 256)))
        # Use a simple valid input for tracing.
        example_ids = torch.zeros((1, T), dtype=torch.long, device=device)
        # Put some non-zero ids to avoid degenerate execution paths.
        vocab_size = len(token2id)
        if vocab_size > 10:
            example_ids[0, 0] = int(token2id.get("<BOS>", 1))
            example_ids[0, 1] = int(token2id.get("<GENRE_TRAP>", 2))
            example_ids[0, 2] = int(token2id.get("<KEY_UNKNOWN>", 3))
        example_genre = torch.zeros((1,), dtype=torch.long, device=device)
        scripted = torch.jit.trace(wrapper, (example_ids, example_genre), strict=False, check_trace=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out_path))
    print(f"Saved TorchScript module: {out_path}")

    if args.copy_to_plugin_bin:
        plugin_bin = PROJECT_ROOT / args.plugin_bin
        plugin_bin.mkdir(parents=True, exist_ok=True)

        dst_model = plugin_bin / "model_best.ts.pt"
        dst_vocab = plugin_bin / "vocab.json"

        dst_model.write_bytes(out_path.read_bytes())
        dst_vocab.write_bytes(vocab_path.read_bytes())
        print(f"Copied model to: {dst_model}")
        print(f"Copied vocab to: {dst_vocab}")


if __name__ == "__main__":
    main()

