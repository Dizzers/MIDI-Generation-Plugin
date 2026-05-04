"""Convert checkpoints/model_best.pth to a TorchScript module compatible with
the existing JUCE plugin (DIPLOM SPACE/plugin/juce/Source/ModelInference.cpp,
which calls module.forward({x, g}).toTensor()).

Steps:
    1. Load checkpoint
    2. Reconstruct TransformerLM with checkpoint hyperparams
    3. torch.jit.script(model) and save to checkpoints/model_best.ts.pt
    4. Sanity round-trip: load the .ts.pt and call forward(dummy_x, dummy_g)
    5. Optionally copy .ts.pt and vocab.json into plugin/juce/bin/

Usage:
    python -m model.export_torchscript [--copy-to-bin] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.paths import resolve_checkpoint_dir, resolve_plugin_bin_dir
from model.transformer import TransformerLM, count_parameters

VOCAB_PATH = PROJECT_ROOT / "dataset" / "processed" / "vocab.json"


def load_checkpoint(path: Path, device: str):
    payload = torch.load(path, map_location=device)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    args = payload.get("args", {}) if isinstance(payload, dict) else {}
    return state_dict, args


def main() -> int:
    parser = argparse.ArgumentParser(description="Export trained TransformerLM to TorchScript")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="checkpoint root (default: same as model.train; under /kaggle/input -> /kaggle/working/...)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to model_best.pth (default: <checkpoint_dir>/model_best.pth)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="TorchScript output path (default: <checkpoint_dir>/model_best.ts.pt)",
    )
    parser.add_argument("--copy-to-bin", action="store_true",
                        help="copy .ts.pt + vocab.json into plugin bin (writable path on Kaggle)")
    args = parser.parse_args()

    checkpoint_dir = resolve_checkpoint_dir(PROJECT_ROOT, args.checkpoint_dir)
    ckpt_path = Path(args.ckpt) if args.ckpt else checkpoint_dir / "model_best.pth"
    plugin_bin_dir = resolve_plugin_bin_dir(PROJECT_ROOT)
    if not ckpt_path.exists():
        print(f"missing checkpoint: {ckpt_path}")
        return 1
    if not VOCAB_PATH.exists():
        print(f"missing vocab.json: {VOCAB_PATH}")
        return 1

    with open(VOCAB_PATH, encoding="utf-8") as handle:
        vocab = json.load(handle)
    token2id = vocab["token2id"]
    pad_id = token2id["<PAD>"]
    vocab_size = len(token2id)
    num_genres = max(1, sum(1 for t in token2id if t.startswith("<GENRE_")))

    state_dict, ckpt_args = load_checkpoint(ckpt_path, args.device)

    cfg = dict(
        d_model=ckpt_args.get("d_model", 512),
        n_heads=ckpt_args.get("n_heads", 8),
        n_layers=ckpt_args.get("n_layers", 8),
        d_ff=ckpt_args.get("d_ff", 2048),
        dropout=ckpt_args.get("dropout", 0.2),
        max_len=ckpt_args.get("max_len", 1024),
    )

    model = TransformerLM(
        vocab_size=vocab_size,
        num_genres=num_genres,
        pad_id=pad_id,
        **cfg,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"  load_state_dict: missing={missing} unexpected={unexpected}")
    model.eval()
    model.to(args.device)

    print(f"model: {count_parameters(model):,} params, vocab={vocab_size}, num_genres={num_genres}")
    print(f"config: {cfg}")

    scripted = torch.jit.script(model)
    out_path = Path(args.out) if args.out else checkpoint_dir / "model_best.ts.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out_path))
    print(f"saved {out_path}")

    loaded = torch.jit.load(str(out_path), map_location=args.device)
    dummy_x = torch.randint(0, vocab_size, (1, 16), dtype=torch.long, device=args.device)
    dummy_g = torch.zeros(1, dtype=torch.long, device=args.device)
    with torch.no_grad():
        out = loaded.forward(dummy_x, dummy_g)
    expected_shape = (1, 16, vocab_size)
    assert tuple(out.shape) == expected_shape, f"unexpected output shape {out.shape}, want {expected_shape}"
    print(f"roundtrip OK: forward(x[1,16], g[1]) -> {tuple(out.shape)}")

    if args.copy_to_bin:
        plugin_bin_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(out_path, plugin_bin_dir / "model_best.ts.pt")
        shutil.copy(VOCAB_PATH, plugin_bin_dir / "vocab.json")
        print(f"copied {out_path.name} and vocab.json into {plugin_bin_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
