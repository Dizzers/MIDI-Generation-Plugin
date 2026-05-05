"""
Export a (fine-tuned) SkyTNT/midi-model checkpoint to ONNX.

Two ONNX files are produced, matching upstream `export.py`:
  * model_base.onnx   - x + past_kv -> hidden + present_kv
  * model_token.onnx  - hidden + x + past_kv -> y + present_kv

Plus `tokenizer_config.json` so the C++ runtime can reproduce the vocab.

Usage:
    python -m skytnt_adapter.export_skytnt_onnx \
        --ckpt checkpoints/skytnt/model.ckpt \
        --config tv2o-medium \
        --out-dir artifacts/onnx
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import skytnt_adapter  # noqa: F401  - sys.path bootstrap

import torch

from midi_model import MIDIModel, MIDIModelConfig, config_name_list
from export import (
    MIDIModelBase,
    MIDIModelToken,
    export_onnx,
    get_past_kv,
)


def _resolve_ckpt(ckpt_arg: str) -> str:
    """Map directory inputs to the actual ckpt file."""
    if os.path.isdir(ckpt_arg):
        for cand in ("model_best.ckpt", "model.ckpt"):
            p = os.path.join(ckpt_arg, cand)
            if os.path.exists(p):
                return p
        # Look inside lightning_ckpt dir too
        ld = os.path.join(ckpt_arg, "lightning_ckpt")
        if os.path.isdir(ld):
            for f in sorted(os.listdir(ld)):
                if f.endswith(".ckpt"):
                    return os.path.join(ld, f)
    if os.path.exists(ckpt_arg):
        return ckpt_arg
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_arg}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Lightning .ckpt / dict checkpoint file or directory")
    parser.add_argument("--config", type=str, default="tv2o-medium",
                        choices=config_name_list)
    parser.add_argument("--lora", type=str, default="",
                        help="Optional LoRA adapter path (HF Hub id or local dir)")
    parser.add_argument("--out-dir", type=str, default="artifacts/onnx")
    parser.add_argument("--tokenizer-out", type=str, default="artifacts/tokenizer/tokenizer_config.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-simplify", action="store_true", default=False)

    opt = parser.parse_args()
    os.makedirs(opt.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(opt.tokenizer_out), exist_ok=True)

    config = MIDIModelConfig.from_name(opt.config)
    tokenizer = config.tokenizer
    model = MIDIModel(config).to(device=opt.device)

    ckpt_path = _resolve_ckpt(opt.ckpt)
    print(f"[export] loading ckpt {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=opt.device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    cleaned = {}
    for k, v in state_dict.items():
        # Lightning sometimes prefixes with "model." -- drop it
        cleaned[k.replace("model.", "", 1) if k.startswith("model.") else k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[export] state_dict loaded (missing={len(missing)}, unexpected={len(unexpected)})")

    if opt.lora:
        print(f"[export] merging LoRA from {opt.lora}")
        model = model.load_merge_lora(opt.lora)

    model.eval()

    # Persist tokenizer config so C++ can build the same vocab
    with open(opt.tokenizer_out, "w", encoding="utf-8") as f:
        json.dump(tokenizer.to_dict(), f, indent=2)
    print(f"[export] tokenizer config -> {opt.tokenizer_out}")

    # Persist the full model config too (used by app_onnx.py and Python tooling)
    full_config_path = os.path.join(os.path.dirname(opt.tokenizer_out), "config.json")
    config.save_pretrained(os.path.dirname(opt.tokenizer_out))
    if not os.path.exists(full_config_path):
        with open(full_config_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)

    model_base = MIDIModelBase(model).eval()
    model_token = MIDIModelToken(model).eval()

    meta = {"config_name": opt.config, "config": config}
    base_out = os.path.join(opt.out_dir, "model_base.onnx")
    token_out = os.path.join(opt.out_dir, "model_token.onnx")

    past_kv_shape = {0: "batch", 2: "past_seq"}
    present_kv_shape = {0: "batch", 2: "present_seq"}

    with torch.no_grad():
        # ===== model_base =====
        dynamic_axes = {
            "x": {0: "batch", 1: "mid_seq", 2: "token_seq"},
            "hidden": {0: "batch", 1: "mid_seq"},
        }
        x = torch.randint(tokenizer.vocab_size, (1, 16, tokenizer.max_token_seq),
                          dtype=torch.int64, device=opt.device)
        past_kv, in_names, out_names = get_past_kv(
            config.net_config, past_seq_len=16, torch_dtype=torch.float32,
            device=opt.device,
        )
        for name in in_names:
            dynamic_axes[name] = past_kv_shape
        for name in out_names:
            dynamic_axes[name] = present_kv_shape
        in_names = ["x"] + in_names
        out_names = ["hidden"] + out_names
        export_onnx(model_base, (x, past_kv), in_names, out_names,
                    dynamic_axes, meta, base_out)

        # ===== model_token =====
        dynamic_axes = {
            "x": {0: "batch", 1: "token_seq"},
            "hidden": {0: "batch", 1: "states"},
            "y": {0: "batch", 1: "token_seq1"},
        }
        hidden = torch.randn(1, 1, config.n_embd, device=opt.device)
        x = torch.randint(tokenizer.vocab_size, (1, tokenizer.max_token_seq // 2),
                          dtype=torch.int64, device=opt.device)
        past_kv, in_names, out_names = get_past_kv(
            config.net_token_config,
            past_seq_len=(tokenizer.max_token_seq // 2),
            torch_dtype=torch.float32, device=opt.device,
        )
        for name in in_names:
            dynamic_axes[name] = past_kv_shape
        for name in out_names:
            dynamic_axes[name] = present_kv_shape
        in_names = ["hidden", "x"] + in_names
        out_names = ["y"] + out_names
        export_onnx(model_token, (hidden, x, past_kv), in_names, out_names,
                    dynamic_axes, meta, token_out)

    print(f"[export] done.\n  base:  {base_out}\n  token: {token_out}")


if __name__ == "__main__":
    main()
