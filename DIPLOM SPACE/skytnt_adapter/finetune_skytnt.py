"""
Finetune SkyTNT/midi-model on a local folder of MIDI files.

Usage (typical):
    python -m skytnt_adapter.finetune_skytnt \
        --data dataset/midi_raw \
        --pretrained skytnt/midi-model-tv2o-medium \
        --output checkpoints/skytnt \
        --max-len 2048 --batch-size 2 --max-step 4000

What it does
------------
1) Builds an MIDIModelConfig matching the chosen tokenizer/size combo.
2) Loads a pretrained checkpoint (HF Hub repo id, .ckpt, or .safetensors).
3) Wraps SkyTNT's MidiDataset / TrainMIDIModel and runs a Lightning Trainer.
4) Saves the best checkpoint, last checkpoint and a `tokenizer_config.json`
   into `--output` so downstream `export_skytnt_onnx.py` can pick it up.

This is a *thin* wrapper over upstream `train.py` so the training behaviour
stays identical.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

# Vendored upstream import path bootstrap
import skytnt_adapter  # noqa: F401  - ensures sys.path setup

import numpy as np
import torch
import lightning as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader

# Upstream modules (now importable thanks to skytnt_adapter.__init__)
from midi_model import MIDIModelConfig, config_name_list
from train import MidiDataset, TrainMIDIModel, get_midi_list


def _try_load_pretrained(model: TrainMIDIModel, source: str) -> None:
    """Load a pretrained checkpoint into ``model``.

    Accepts:
      * HuggingFace Hub repo id (e.g. ``skytnt/midi-model-tv2o-medium``)
      * Local path to a Lightning ``.ckpt`` or PyTorch ``.pt`` / ``.bin``
      * Local directory containing ``model.safetensors`` or ``pytorch_model.bin``
    """
    if not source:
        print("[finetune] no --pretrained set, training from scratch")
        return

    if os.path.exists(source):
        path = source
        if os.path.isdir(path):
            for cand in ("model.safetensors", "pytorch_model.bin", "model.ckpt"):
                cand_path = os.path.join(path, cand)
                if os.path.exists(cand_path):
                    path = cand_path
                    break
            else:
                raise FileNotFoundError(
                    f"--pretrained directory has no model.safetensors / "
                    f"pytorch_model.bin / model.ckpt: {source}"
                )
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(path, device="cpu")
        else:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[finetune] loaded local weights from {path} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        return

    # Otherwise: try HuggingFace Hub
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to fetch pretrained weights from the Hub."
        ) from exc
    print(f"[finetune] downloading pretrained from HF Hub: {source}")
    local_dir = snapshot_download(repo_id=source, allow_patterns=[
        "*.safetensors", "*.bin", "*.ckpt", "config.json"
    ])
    return _try_load_pretrained(model, local_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune SkyTNT/midi-model on local MIDI files")
    parser.add_argument("--data", type=str, required=True,
                        help="Folder with MIDI files (recursively scanned)")
    parser.add_argument("--config", type=str, default="tv2o-medium",
                        choices=config_name_list,
                        help="SkyTNT config name (matches the pretrained variant)")
    parser.add_argument("--pretrained", type=str, default="",
                        help="Pretrained source (HF Hub id / .ckpt / .safetensors / directory)")
    parser.add_argument("--output", type=str, default="checkpoints/skytnt",
                        help="Output directory for checkpoints + tokenizer config")
    parser.add_argument("--data-val-split", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batch-size-val", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--workers-val", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-step", type=int, default=200)
    parser.add_argument("--max-step", type=int, default=4000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--acc-grad", type=int, default=2)
    parser.add_argument("--accelerator", type=str, default="auto",
                        choices=["cpu", "gpu", "mps", "auto"])
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--devices", type=int, default=-1)
    parser.add_argument("--val-step", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", type=str, default="train", choices=["train", "lora"])
    parser.add_argument("--sample-seq", action="store_true", default=False)
    parser.add_argument("--gen-example-interval", type=int, default=0,
                        help="Disable inline generation by default to keep training fast")
    parser.add_argument("--quality", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default="",
                        help="Resume Lightning ckpt path")

    opt = parser.parse_args()
    os.makedirs(opt.output, exist_ok=True)
    pl.seed_everything(opt.seed)

    print(f"[finetune] config={opt.config} data={opt.data} output={opt.output}")
    config = MIDIModelConfig.from_name(opt.config)
    tokenizer = config.tokenizer

    midi_files = get_midi_list(opt.data)
    if not midi_files:
        print(f"[finetune] ERROR: no MIDI files found under {opt.data}", file=sys.stderr)
        sys.exit(2)
    random.shuffle(midi_files)
    val_n = min(opt.data_val_split, max(1, len(midi_files) // 20))
    train_files = midi_files[:-val_n]
    val_files = midi_files[-val_n:]
    print(f"[finetune] dataset: train={len(train_files)} val={len(val_files)}")

    train_ds = MidiDataset(train_files, tokenizer, max_len=opt.max_len, aug=True,
                           check_quality=opt.quality, rand_start=True)
    val_ds = MidiDataset(val_files, tokenizer, max_len=opt.max_len, aug=False,
                         check_quality=opt.quality, rand_start=False)
    train_dl = DataLoader(
        train_ds, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.workers, pin_memory=True,
        persistent_workers=opt.workers > 0,
        collate_fn=train_ds.collate_fn,
    )
    val_dl = DataLoader(
        val_ds, batch_size=opt.batch_size_val, shuffle=False,
        num_workers=opt.workers_val, pin_memory=True,
        persistent_workers=opt.workers_val > 0,
        collate_fn=val_ds.collate_fn,
    )

    if torch.cuda.is_available():
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)

    model = TrainMIDIModel(
        config,
        lr=opt.lr,
        weight_decay=opt.weight_decay,
        warmup=opt.warmup_step,
        max_step=opt.max_step,
        sample_seq=opt.sample_seq,
        gen_example_interval=opt.gen_example_interval,
        example_batch=1,
    )

    _try_load_pretrained(model, opt.pretrained)

    if opt.task == "lora":
        from peft import LoraConfig, TaskType
        model.requires_grad_(False)
        model.add_adapter(LoraConfig(
            r=64,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none", lora_alpha=128, lora_dropout=0,
        ))

    config.save_pretrained(opt.output)
    with open(os.path.join(opt.output, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer.to_dict(), f, indent=2)

    ckpt_callback = ModelCheckpoint(
        dirpath=os.path.join(opt.output, "lightning_ckpt"),
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        filename="best-epoch={epoch:02d}-val={val/loss:.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        default_root_dir=opt.output,
        precision=opt.precision,
        accumulate_grad_batches=opt.acc_grad,
        gradient_clip_val=opt.grad_clip,
        accelerator=opt.accelerator,
        devices=opt.devices,
        max_steps=opt.max_step,
        val_check_interval=opt.val_step or None,
        log_every_n_steps=10,
        callbacks=[ckpt_callback, lr_monitor],
    )
    trainer.fit(model, train_dl, val_dl, ckpt_path=opt.resume or None)

    # Persist a "model.ckpt" file with the final state_dict so export.py can ingest
    final_ckpt = os.path.join(opt.output, "model.ckpt")
    torch.save({"state_dict": model.state_dict()}, final_ckpt)
    print(f"[finetune] wrote {final_ckpt}")
    if ckpt_callback.best_model_path:
        best_dst = os.path.join(opt.output, "model_best.ckpt")
        shutil.copy2(ckpt_callback.best_model_path, best_dst)
        print(f"[finetune] best ckpt copied to {best_dst}")
    print("[finetune] done.")


if __name__ == "__main__":
    main()
