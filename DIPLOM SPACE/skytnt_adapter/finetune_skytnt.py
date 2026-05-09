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
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch.utils.data import DataLoader

# Upstream modules (now importable thanks to skytnt_adapter.__init__)
from midi_model import MIDIModelConfig, config_name_list
from train import MidiDataset, TrainMIDIModel, get_midi_list

import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score, precision_score


class ExtendedTrainMIDIModel(TrainMIDIModel):
    """TrainMIDIModel with richer validation metrics.

    Extra metrics logged each validation step:
      val/perplexity   – exp(val_loss), standard LM metric
      val/top3_acc     – top-3 token accuracy (reflects sampling quality)
      val/top5_acc     – top-5 token accuracy
      val/f1_micro     – micro-averaged F1 across all token classes
      val/f1_macro     – macro-averaged F1 (equal weight to rare tokens)
      val/recall_macro – macro-averaged recall
      val/prec_macro   – macro-averaged precision

    At the end of each training epoch, prints a short summary and appends one
    JSON line to ``<output>/epoch_metrics.jsonl`` (same folder as ``--output``).
    """

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        self.log(
            "train/loss_epoch_avg",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[:, :-1].contiguous()
        y = batch[:, 1:].contiguous()
        hidden = self.forward(x)
        hidden = hidden.reshape(-1, hidden.shape[-1])
        y_flat = y.reshape(-1, y.shape[-1])
        x_in = y_flat[:, :-1]
        logits = self.forward_token(hidden, x_in)

        targets = y_flat.reshape(-1)
        logits_2d = logits.view(-1, self.tokenizer.vocab_size)

        pad_id = self.tokenizer.pad_id
        mask = targets != pad_id
        logits_m = logits_2d[mask]
        targets_m = targets[mask]

        loss = F.cross_entropy(logits_2d, targets, reduction="mean", ignore_index=pad_id)
        perplexity = torch.exp(loss)

        # Token-level accuracy (top-1)
        preds = logits_m.argmax(dim=-1)
        acc = (preds == targets_m).float().mean()

        # Top-k accuracy
        def topk_acc(k):
            topk = logits_m.topk(k, dim=-1).indices
            return (topk == targets_m.unsqueeze(-1)).any(dim=-1).float().mean()

        top3 = topk_acc(3)
        top5 = topk_acc(5)

        # sklearn classification metrics (computed on CPU, sampled if too large)
        y_np = targets_m.cpu().numpy()
        p_np = preds.cpu().numpy()
        MAX_SAMPLES = 50_000
        if len(y_np) > MAX_SAMPLES:
            idx = np.random.choice(len(y_np), MAX_SAMPLES, replace=False)
            y_np, p_np = y_np[idx], p_np[idx]

        sk_kw = dict(zero_division=0)
        f1_micro   = f1_score(y_np, p_np, average="micro",  **sk_kw)
        f1_macro   = f1_score(y_np, p_np, average="macro",  **sk_kw)
        rec_macro  = recall_score(y_np,   p_np, average="macro", **sk_kw)
        prec_macro = precision_score(y_np, p_np, average="macro", **sk_kw)

        self.log_dict({
            "val/loss":        loss,
            "val/acc":         acc,
            "val/perplexity":  perplexity,
            "val/top3_acc":    top3,
            "val/top5_acc":    top5,
            "val/f1_micro":    torch.tensor(f1_micro,   dtype=torch.float32),
            "val/f1_macro":    torch.tensor(f1_macro,   dtype=torch.float32),
            "val/recall_macro": torch.tensor(rec_macro, dtype=torch.float32),
            "val/prec_macro":  torch.tensor(prec_macro, dtype=torch.float32),
        }, sync_dist=True)
        return loss

    @staticmethod
    def _metrics_to_jsonable(raw: dict) -> dict[str, float | int | str]:
        out: dict[str, float | int | str] = {}
        for k, v in raw.items():
            if not k.startswith(("train/", "val/")):
                continue
            if torch.is_tensor(v):
                out[k] = float(v.detach().float().cpu().item())
            elif isinstance(v, (float, int)):
                out[k] = v
            else:
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = str(v)
        return out

    @rank_zero_only
    def on_train_epoch_end(self) -> None:
        trainer = self.trainer
        if trainer is None:
            return
        flat = self._metrics_to_jsonable(dict(trainer.callback_metrics))
        row = {
            "epoch": int(self.current_epoch),
            "global_step": int(self.global_step),
            **flat,
        }
        root = Path(trainer.default_root_dir or ".")
        log_path = root / "epoch_metrics.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        t_loss = flat.get("train/loss_epoch_avg", flat.get("train/loss"))
        v_loss = flat.get("val/loss")
        parts = [
            f"[finetune] epoch {self.current_epoch} end",
            f"step={self.global_step}",
        ]
        if t_loss is not None:
            parts.append(f"train_loss={t_loss:.6f}")
        if v_loss is not None:
            parts.append(f"val_loss={v_loss:.6f}")
            for key in (
                "val/acc",
                "val/perplexity",
                "val/f1_macro",
                "val/top3_acc",
            ):
                if key in flat:
                    parts.append(f"{key}={flat[key]:.4f}")
        print(" | ".join(parts), flush=True)


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
    parser.add_argument(
        "--data-val-split",
        type=int,
        default=64,
        help="Upper cap on how many MIDI files go to validation (see also --val-fraction)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of files for validation after shuffle (default 0.15). "
        "Capped by --data-val-split and at least 1 train file remains.",
    )
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
    parser.add_argument(
        "--ckpt-weights-only",
        action="store_true",
        help="Lightning checkpoints store only weights (smaller on disk; full ckpt needed for --resume)",
    )
    parser.add_argument(
        "--no-save-last",
        action="store_true",
        help="Do not write last.ckpt (keeps only best; saves one large file on disk)",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop if val/loss does not improve for this many validation runs (0 = off)",
    )
    parser.add_argument(
        "--min-file-bytes",
        type=int,
        default=3000,
        help="Skip MIDI files smaller than this (bytes). Tiny files are often broken or empty.",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=384000,
        help="Skip MIDI files larger than this (bytes). Raise if your corpus is mostly "
        "long multi-track exports (may need smaller --max-len / batch if OOM or loader errors).",
    )

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

    # Pre-filter by file size on disk (not musical length). Defaults drop tiny junk and
    # very large exports that sometimes break loaders or blow memory after tokenization.
    lo, hi = opt.min_file_bytes, opt.max_file_bytes
    if lo < 0 or hi < lo:
        print("[finetune] ERROR: need 0 <= --min-file-bytes <= --max-file-bytes", file=sys.stderr)
        sys.exit(2)
    before = len(midi_files)
    midi_files = [p for p in midi_files if lo <= os.path.getsize(p) <= hi]
    removed = before - len(midi_files)
    if removed:
        print(f"[finetune] filtered out {removed} files outside size range [{lo}, {hi}] bytes")
    if not midi_files:
        print(f"[finetune] ERROR: no valid MIDI files remain after size filtering", file=sys.stderr)
        sys.exit(2)

    if not (0.0 < opt.val_fraction < 1.0):
        print("[finetune] ERROR: --val-fraction must be between 0 and 1 (exclusive)", file=sys.stderr)
        sys.exit(2)

    random.shuffle(midi_files)
    val_n = int(round(len(midi_files) * opt.val_fraction))
    val_n = max(1, val_n)
    val_n = min(val_n, opt.data_val_split, len(midi_files) - 1)
    train_files = midi_files[:-val_n]
    val_files = midi_files[-val_n:]
    print(
        f"[finetune] dataset: train={len(train_files)} val={len(val_files)} "
        f"(val_fraction={opt.val_fraction}, cap={opt.data_val_split})"
    )

    train_ds = MidiDataset(
        train_files,
        tokenizer,
        max_len=opt.max_len,
        min_file_size=opt.min_file_bytes,
        max_file_size=opt.max_file_bytes,
        aug=True,
        check_quality=opt.quality,
        rand_start=True,
    )
    val_ds = MidiDataset(
        val_files,
        tokenizer,
        max_len=opt.max_len,
        min_file_size=opt.min_file_bytes,
        max_file_size=opt.max_file_bytes,
        aug=False,
        check_quality=opt.quality,
        rand_start=False,
    )
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

    model = ExtendedTrainMIDIModel(
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
        save_last=not opt.no_save_last,
        save_weights_only=opt.ckpt_weights_only,
        auto_insert_metric_name=False,
        filename="best-epoch={epoch:02d}-val={val/loss:.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks: list = [ckpt_callback, lr_monitor]
    if opt.early_stop_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=opt.early_stop_patience,
                verbose=True,
            )
        )

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
        callbacks=callbacks,
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
