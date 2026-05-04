"""Pure-PyTorch training loop, designed to fit a Kaggle T4 (~15 GB VRAM) but
also runs on Mac MPS or CPU for local sanity-checks.

Usage:
    python -m model.train

Outputs:
    checkpoints/model_best.pth          best-by-val-loss state_dict
    checkpoints/model_last.pth          most recent epoch
    checkpoints/history.json            per-epoch metrics
    checkpoints/plots/loss_curve.png    if matplotlib available
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.dataset import MIDITokenDataset
from model.music_metrics import (
    TokenClassificationStats,
    aggregate_metrics,
    sequence_metrics,
)
from model.paths import resolve_checkpoint_dir
from model.transformer import TransformerLM, count_parameters

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

VOCAB_PATH = PROJECT_ROOT / "dataset" / "processed" / "vocab.json"
TRAIN_CHUNKS = PROJECT_ROOT / "dataset" / "processed" / "chunks" / "full_chunks_train.npy"
VAL_CHUNKS = PROJECT_ROOT / "dataset" / "processed" / "chunks" / "full_chunks_val.npy"
TEST_CHUNKS = PROJECT_ROOT / "dataset" / "processed" / "chunks" / "full_chunks_test.npy"

DEFAULTS = dict(
    seed=42,
    num_epochs=100,
    early_stopping_patience=12,
    batch_size=16,
    grad_accum_steps=2,
    learning_rate=3e-4,
    weight_decay=0.05,
    label_smoothing=0.05,
    warmup_epochs=4,
    min_lr_scale=0.1,
    max_grad_norm=1.0,
    max_len=1024,
    d_model=512,
    n_heads=8,
    n_layers=8,
    d_ff=2048,
    dropout=0.2,
    num_workers=2,
    sample_metric_max=1024,
    stratify_keys=1,
    ema_decay=0.999,
    key_weight_power=0.5,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MIDI Transformer LM from scratch")
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    p.add_argument("--device", type=str, default=None,
                   help="cuda|mps|cpu (auto if omitted)")
    p.add_argument("--no_amp", action="store_true",
                   help="disable mixed-precision (default: on for cuda)")
    p.add_argument("--resume", type=str, default=None,
                   help="path to checkpoint to resume from")
    p.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="writable dir for .pth / history / plots (default: project/checkpoints, "
        "or /kaggle/working/midi_gen_checkpoints if project is under /kaggle/input)",
    )
    return p.parse_args()


def pick_device(arg: str | None) -> str:
    if arg:
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        except Exception:
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def amp_context(device: str, enabled: bool):
    if not enabled or device != "cuda":
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=torch.float16)


def build_loaders(args, vocab_path: Path):
    train_ds = MIDITokenDataset(TRAIN_CHUNKS, vocab_path, max_len=args.max_len,
                                seed=args.seed + 1, augment=True)
    val_ds = MIDITokenDataset(VAL_CHUNKS, vocab_path, max_len=args.max_len,
                              seed=args.seed + 2, augment=False)
    test_ds = MIDITokenDataset(TEST_CHUNKS, vocab_path, max_len=args.max_len,
                               seed=args.seed + 3, augment=False)

    pin = (pick_device(args.device) == "cuda")
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    if int(getattr(args, "stratify_keys", 0)) > 0:
        keys = train_ds.key_token_per_sample()
        counts = Counter(keys)
        power = float(getattr(args, "key_weight_power", 0.5))
        weights = [1.0 / max(1, counts[k]) ** power for k in keys]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, sampler=sampler, **common)
        rare = min(counts.items(), key=lambda kv: kv[1])
        common_k = max(counts.items(), key=lambda kv: kv[1])
        print(
            f"key-stratified sampler enabled (power={power}); "
            f"rarest={rare[0]}({rare[1]}), most={common_k[0]}({common_k[1]})"
        )
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **common)

    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def build_scheduler(optimizer, num_epochs: int, warmup: int, min_lr_scale: float):
    if num_epochs <= 1:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
    warmup_eff = max(1, min(warmup, num_epochs - 1))
    warm = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=max(1e-3, 1.0 / float(warmup_eff)),
        end_factor=1.0,
        total_iters=warmup_eff,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, num_epochs - warmup_eff),
        eta_min=optimizer.param_groups[0]["lr"] * min_lr_scale,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warm, cosine], milestones=[warmup_eff]
    )


class ModelEMA:
    """Exponential moving average of parameters, kept in fp32.

    Use `update()` after every optimizer step. For evaluation, call
    `apply()` to swap shadow weights into the model and `restore()` to
    bring back the live training weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone().to(dtype=torch.float32)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow and param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.detach().to(dtype=torch.float32),
                    alpha=1.0 - self.decay,
                )

    @torch.no_grad()
    def apply(self, model: nn.Module) -> dict[str, torch.Tensor]:
        backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.detach().clone()
                param.data.copy_(
                    self.shadow[name].to(dtype=param.dtype, device=param.device)
                )
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state.get("decay", self.decay))
        loaded = state.get("shadow", {})
        for name in list(self.shadow.keys()):
            if name in loaded:
                self.shadow[name].copy_(loaded[name].to(self.shadow[name]))


def compute_loss(logits, targets, pad_id: int, label_smoothing: float):
    B, T, V = logits.shape
    flat_logits = logits.view(-1, V)
    flat_targets = targets.view(-1)
    losses = F.cross_entropy(
        flat_logits,
        flat_targets,
        reduction="none",
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
    ).view(B, T)
    valid = targets != pad_id
    valid[:, :2] = False  # ignore predicting the deterministic <GENRE_*>/<KEY_*> from <BOS>
    valid_f = valid.float()
    token_count = valid_f.sum(dim=1).clamp_min(1.0)
    sample_loss = (losses * valid_f).sum(dim=1) / token_count
    return sample_loss, valid


@torch.no_grad()
def evaluate(model, loader, pad_id, id2token, device, amp_enabled, sample_metric_max):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_samples = 0
    metric_rows: list[dict] = []
    metric_budget = sample_metric_max
    classification = TokenClassificationStats(id2token)

    for x, y, g in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        g = g.to(device, non_blocking=True)
        with amp_context(device, amp_enabled):
            logits = model(x, g)
        sample_loss, valid = compute_loss(logits, y, pad_id, 0.0)
        total_loss += float(sample_loss.sum().item())
        total_samples += logits.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += int(((preds == y) & valid).sum().item())
        total_tokens += int(valid.sum().item())

        classification.update(logits, y, valid)

        if metric_budget > 0:
            n = min(metric_budget, logits.size(0))
            for i in range(n):
                ids = preds[i][valid[i]].detach().cpu().tolist()
                tokens = [id2token.get(str(int(t)), "<UNK>") for t in ids]
                metric_rows.append(sequence_metrics(tokens))
            metric_budget -= n

    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_tokens)
    perplexity = math.exp(min(avg_loss, 20.0))
    metrics = aggregate_metrics(metric_rows)
    metrics.update({
        "loss": avg_loss,
        "accuracy": accuracy,
        "perplexity": perplexity,
        "samples_for_music_metrics": len(metric_rows),
    })
    metrics.update(classification.compute())
    return metrics


def maybe_plot(history: list[dict], path: Path) -> None:
    if plt is None or not history:
        return
    epochs = [h["epoch"] for h in history]
    train = [h["train_loss"] for h in history]
    train_ce = [h.get("train_ce", h["train_loss"]) for h in history]
    val = [h["val_loss"] for h in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train, label="train (smoothed+aug)", linestyle="--", alpha=0.6)
    plt.plot(epochs, train_ce, label="train CE (clean)")
    plt.plot(epochs, val, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Training curves")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> int:
    args = parse_args()
    device = pick_device(args.device)
    set_seed(args.seed, device)

    if not VOCAB_PATH.exists() or not TRAIN_CHUNKS.exists():
        print("missing vocab.json or chunks; run dataset/* pipeline first")
        return 1

    checkpoint_dir = resolve_checkpoint_dir(PROJECT_ROOT, args.checkpoint_dir)
    plots_dir = checkpoint_dir / "plots"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"checkpoints -> {checkpoint_dir}")

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_loaders(
        args, VOCAB_PATH
    )
    pad_id = train_ds.pad
    num_genres = train_ds.num_genres
    vocab_size = len(train_ds.token2id)

    model = TransformerLM(
        vocab_size=vocab_size,
        num_genres=num_genres,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len,
        pad_id=pad_id,
    ).to(device)

    n_params = count_parameters(model)
    print(f"device={device} vocab={vocab_size} num_genres={num_genres} params={n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = build_scheduler(
        optimizer, args.num_epochs, args.warmup_epochs, args.min_lr_scale
    )

    amp_enabled = (not args.no_amp) and device == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    ema_decay = float(getattr(args, "ema_decay", 0.0))
    ema = ModelEMA(model, decay=ema_decay) if ema_decay > 0.0 else None
    if ema is not None:
        print(f"EMA enabled (decay={ema_decay})")

    start_epoch = 1
    history: list[dict] = []
    best_val_loss = float("inf")
    epochs_without_improve = 0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        history = ckpt.get("history", [])
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        epochs_without_improve = int(ckpt.get("epochs_without_improve", 0))
        if ema is not None and "ema" in ckpt and ckpt["ema"] is not None:
            ema.load_state_dict(ckpt["ema"])
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    id2token = train_ds.id2token

    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        epoch_started = time.time()
        train_loss_sum = 0.0
        train_ce_sum = 0.0
        train_samples = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y, g) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            g = g.to(device, non_blocking=True)
            with amp_context(device, amp_enabled):
                logits = model(x, g)
                sample_loss, _ = compute_loss(logits, y, pad_id, args.label_smoothing)
                loss = sample_loss.mean() / args.grad_accum_steps

            if amp_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            with torch.no_grad():
                # unsmoothed cross-entropy on the same forward — directly
                # comparable to val_loss (which also uses smoothing=0).
                sample_ce, _ = compute_loss(logits.detach(), y, pad_id, 0.0)
                train_ce_sum += float(sample_ce.sum().item())

            if (step + 1) % args.grad_accum_steps == 0:
                if amp_enabled:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)

            train_loss_sum += float(sample_loss.sum().item())
            train_samples += logits.size(0)

        scheduler.step()
        avg_train_loss = train_loss_sum / max(1, train_samples)
        avg_train_ce = train_ce_sum / max(1, train_samples)

        ema_backup = ema.apply(model) if ema is not None else None
        val_metrics = evaluate(
            model, val_loader, pad_id, id2token, device, amp_enabled,
            args.sample_metric_max,
        )
        if ema is not None and ema_backup is not None:
            ema.restore(model, ema_backup)

        record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_ce": avg_train_ce,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_top5_accuracy": val_metrics["top5_accuracy"],
            "val_perplexity": val_metrics["perplexity"],
            "val_macro_f1": val_metrics["category_macro_f1"],
            "val_micro_f1": val_metrics["category_micro_f1"],
            "val_weighted_f1": val_metrics["category_weighted_f1"],
            "val_macro_precision": val_metrics["category_macro_precision"],
            "val_macro_recall": val_metrics["category_macro_recall"],
            "val_pitch_class_iou": val_metrics["pitch_class_iou"],
            "val_repeat_rate": val_metrics["repeat_rate"],
            "val_unique_token_ratio": val_metrics["unique_token_ratio"],
            "val_rhythm_diversity": val_metrics["rhythm_diversity"],
            "val_scale_coverage": val_metrics["scale_coverage"],
            "val_per_category": val_metrics["per_category"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_seconds": round(time.time() - epoch_started, 2),
        }
        history.append(record)
        with open(checkpoint_dir / "history.json", "w", encoding="utf-8") as h:
            json.dump(history, h, indent=2, ensure_ascii=False)
        maybe_plot(history, plots_dir / "loss_curve.png")
        print(json.dumps(record, ensure_ascii=False))

        ckpt_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "history": history,
            "best_val_loss": best_val_loss,
            "epochs_without_improve": epochs_without_improve,
            "args": vars(args),
            "ema": ema.state_dict() if ema is not None else None,
        }
        torch.save(ckpt_payload, checkpoint_dir / "model_last.pth")

        if val_metrics["loss"] < best_val_loss - 1e-4:
            best_val_loss = val_metrics["loss"]
            epochs_without_improve = 0
            # Persist the EMA-evaluated weights as the "best" snapshot so
            # downstream torchscript export uses what we just validated.
            best_payload = dict(ckpt_payload)
            if ema is not None:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                for name, shadow in ema.shadow.items():
                    if name in best_state:
                        best_state[name] = shadow.detach().clone().to(best_state[name].dtype)
                best_payload["model"] = best_state
            torch.save(best_payload, checkpoint_dir / "model_best.pth")
            print(f"  -> new best val_loss={best_val_loss:.4f}")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.early_stopping_patience:
                print(f"early stop after {args.early_stopping_patience} epochs without val improvement")
                break

    print("training done; running final test eval")
    if (checkpoint_dir / "model_best.pth").exists():
        ckpt = torch.load(checkpoint_dir / "model_best.pth", map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
    test_metrics = evaluate(
        model, test_loader, pad_id, id2token, device, amp_enabled,
        args.sample_metric_max,
    )
    with open(checkpoint_dir / "test_metrics.json", "w", encoding="utf-8") as h:
        json.dump(test_metrics, h, indent=2, ensure_ascii=False)
    print(json.dumps({"test": test_metrics}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
