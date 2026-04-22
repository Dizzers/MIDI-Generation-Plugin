import json
import math
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.dataset import MIDIDataset
from model.transformer import TransformerLM

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

NUM_EPOCHS = 50
BATCH_SIZE = 12
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.02
GRAD_ACCUM_STEPS = 2
LABEL_SMOOTHING = 0.08
EARLY_STOPPING_PATIENCE = 10
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

MAX_LEN = 256
D_MODEL = 384
N_HEADS = 8
N_LAYERS = 8
D_FF = 2048
DROPOUT = 0.15
WARMUP_EPOCHS = 5
MIN_LR_SCALE = 0.10

AUGMENT_CONFIG = {
    "transpose_prob": 0.30,
    "transpose_range": 5,
    "time_stretch_prob": 0.25,
    "time_stretch_range": (0.93, 1.07),
    "velocity_jitter_prob": 0.15,
    "velocity_jitter": 2,
}

VOCAB_PATH = PROJECT_ROOT / "dataset" / "processed" / "vocab.json"
TRAIN_CHUNKS_PATH = PROJECT_ROOT / "dataset" / "processed" / "chunks" / "full_chunks_train.npy"
VAL_CHUNKS_PATH = PROJECT_ROOT / "dataset" / "processed" / "chunks" / "full_chunks_val.npy"
TEST_CHUNKS_PATH = PROJECT_ROOT / "dataset" / "processed" / "chunks" / "full_chunks_test.npy"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
PLOTS_DIR = CHECKPOINT_DIR / "plots"

USE_AMP = True
NUM_WORKERS = 4
PIN_MEMORY = DEVICE == "cuda"
PERSISTENT_WORKERS = True
RESUME_FROM_CHECKPOINT = False
RESUME_CHECKPOINT_PATH = CHECKPOINT_DIR / "model_best.pth"


def autocast_context(enabled):
    return torch.amp.autocast(device_type="cuda") if enabled else nullcontext()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def clean_state_dict(state_dict):
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def build_scheduler(optimizer):
    if NUM_EPOCHS <= 1:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=max(1e-3, 1.0 / float(max(1, WARMUP_EPOCHS))),
        end_factor=1.0,
        total_iters=max(1, WARMUP_EPOCHS),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, NUM_EPOCHS - max(1, WARMUP_EPOCHS)),
        eta_min=LEARNING_RATE * MIN_LR_SCALE,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[max(1, WARMUP_EPOCHS)],
    )


def build_loaders():
    train_dataset = MIDIDataset(
        chunks_path=str(TRAIN_CHUNKS_PATH),
        vocab_path=str(VOCAB_PATH),
        max_len=MAX_LEN,
        samples_per_epoch=None,
        seed=SEED + 11,
        augment_config=AUGMENT_CONFIG,
        apply_augmentation=True,
    )
    val_dataset = MIDIDataset(
        chunks_path=str(VAL_CHUNKS_PATH),
        vocab_path=str(VOCAB_PATH),
        max_len=MAX_LEN,
        samples_per_epoch=None,
        seed=SEED + 22,
        augment_config=AUGMENT_CONFIG,
        apply_augmentation=False,
    )
    test_dataset = MIDIDataset(
        chunks_path=str(TEST_CHUNKS_PATH),
        vocab_path=str(VOCAB_PATH),
        max_len=MAX_LEN,
        samples_per_epoch=None,
        seed=SEED + 33,
        augment_config=AUGMENT_CONFIG,
        apply_augmentation=False,
    )

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "persistent_workers": PERSISTENT_WORKERS and NUM_WORKERS > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def compute_sample_losses(logits, targets, pad_id):
    batch_size, seq_len, vocab_size = logits.shape
    token_losses = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        reduction="none",
        ignore_index=pad_id,
        label_smoothing=LABEL_SMOOTHING,
    ).view(batch_size, seq_len)

    valid_mask = targets != pad_id
    # Prefix structure is: <BOS>, <GENRE_*>, <KEY_*>
    prefix_len = min(3, seq_len)
    valid_mask[:, :prefix_len] = False
    valid_mask_f = valid_mask.float()
    token_count = valid_mask_f.sum(dim=1).clamp_min(1.0)
    sample_loss = (token_losses * valid_mask_f).sum(dim=1) / token_count
    return sample_loss, valid_mask


def best_scale_coverage(note_pitches):
    if not note_pitches:
        return 0.0
    pitch_classes = [pitch % 12 for pitch in note_pitches]
    major = {0, 2, 4, 5, 7, 9, 11}
    minor = {0, 2, 3, 5, 7, 8, 10}
    best = 0.0
    for root in range(12):
        major_set = {(n + root) % 12 for n in major}
        minor_set = {(n + root) % 12 for n in minor}
        major_cov = sum(pc in major_set for pc in pitch_classes) / len(pitch_classes)
        minor_cov = sum(pc in minor_set for pc in pitch_classes) / len(pitch_classes)
        best = max(best, major_cov, minor_cov)
    return best


def sequence_music_metrics(token_ids, id2token, pad_id):
    ids = [int(i) for i in token_ids if int(i) != pad_id]
    tokens = [id2token.get(i, "<UNK>") for i in ids]
    if len(tokens) < 2:
        return {"repeat_rate": 0.0, "unique_token_ratio": 0.0, "rhythm_diversity": 0.0, "scale_coverage": 0.0}

    repeat_rate = sum(1 for idx in range(1, len(tokens)) if tokens[idx] == tokens[idx - 1]) / (len(tokens) - 1)
    unique_token_ratio = len(set(tokens)) / len(tokens)
    note_pitches = []
    time_shifts = []
    for token in tokens:
        if token.startswith("NOTE_ON_"):
            try:
                note_pitches.append(int(token.split("_")[2], 16))
            except Exception:
                pass
        elif token.startswith("TIME_SHIFT_"):
            try:
                time_shifts.append(int(token.split("_")[2], 16))
            except Exception:
                pass
    rhythm_diversity = len(set(time_shifts)) / max(1, len(time_shifts))
    scale_coverage = best_scale_coverage(note_pitches)
    return {
        "repeat_rate": repeat_rate,
        "unique_token_ratio": unique_token_ratio,
        "rhythm_diversity": rhythm_diversity,
        "scale_coverage": scale_coverage,
    }


def evaluate(model, loader, pad_id, id2token, amp_enabled):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    total_tokens = 0
    music_agg = {"repeat_rate": 0.0, "unique_token_ratio": 0.0, "rhythm_diversity": 0.0, "scale_coverage": 0.0}
    music_count = 0

    with torch.no_grad():
        for x, y, genre_idx in loader:
            x = x.to(DEVICE, non_blocking=PIN_MEMORY)
            y = y.to(DEVICE, non_blocking=PIN_MEMORY)
            genre_idx = genre_idx.to(DEVICE, non_blocking=PIN_MEMORY)

            with autocast_context(amp_enabled):
                logits = model(x, genre_id=genre_idx)

            sample_losses, valid_mask = compute_sample_losses(logits, y, pad_id)
            total_loss += sample_losses.sum().item()
            total_samples += logits.size(0)

            preds = logits.argmax(dim=-1)
            total_correct += ((preds == y) & valid_mask).sum().item()
            total_tokens += valid_mask.sum().item()

            for idx in range(logits.size(0)):
                pred_ids = preds[idx][valid_mask[idx]].detach().cpu().tolist()
                metrics = sequence_music_metrics(pred_ids, id2token, pad_id)
                for key in music_agg:
                    music_agg[key] += metrics[key]
                music_count += 1

    model.train()
    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_tokens)
    avg_ppl = math.exp(min(avg_loss, 20.0))
    music_metrics = {key: value / max(1, music_count) for key, value in music_agg.items()}
    return avg_loss, avg_acc, avg_ppl, music_metrics


def save_plots(history):
    if plt is None:
        return
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "loss_curve.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "val_acc_curve.png", dpi=140)
    plt.close()


def main():
    print("Loading full-MIDI training pipeline...")
    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = build_loaders()
    print(f"Device: {DEVICE}")
    print(f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    print(f"Steps/epoch: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    with open(VOCAB_PATH) as handle:
        vocab = json.load(handle)
    token2id = vocab["token2id"]
    id2token = {int(idx): token for idx, token in vocab["id2token"].items()}
    vocab_size = len(token2id)
    pad_id = token2id["<PAD>"]

    model_config = {
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "d_ff": D_FF,
        "dropout": DROPOUT,
        "max_len": MAX_LEN,
        "pad_id": pad_id,
        "num_roles": 0,
        "num_genres": train_dataset.num_genres,
    }
    model = TransformerLM(vocab_size, **model_config).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer)
    amp_enabled = USE_AMP and DEVICE.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_ppl": [], "learning_rates": [], "music_metrics": []}
    best_val_loss = float("inf")
    early_stop_counter = 0

    if RESUME_FROM_CHECKPOINT and RESUME_CHECKPOINT_PATH.exists():
        checkpoint = torch.load(RESUME_CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(clean_state_dict(checkpoint["model_state_dict"]))

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for step, (x, y, genre_idx) in enumerate(progress, start=1):
            x = x.to(DEVICE, non_blocking=PIN_MEMORY)
            y = y.to(DEVICE, non_blocking=PIN_MEMORY)
            genre_idx = genre_idx.to(DEVICE, non_blocking=PIN_MEMORY)

            with autocast_context(amp_enabled):
                logits = model(x, genre_id=genre_idx)
                batch_loss, _ = compute_sample_losses(logits, y, pad_id)
                loss = batch_loss.mean() / max(1, GRAD_ACCUM_STEPS)

            scaler.scale(loss).backward()
            total_train_loss += loss.item() * max(1, GRAD_ACCUM_STEPS)

            if step % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix({"loss": f"{loss.item() * max(1, GRAD_ACCUM_STEPS):.4f}"})

        if len(train_loader) % GRAD_ACCUM_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        val_loss, val_acc, val_ppl, music_metrics = evaluate(model, val_loader, pad_id, id2token, amp_enabled)
        scheduler.step()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_ppl"].append(val_ppl)
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])
        history["music_metrics"].append(music_metrics)

        print(f"Epoch {epoch + 1}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Acc: {val_acc:.4f}")
        print(f"  Val PPL: {val_ppl:.2f}")
        print(
            f"  Music: repeat={music_metrics['repeat_rate']:.3f}, "
            f"diversity={music_metrics['unique_token_ratio']:.3f}, "
            f"rhythm_div={music_metrics['rhythm_diversity']:.3f}, "
            f"scale_cov={music_metrics['scale_coverage']:.3f}"
        )
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "model_config": {**model_config, "vocab_size": vocab_size},
                },
                CHECKPOINT_DIR / "model_best.pth",
            )
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping activated.")
                break

    torch.save(model.state_dict(), CHECKPOINT_DIR / "model_final.pth")
    with open(CHECKPOINT_DIR / "history.json", "w") as handle:
        json.dump(history, handle, indent=2)

    best_cp = CHECKPOINT_DIR / "model_best.pth"
    if best_cp.exists():
        checkpoint = torch.load(best_cp, map_location=DEVICE)
        model.load_state_dict(clean_state_dict(checkpoint["model_state_dict"]))
        test_loss, test_acc, test_ppl, test_music_metrics = evaluate(model, test_loader, pad_id, id2token, amp_enabled)
        print("Test metrics (best checkpoint):")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Acc: {test_acc:.4f}")
        print(f"  Test PPL: {test_ppl:.2f}")
        print(
            f"  Music Test: repeat={test_music_metrics['repeat_rate']:.3f}, "
            f"diversity={test_music_metrics['unique_token_ratio']:.3f}, "
            f"rhythm_div={test_music_metrics['rhythm_diversity']:.3f}, "
            f"scale_cov={test_music_metrics['scale_coverage']:.3f}"
        )

    save_plots(history)


if __name__ == "__main__":
    main()
