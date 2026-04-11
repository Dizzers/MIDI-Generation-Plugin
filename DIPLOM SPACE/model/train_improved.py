import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Matplotlib cache (Kaggle input dirs are read-only)
try:
    mpl_cache_dir = PROJECT_ROOT / ".cache" / "matplotlib"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))
except OSError:
    tmp_root = Path(os.environ.get("TMPDIR", "/tmp"))
    mpl_cache_dir = tmp_root / "matplotlib"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

from model.dataset import MIDIDataset
from model.transformer import TransformerLM

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


SEED = 42
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

NUM_EPOCHS = 80
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01

SAMPLES_PER_EPOCH = 12000
GRAD_ACCUM_STEPS = 1
LABEL_SMOOTHING = 0.05

EARLY_STOPPING_PATIENCE = 8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
SAMPLING_STRATEGY = "role_cap"
ROLE_SAMPLING_MAX_FRACTION = {
    "chords": 0.40,
    "melody": 0.35,
    "bass": 0.25,
}

ROLE_WEIGHT_ALPHA = 0.8
MAX_ROLE_WEIGHT = 10.0

AUGMENT_CONFIG = {
    "transpose_prob": 0.5,
    "transpose_range": 5,
    "time_stretch_prob": 0.35,
    "time_stretch_range": (0.9, 1.1),
    "velocity_jitter_prob": 0.35,
    "velocity_jitter": 1,
}

VOCAB_PATH = PROJECT_ROOT / "dataset" / "processed" / "vocab.json"
CHUNKS_DIR = PROJECT_ROOT / "dataset" / "processed" / "chunks"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
PLOTS_DIR = CHECKPOINT_DIR / "plots"

RESUME_FROM_CHECKPOINT = False
RESUME_CHECKPOINT_PATH = CHECKPOINT_DIR / "model_best.pth"
LOAD_OPTIMIZER_STATE = False
ALLOW_OLD_CHECKPOINT_RESUME = False

MAX_LEN = 512

D_MODEL = 256
N_HEADS = 8
N_LAYERS = 6
D_FF = 1024
DROPOUT = 0.20

USE_DDP = False  
DDP_BACKEND = "nccl"
ENABLE_DATA_PARALLEL = True 
USE_AMP = True
NUM_WORKERS = 4
PIN_MEMORY = DEVICE == "cuda"
PERSISTENT_WORKERS = True


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
    if DEVICE == "mps":
        torch.mps.manual_seed(seed)


def setup_distributed():
    if not USE_DDP:
        return False, 0, 1, 0
    if DEVICE != "cuda":
        return False, 0, 1, 0
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 1, 0
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=DDP_BACKEND)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return True, local_rank, world_size, rank


def is_main_process(use_ddp, rank):
    return (not use_ddp) or rank == 0


def unwrap_model(model):
    return model.module if isinstance(model, (nn.DataParallel, DDP)) else model


def clean_state_dict(state_dict):
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def is_compatible_checkpoint(checkpoint, model_config, vocab_size):
    if not isinstance(checkpoint, dict):
        return False, "checkpoint is not a dict"
    if "model_config" not in checkpoint:
        return False, "checkpoint has no model_config"
    ckpt_cfg = checkpoint["model_config"]
    for key in ["d_model", "n_heads", "n_layers", "d_ff", "dropout", "max_len", "num_roles", "num_genres"]:
        if ckpt_cfg.get(key) != model_config.get(key):
            return False, f"model_config mismatch on {key}"
    if ckpt_cfg.get("pad_id") != model_config.get("pad_id"):
        return False, "model_config mismatch on pad_id"
    if ckpt_cfg.get("vocab_size") is not None and ckpt_cfg.get("vocab_size") != vocab_size:
        return False, "model_config mismatch on vocab_size"
    return True, "compatible"


def build_loaders(use_ddp=False, rank=0, world_size=1):
    base_dataset = MIDIDataset(
        chunks_dir=str(CHUNKS_DIR),
        vocab_path=str(VOCAB_PATH),
        max_len=MAX_LEN,
        samples_per_epoch=SAMPLES_PER_EPOCH,
        sampling_strategy=SAMPLING_STRATEGY,
        seed=SEED + rank,
        augment_config=AUGMENT_CONFIG,
        role_sampling_max_fraction=ROLE_SAMPLING_MAX_FRACTION,
    )

    if VAL_SPLIT + TEST_SPLIT >= 1.0:
        raise ValueError("VAL_SPLIT + TEST_SPLIT must be < 1.0")

    split_rng = random.Random(SEED)
    split_data = {"train": {}, "val": {}, "test": {}}
    role_counts = {}

    for role in ["melody", "bass", "chords"]:
        role_items = list(base_dataset.data[role])
        role_counts[role.upper()] = len(role_items)
        role_indices = list(range(len(role_items)))
        split_rng.shuffle(role_indices)

        val_size = max(1, int(len(role_indices) * VAL_SPLIT)) if len(role_indices) > 2 else 0
        test_size = max(1, int(len(role_indices) * TEST_SPLIT)) if len(role_indices) - val_size > 1 else 0
        train_size = len(role_indices) - val_size - test_size

        if train_size <= 0:
            train_size = max(1, len(role_indices) - 2)
            remaining = len(role_indices) - train_size
            val_size = 1 if remaining > 0 else 0
            test_size = 1 if remaining > 1 else 0

        train_idx = role_indices[:train_size]
        val_idx = role_indices[train_size:train_size + val_size]
        test_idx = role_indices[train_size + val_size:train_size + val_size + test_size]

        split_data["train"][role] = [role_items[i] for i in train_idx]
        split_data["val"][role] = [role_items[i] for i in val_idx]
        split_data["test"][role] = [role_items[i] for i in test_idx]

    train_dataset = base_dataset.clone_with_data(
        split_data["train"],
        samples_per_epoch=SAMPLES_PER_EPOCH,
        apply_augmentation=True,
        seed_offset=11,
    )
    val_dataset = base_dataset.clone_with_data(
        split_data["val"],
        samples_per_epoch=None,
        apply_augmentation=False,
        seed_offset=22,
    )
    test_dataset = base_dataset.clone_with_data(
        split_data["test"],
        samples_per_epoch=None,
        apply_augmentation=False,
        seed_offset=33,
    )

    train_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
        "persistent_workers": PERSISTENT_WORKERS and NUM_WORKERS > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_dataset, role_counts, train_loader, val_loader, test_loader, train_sampler


def build_role_weights(role_counts):
    max_count = max(role_counts.values())
    raw = {
        role: min((max_count / max(1, count)) ** ROLE_WEIGHT_ALPHA, MAX_ROLE_WEIGHT)
        for role, count in role_counts.items()
    }
    mean_w = sum(raw.values()) / len(raw)
    return {k: v / mean_w for k, v in raw.items()}


def weighted_loss(logits, targets, role_token_ids, role_weights, pad_id):
    batch_size, seq_len, vocab_size = logits.shape
    token_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1),
        reduction="none",
        ignore_index=pad_id,
        label_smoothing=LABEL_SMOOTHING,
    ).view(batch_size, seq_len)

    valid_mask = (targets != pad_id).float()
    token_count = valid_mask.sum(dim=1).clamp_min(1.0)
    sample_loss = (token_loss * valid_mask).sum(dim=1) / token_count

    role_ids = targets[:, 0]
    loss_weights = torch.ones(batch_size, device=logits.device)
    for role_name, role_token_id in role_token_ids.items():
        mask = role_ids == role_token_id
        loss_weights = torch.where(mask, torch.full_like(loss_weights, role_weights[role_name]), loss_weights)

    return (sample_loss * loss_weights).mean()


def best_scale_coverage(note_pitches):
    if not note_pitches:
        return 0.0
    pitch_classes = [p % 12 for p in note_pitches]
    major = {0, 2, 4, 5, 7, 9, 11}
    minor = {0, 2, 3, 5, 7, 8, 10}
    best = 0.0
    for root in range(12):
        maj_set = {(n + root) % 12 for n in major}
        min_set = {(n + root) % 12 for n in minor}
        maj_cov = sum(pc in maj_set for pc in pitch_classes) / len(pitch_classes)
        min_cov = sum(pc in min_set for pc in pitch_classes) / len(pitch_classes)
        best = max(best, maj_cov, min_cov)
    return best


def sequence_music_metrics(token_ids, id2token, pad_id):
    ids = [int(i) for i in token_ids if int(i) != pad_id]
    tokens = [id2token.get(i, "<UNK>") for i in ids]
    if len(tokens) < 2:
        return {
            "repeat_rate": 0.0,
            "unique_token_ratio": 0.0,
            "note_density": 0.0,
            "pitch_range": 0.0,
            "rhythm_diversity": 0.0,
            "scale_coverage": 0.0,
        }

    repeats = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i - 1])
    repeat_rate = repeats / (len(tokens) - 1)
    unique_token_ratio = len(set(tokens)) / len(tokens)

    note_pitches = []
    time_shifts = []
    for tok in tokens:
        if tok.startswith("NOTE_ON_"):
            try:
                note_pitches.append(int(tok.split("_")[2], 16))
            except Exception:
                pass
        elif tok.startswith("TIME_SHIFT_"):
            try:
                time_shifts.append(int(tok.split("_")[2], 16))
            except Exception:
                pass

    total_time = max(1.0, sum(time_shifts) * 0.05)
    note_density = len(note_pitches) / total_time
    pitch_range = (max(note_pitches) - min(note_pitches)) / 127.0 if note_pitches else 0.0
    rhythm_diversity = len(set(time_shifts)) / max(1, len(time_shifts))
    scale_coverage = best_scale_coverage(note_pitches)

    return {
        "repeat_rate": repeat_rate,
        "unique_token_ratio": unique_token_ratio,
        "note_density": note_density,
        "pitch_range": pitch_range,
        "rhythm_diversity": rhythm_diversity,
        "scale_coverage": scale_coverage,
    }


def evaluate(model, loader, vocab_size, pad_id, role_token_ids, id2token, amp_enabled):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    total_tokens = 0

    role_stats = {
        "MELODY": {"loss_sum": 0.0, "samples": 0, "correct": 0, "tokens": 0},
        "BASS": {"loss_sum": 0.0, "samples": 0, "correct": 0, "tokens": 0},
        "CHORDS": {"loss_sum": 0.0, "samples": 0, "correct": 0, "tokens": 0},
    }
    music_agg = {
        "repeat_rate": 0.0,
        "unique_token_ratio": 0.0,
        "note_density": 0.0,
        "pitch_range": 0.0,
        "rhythm_diversity": 0.0,
        "scale_coverage": 0.0,
    }
    music_count = 0

    with torch.no_grad():
        for x, y, role_idx, genre_idx in loader:
            x = x.to(DEVICE, non_blocking=PIN_MEMORY)
            y = y.to(DEVICE, non_blocking=PIN_MEMORY)
            role_idx = role_idx.to(DEVICE, non_blocking=PIN_MEMORY)
            genre_idx = genre_idx.to(DEVICE, non_blocking=PIN_MEMORY)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(x, role_id=role_idx, genre_id=genre_idx)

            batch_size, seq_len, _ = logits.shape
            token_losses = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                reduction="none",
                ignore_index=pad_id,
                label_smoothing=LABEL_SMOOTHING,
            ).view(batch_size, seq_len)

            valid_mask = (y != pad_id)
            token_count = valid_mask.sum(dim=1).clamp_min(1)
            sample_losses = (token_losses * valid_mask.float()).sum(dim=1) / token_count.float()

            total_loss += sample_losses.sum().item()
            total_samples += batch_size

            preds = logits.argmax(dim=-1)
            total_correct += ((preds == y) & valid_mask).sum().item()
            total_tokens += valid_mask.sum().item()

            role_ids = y[:, 0]
            for role_name, role_token_id in role_token_ids.items():
                mask = role_ids == role_token_id
                if mask.any():
                    role_stats[role_name]["loss_sum"] += sample_losses[mask].sum().item()
                    role_stats[role_name]["samples"] += mask.sum().item()
                    role_valid = valid_mask[mask]
                    role_preds = preds[mask]
                    role_targets = y[mask]
                    role_stats[role_name]["correct"] += ((role_preds == role_targets) & role_valid).sum().item()
                    role_stats[role_name]["tokens"] += role_valid.sum().item()

            for i in range(batch_size):
                pred_ids = preds[i][valid_mask[i]].detach().cpu().tolist()
                mm = sequence_music_metrics(pred_ids, id2token, pad_id)
                for k in music_agg:
                    music_agg[k] += mm[k]
                music_count += 1

    model.train()
    avg_loss = total_loss / max(1, total_samples)
    val_acc = total_correct / max(1, total_tokens)
    val_ppl = math.exp(min(avg_loss, 20.0))

    role_metrics = {}
    for role_name, stats in role_stats.items():
        role_metrics[role_name] = {
            "loss": stats["loss_sum"] / max(1, stats["samples"]),
            "acc": stats["correct"] / max(1, stats["tokens"]),
            "samples": stats["samples"],
        }

    music_metrics = {k: v / max(1, music_count) for k, v in music_agg.items()}
    return avg_loss, val_acc, val_ppl, role_metrics, music_metrics


def save_plots(history):
    if plt is None:
        print(" matplotlib не установлен, графики пропущены")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "loss_curve.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "val_acc_curve.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_ppl"], label="val_ppl")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Validation Perplexity")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "val_ppl_curve.png", dpi=140)
    plt.close()

    for metric_name in ["repeat_rate", "unique_token_ratio", "note_density", "pitch_range", "rhythm_diversity", "scale_coverage"]:
        plt.figure(figsize=(8, 5))
        values = [m[metric_name] for m in history["music_metrics"]]
        plt.plot(epochs, values, label=metric_name)
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(f"Music Metric: {metric_name}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{metric_name}_curve.png", dpi=140)
        plt.close()


def main():
    print("📦 Загружаем компоненты...")
    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    use_ddp, local_rank, world_size, rank = setup_distributed()
    if use_ddp:
        global DEVICE
        DEVICE = f"cuda:{local_rank}"
    is_main = is_main_process(use_ddp, rank)
    if is_main:
        print(f"🖥️  Device: {DEVICE}")

    dataset, role_counts, train_loader, val_loader, test_loader, train_sampler = build_loaders(
        use_ddp=use_ddp,
        rank=rank,
        world_size=world_size,
    )

    if is_main:
        print("✓ Датасет загружен:")
        print(f"  - Train: {len(train_loader.dataset)}")
        print(f"  - Val: {len(val_loader.dataset)}")
        print(f"  - Test: {len(test_loader.dataset)}")
        print(
            f"  - Role chunks: MELODY={role_counts['MELODY']}, "
            f"BASS={role_counts['BASS']}, CHORDS={role_counts['CHORDS']}"
        )

    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    token2id = vocab["token2id"]
    id2token = {int(k): v for k, v in vocab["id2token"].items()}
    vocab_size = len(token2id)
    pad_id = token2id["<PAD>"]

    num_roles = 3
    num_genres = dataset.num_genres

    model_init_config = {
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "d_ff": D_FF,
        "dropout": DROPOUT,
        "max_len": MAX_LEN,
        "pad_id": pad_id,
        "num_roles": num_roles,
        "num_genres": num_genres,
    }
    model_config = {**model_init_config, "vocab_size": vocab_size}

    model = TransformerLM(vocab_size, **model_init_config).to(DEVICE)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank])
    elif DEVICE == "cuda" and ENABLE_DATA_PARALLEL and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if is_main:
            print(f"DataParallel enabled: {torch.cuda.device_count()} GPUs")

    optimizer = torch.optim.AdamW(
        unwrap_model(model).parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    amp_enabled = USE_AMP and DEVICE.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    role_token_ids = {
        "MELODY": token2id["<ROLE_MELODY>"],
        "BASS": token2id["<ROLE_BASS>"],
        "CHORDS": token2id["<ROLE_CHORDS>"]
    }
    role_weights = build_role_weights(role_counts)
    if is_main:
        print(
            f"⚖️  Role weights: MELODY={role_weights['MELODY']:.2f}, "
            f"BASS={role_weights['BASS']:.2f}, CHORDS={role_weights['CHORDS']:.2f}"
        )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_ppl": [],
        "learning_rates": [],
        "role_metrics": [],
        "music_metrics": [],
    }

    best_val_loss = float("inf")
    early_stop_counter = 0
    start_epoch = 0

    if RESUME_FROM_CHECKPOINT and RESUME_CHECKPOINT_PATH.exists():
        checkpoint = torch.load(RESUME_CHECKPOINT_PATH, map_location=DEVICE)
        ok, reason = is_compatible_checkpoint(checkpoint, model_config, vocab_size)
        if ok:
            unwrap_model(model).load_state_dict(clean_state_dict(checkpoint["model_state_dict"]))
            if LOAD_OPTIMIZER_STATE and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = int(checkpoint.get("epoch", 0))
            best_val_loss = float(checkpoint.get("val_loss", best_val_loss))
            if is_main:
                print(f"🔄 Resume from checkpoint: {RESUME_CHECKPOINT_PATH} (start_epoch={start_epoch})")
        elif ALLOW_OLD_CHECKPOINT_RESUME:
            try:
                state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
                unwrap_model(model).load_state_dict(clean_state_dict(state_dict))
                if is_main:
                    print("⚠️ Loaded old checkpoint without model_config. Use with caution.")
            except Exception as exc:
                if is_main:
                    print(f"⚠️ Skipping resume: incompatible checkpoint ({exc})")
        else:
            if is_main:
                print(f"⚠️ Skipping resume: incompatible checkpoint ({reason})")

    history_path = CHECKPOINT_DIR / "history.json"
    if RESUME_FROM_CHECKPOINT and history_path.exists() and is_main:
        try:
            with open(history_path) as f:
                loaded_history = json.load(f)
            for key in history:
                if key in loaded_history and isinstance(loaded_history[key], list):
                    history[key] = loaded_history[key]
            print(f"📚 История подгружена: {history_path}")
        except Exception:
            print(" Не удалось загрузить history.json, создаю новую историю")

    if is_main:
        print("\n Начинаем обучение...\n")

    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            model.train()
            total_train_loss = 0.0

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [TRAIN]") if is_main else train_loader

            optimizer.zero_grad(set_to_none=True)
            step_in_epoch = 0

            for x, y, role_idx, genre_idx in pbar:
                x = x.to(DEVICE, non_blocking=PIN_MEMORY)
                y = y.to(DEVICE, non_blocking=PIN_MEMORY)
                role_idx = role_idx.to(DEVICE, non_blocking=PIN_MEMORY)
                genre_idx = genre_idx.to(DEVICE, non_blocking=PIN_MEMORY)

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    logits = model(x, role_id=role_idx, genre_id=genre_idx)
                    loss = weighted_loss(logits, y, role_token_ids, role_weights, pad_id)
                    loss = loss / max(1, GRAD_ACCUM_STEPS)

                scaler.scale(loss).backward()
                step_in_epoch += 1

                if step_in_epoch % GRAD_ACCUM_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                total_train_loss += loss.item() * max(1, GRAD_ACCUM_STEPS)
                if is_main:
                    pbar.set_postfix({"loss": f"{loss.item() * max(1, GRAD_ACCUM_STEPS):.4f}"})

            if step_in_epoch % GRAD_ACCUM_STEPS != 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            avg_train_loss = total_train_loss / max(1, len(train_loader))

            if is_main:
                avg_val_loss, val_acc, val_ppl, role_metrics, music_metrics = evaluate(
                    model, val_loader, vocab_size, pad_id, role_token_ids, id2token, amp_enabled
                )
            else:
                avg_val_loss, val_acc, val_ppl, role_metrics, music_metrics = 0.0, 0.0, 0.0, {}, {}

            if use_ddp:
                loss_tensor = torch.tensor([avg_val_loss], device=DEVICE, dtype=torch.float32)
                dist.broadcast(loss_tensor, src=0)
                avg_val_loss = float(loss_tensor.item())

            scheduler.step(avg_val_loss)

            if is_main:
                history["train_loss"].append(avg_train_loss)
                history["val_loss"].append(avg_val_loss)
                history["val_acc"].append(val_acc)
                history["val_ppl"].append(val_ppl)
                history["learning_rates"].append(optimizer.param_groups[0]["lr"])
                history["role_metrics"].append(role_metrics)
                history["music_metrics"].append(music_metrics)

                print(f"✓ Epoch {epoch + 1}")
                print(f"  Train Loss: {avg_train_loss:.4f}")
                print(f"  Val Loss: {avg_val_loss:.4f}")
                print(f"  Val Acc: {val_acc:.4f}")
                print(f"  Val PPL: {val_ppl:.2f}")
                print(
                    f"  Role Val Acc: M={role_metrics['MELODY']['acc']:.3f}, "
                    f"B={role_metrics['BASS']['acc']:.3f}, C={role_metrics['CHORDS']['acc']:.3f}"
                )
                print(
                    f"  Music: repeat={music_metrics['repeat_rate']:.3f}, "
                    f"diversity={music_metrics['unique_token_ratio']:.3f}, "
                    f"scale_cov={music_metrics['scale_coverage']:.3f}"
                )
                print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}\n")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    early_stop_counter = 0
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": unwrap_model(model).state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": avg_val_loss,
                            "model_config": model_config,
                        },
                        str(CHECKPOINT_DIR / "model_best.pth"),
                    )
                    print(f"💾 Лучшая модель сохранена (Val Loss: {avg_val_loss:.4f})\n")
                else:
                    early_stop_counter += 1
                    print(f"⚠️  Val Loss не улучшилась ({early_stop_counter}/{EARLY_STOPPING_PATIENCE})\n")

            if use_ddp:
                stop_tensor = torch.tensor([1 if early_stop_counter >= EARLY_STOPPING_PATIENCE else 0], device=DEVICE)
                dist.broadcast(stop_tensor, src=0)
                if stop_tensor.item() == 1:
                    if is_main:
                        print(" Early stopping активирован!")
                    break
            else:
                if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                    if is_main:
                        print(" Early stopping активирован!")
                    break

        if is_main:
            print("\n Обучение завершено!")
            torch.save(unwrap_model(model).state_dict(), str(CHECKPOINT_DIR / "model_final.pth"))
            print(" Финальная модель сохранена")

            with open(CHECKPOINT_DIR / "history.json", "w") as f:
                json.dump(history, f, indent=2)
            print(" История обучения сохранена")

            best_cp = CHECKPOINT_DIR / "model_best.pth"
            if best_cp.exists():
                checkpoint = torch.load(best_cp, map_location=DEVICE)
                unwrap_model(model).load_state_dict(clean_state_dict(checkpoint["model_state_dict"]))
                test_loss, test_acc, test_ppl, test_role_metrics, test_music_metrics = evaluate(
                    model, test_loader, vocab_size, pad_id, role_token_ids, id2token, amp_enabled
                )
                print("\n🧪 Test метрики (best checkpoint):")
                print(f"  Test Loss: {test_loss:.4f}")
                print(f"  Test Acc: {test_acc:.4f}")
                print(f"  Test PPL: {test_ppl:.2f}")
                print(
                    f"  Role Test Acc: M={test_role_metrics['MELODY']['acc']:.3f}, "
                    f"B={test_role_metrics['BASS']['acc']:.3f}, "
                    f"C={test_role_metrics['CHORDS']['acc']:.3f}"
                )
                print(
                    f"  Music Test: repeat={test_music_metrics['repeat_rate']:.3f}, "
                    f"diversity={test_music_metrics['unique_token_ratio']:.3f}, "
                    f"scale_cov={test_music_metrics['scale_coverage']:.3f}"
                )

            save_plots(history)
            print(f"Графики сохранены в {PLOTS_DIR}")

    except KeyboardInterrupt:
        if is_main:
            print("\n Обучение прервано пользователем")
            torch.save(unwrap_model(model).state_dict(), str(CHECKPOINT_DIR / "model_interrupted.pth"))
            print("Модель сохранена как model_interrupted.pth")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
