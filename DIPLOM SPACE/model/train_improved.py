import json
import math
import random
import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

mpl_cache_dir = PROJECT_ROOT / ".cache" / "matplotlib"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

from model.dataset import MIDIDataset
from model.transformer import TransformerLM

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# =========================
# CONFIG (редактируй здесь)
# =========================
SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

NUM_EPOCHS = 120
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

# Количество сэмплов на эпоху.
# Для реального обучения увеличивай до 12000-20000.
SAMPLES_PER_EPOCH = 12000

EARLY_STOPPING_PATIENCE = 8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
SAMPLING_STRATEGY = "mixed"  # "balanced" | "mixed"

# Стабилизация role-weights
ROLE_WEIGHT_ALPHA = 0.2
MAX_ROLE_WEIGHT = 2.0

VOCAB_PATH = PROJECT_ROOT / "dataset" / "processed" / "vocab.json"
CHUNKS_DIR = PROJECT_ROOT / "dataset" / "processed" / "chunks"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
PLOTS_DIR = CHECKPOINT_DIR / "plots"

# Дообучение с лучшего чекпоинта
RESUME_FROM_CHECKPOINT = True
RESUME_CHECKPOINT_PATH = CHECKPOINT_DIR / "model_best.pth"
LOAD_OPTIMIZER_STATE = False

MAX_LEN = 512
# =========================


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "mps":
        torch.mps.manual_seed(seed)


def build_loaders():
    dataset = MIDIDataset(
        chunks_dir=str(CHUNKS_DIR),
        vocab_path=str(VOCAB_PATH),
        max_len=MAX_LEN,
        samples_per_epoch=SAMPLES_PER_EPOCH,
        sampling_strategy=SAMPLING_STRATEGY,
        seed=SEED,
    )

    role_counts = {
        "MELODY": len(dataset.data["melody"]),
        "BASS": len(dataset.data["bass"]),
        "CHORDS": len(dataset.data["chords"]),
    }

    if VAL_SPLIT + TEST_SPLIT >= 1.0:
        raise ValueError("VAL_SPLIT + TEST_SPLIT must be < 1.0")

    total = len(dataset)
    val_size = max(1, int(total * VAL_SPLIT))
    test_size = max(1, int(total * TEST_SPLIT))
    train_size = total - val_size - test_size
    if train_size <= 0:
        raise ValueError("Train split became empty")

    indices = list(range(total))
    random.shuffle(indices)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)

    return dataset, role_counts, train_loader, val_loader, test_loader


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


def evaluate(model, loader, vocab_size, pad_id, role_token_ids, id2token):
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
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            batch_size, seq_len, _ = logits.shape

            token_losses = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                reduction="none",
                ignore_index=pad_id
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
        print("⚠️ matplotlib не установлен, графики пропущены")
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
    print(f"🖥️  Device: {DEVICE}")

    dataset, role_counts, train_loader, val_loader, test_loader = build_loaders()
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

    model = TransformerLM(vocab_size, pad_id=pad_id).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    role_token_ids = {
        "MELODY": token2id["<ROLE_MELODY>"],
        "BASS": token2id["<ROLE_BASS>"],
        "CHORDS": token2id["<ROLE_CHORDS>"],
    }
    role_weights = build_role_weights(role_counts)
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
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if LOAD_OPTIMIZER_STATE and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = int(checkpoint.get("epoch", 0))
            best_val_loss = float(checkpoint.get("val_loss", best_val_loss))
        else:
            model.load_state_dict(checkpoint)
        print(f"🔄 Resume from checkpoint: {RESUME_CHECKPOINT_PATH} (start_epoch={start_epoch})")

    history_path = CHECKPOINT_DIR / "history.json"
    if RESUME_FROM_CHECKPOINT and history_path.exists():
        try:
            with open(history_path) as f:
                loaded_history = json.load(f)
            for key in history:
                if key in loaded_history and isinstance(loaded_history[key], list):
                    history[key] = loaded_history[key]
            print(f"📚 История подгружена: {history_path}")
        except Exception:
            print("⚠️ Не удалось загрузить history.json, создаю новую историю")

    print("\n🚀 Начинаем обучение...\n")
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            model.train()
            total_train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [TRAIN]")

            for x, y in pbar:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                logits = model(x)
                loss = weighted_loss(logits, y, role_token_ids, role_weights, pad_id)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_train_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = total_train_loss / max(1, len(train_loader))
            avg_val_loss, val_acc, val_ppl, role_metrics, music_metrics = evaluate(
                model, val_loader, vocab_size, pad_id, role_token_ids, id2token
            )

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
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": avg_val_loss,
                    },
                    str(CHECKPOINT_DIR / "model_best.pth"),
                )
                print(f"💾 Лучшая модель сохранена (Val Loss: {avg_val_loss:.4f})\n")
            else:
                early_stop_counter += 1
                print(f"⚠️  Val Loss не улучшилась ({early_stop_counter}/{EARLY_STOPPING_PATIENCE})\n")

            scheduler.step(avg_val_loss)
            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                print("🛑 Early stopping активирован!")
                break

        print("\n✅ Обучение завершено!")
        torch.save(model.state_dict(), str(CHECKPOINT_DIR / "model_final.pth"))
        print("💾 Финальная модель сохранена")

        with open(CHECKPOINT_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        print("📊 История обучения сохранена")

        best_cp = CHECKPOINT_DIR / "model_best.pth"
        if best_cp.exists():
            checkpoint = torch.load(best_cp, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
            test_loss, test_acc, test_ppl, test_role_metrics, test_music_metrics = evaluate(
                model, test_loader, vocab_size, pad_id, role_token_ids, id2token
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
        print(f"📈 Графики сохранены в {PLOTS_DIR}")

    except KeyboardInterrupt:
        print("\n⚠️  Обучение прервано пользователем")
        torch.save(model.state_dict(), str(CHECKPOINT_DIR / "model_interrupted.pth"))
        print("💾 Модель сохранена как model_interrupted.pth")


if __name__ == "__main__":
    main()
