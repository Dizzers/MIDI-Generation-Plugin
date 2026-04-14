import json
import math
import os
import random
import sys
from pathlib import Path

import matplotlib
mpl_cache_dir = Path(os.environ.get("MPLCONFIGDIR", ""))
if not str(mpl_cache_dir).strip():
    mpl_cache_dir = Path("/tmp/matplotlib_cache")
os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.transformer import TransformerLM, count_parameters

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

SEED = 42
NUM_EPOCHS = 80
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
SAMPLES_PER_EPOCH = 8000
EARLY_STOPPING_PATIENCE = 8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
MAX_LEN = 256
LABEL_SMOOTHING = 0.05

AUGMENT_CONFIG = {
    "transpose_prob": 0.35,
    "transpose_range": 4,
    "time_stretch_prob": 0.25,
    "time_stretch_range": (0.93, 1.07),
    "velocity_jitter_prob": 0.25,
    "velocity_jitter": 1,
}

VOCAB_PATH = PROJECT_ROOT / "dataset" / "processed" / "vocab_full.json"
CHUNKS_PATH = PROJECT_ROOT / "dataset" / "processed" / "chunks" / "full_chunks.npy"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
PLOTS_DIR = CHECKPOINT_DIR / "plots_full"

RESUME_FROM_CHECKPOINT = False
RESUME_CHECKPOINT_PATH = CHECKPOINT_DIR / "model_best_full.pth"

D_MODEL = 256
N_HEADS = 8
N_LAYERS = 6
D_FF = 1024
DROPOUT = 0.2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


def parse_hex_token(token):
    try:
        parts = token.split("_")
        if len(parts) >= 3:
            return int(parts[2], 16)
        if len(parts) == 2:
            return int(parts[1], 16)
    except Exception:
        pass
    return None


class FullMIDIDataset(Dataset):
    def __init__(self, data, token2id, genre_to_index, max_len=256, augment=False, augment_cfg=None, seed=42):
        self.data = np.array(data, dtype=object)
        self.token2id = token2id
        self.genre_to_index = genre_to_index
        self.max_len = max_len
        self.augment = augment
        self.augment_cfg = augment_cfg or {}
        self.rng = random.Random(seed)

        self.pad = token2id["<PAD>"]
        self.bos = token2id["<BOS>"]
        self.eos = token2id["<EOS>"]
        self.unk = token2id["<UNK>"]

        self.time_shift_values = sorted({parse_hex_token(t) for t in token2id if t.startswith("TIME_SHIFT_")})
        self.time_shift_values = [v for v in self.time_shift_values if v is not None]
        self.velocity_values = sorted({parse_hex_token(t) for t in token2id if t.startswith("VELOCITY_")})
        self.velocity_values = [v for v in self.velocity_values if v is not None]

    def __len__(self):
        return len(self.data)

    def encode(self, tokens):
        ids = [self.bos]
        for t in tokens:
            ids.append(self.token2id.get(t, self.unk))
        ids.append(self.eos)
        return ids[:self.max_len]

    def _nearest(self, values, target):
        if not values:
            return target
        return min(values, key=lambda x: abs(x - target))

    def _shift_note_token(self, tok, semitones):
        if not (tok.startswith("NOTE_ON_") or tok.startswith("NOTE_OFF_")):
            return tok
        try:
            p = int(tok.split("_")[2], 16)
            p2 = max(0, min(127, p + semitones))
            return f"{tok.split('_')[0]}_{tok.split('_')[1]}_{p2:02X}"
        except Exception:
            return tok

    def _stretch_time_token(self, tok, factor):
        if not tok.startswith("TIME_SHIFT_"):
            return tok
        try:
            val = int(tok.split("_")[2], 16)
            new_val = max(1, int(round(val * factor)))
            new_val = self._nearest(self.time_shift_values, new_val)
            return f"TIME_SHIFT_{new_val:02X}"
        except Exception:
            return tok

    def _jitter_velocity_token(self, tok, jitter):
        if not tok.startswith("VELOCITY_"):
            return tok
        try:
            val = int(tok.split("_")[1], 16)
            maxv = max(self.velocity_values) if self.velocity_values else 7
            v2 = max(0, min(maxv, val + jitter))
            return f"VELOCITY_{v2:02X}"
        except Exception:
            return tok

    def _apply_augmentation(self, tokens):
        out = list(tokens)
        if not self.augment:
            return out

        if self.rng.random() < self.augment_cfg.get("transpose_prob", 0.0):
            tr = int(self.augment_cfg.get("transpose_range", 0))
            if tr > 0:
                sem = self.rng.randint(-tr, tr)
                if sem != 0:
                    out = [self._shift_note_token(t, sem) for t in out]

        if self.rng.random() < self.augment_cfg.get("time_stretch_prob", 0.0):
            lo, hi = self.augment_cfg.get("time_stretch_range", (1.0, 1.0))
            if lo != 1.0 or hi != 1.0:
                fac = self.rng.uniform(lo, hi)
                out = [self._stretch_time_token(t, fac) for t in out]

        if self.rng.random() < self.augment_cfg.get("velocity_jitter_prob", 0.0):
            jit = int(self.augment_cfg.get("velocity_jitter", 0))
            if jit > 0:
                j = self.rng.randint(-jit, jit)
                if j != 0:
                    out = [self._jitter_velocity_token(t, j) for t in out]

        return out

    def _extract_genre_index(self, tokens):
        for t in tokens:
            if t.startswith("<GENRE_"):
                return self.genre_to_index.get(t, 0)
        return 0

    def __getitem__(self, idx):
        tokens = list(self.data[idx])
        tokens = self._apply_augmentation(tokens)

        genre_idx = self._extract_genre_index(tokens)
        ids = self.encode(tokens)
        if len(ids) < self.max_len:
            ids += [self.pad] * (self.max_len - len(ids))

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        genre_idx = torch.tensor(genre_idx, dtype=torch.long)
        return x, y, genre_idx


def build_datasets(token2id):
    chunks = np.load(CHUNKS_PATH, allow_pickle=True)
    idx = np.arange(len(chunks))
    np.random.shuffle(idx)
    chunks = chunks[idx]

    n_total = len(chunks)
    n_val = max(1, int(n_total * VAL_SPLIT))
    n_test = max(1, int(n_total * TEST_SPLIT))
    n_train = max(1, n_total - n_val - n_test)

    train_raw = chunks[:n_train]
    val_raw = chunks[n_train:n_train + n_val]
    test_raw = chunks[n_train + n_val:]

    # Sample train with replacement each epoch for stable batch count.
    if SAMPLES_PER_EPOCH > 0:
        rng = np.random.default_rng(SEED)
        pick = rng.choice(len(train_raw), size=SAMPLES_PER_EPOCH, replace=True)
        train_raw = train_raw[pick]

    genre_tokens = sorted([t for t in token2id if t.startswith("<GENRE_")])
    genre_to_index = {t: i for i, t in enumerate(genre_tokens)}

    train_ds = FullMIDIDataset(train_raw, token2id, genre_to_index, max_len=MAX_LEN, augment=True, augment_cfg=AUGMENT_CONFIG, seed=SEED)
    val_ds = FullMIDIDataset(val_raw, token2id, genre_to_index, max_len=MAX_LEN, augment=False, seed=SEED + 1)
    test_ds = FullMIDIDataset(test_raw, token2id, genre_to_index, max_len=MAX_LEN, augment=False, seed=SEED + 2)

    return train_ds, val_ds, test_ds, len(genre_tokens)


def token_loss(logits, targets, pad_id):
    b, t, v = logits.shape
    losses = F.cross_entropy(
        logits.view(-1, v),
        targets.view(-1),
        reduction="none",
        ignore_index=pad_id,
        label_smoothing=LABEL_SMOOTHING,
    ).view(b, t)

    valid = targets != pad_id
    prefix_len = min(2, t)  # <GENRE_...>, <KEY_...>
    valid[:, :prefix_len] = False

    denom = valid.sum(dim=1).clamp_min(1).float()
    per_sample = (losses * valid.float()).sum(dim=1) / denom
    return per_sample.mean(), valid


def sequence_music_metrics(pred_ids, id2token, pad_id):
    ids = [int(i) for i in pred_ids if int(i) != pad_id]
    tokens = [id2token.get(i, "<UNK>") for i in ids]
    if len(tokens) < 2:
        return {"repeat": 0.0, "diversity": 0.0, "scale_cov": 0.0}

    reps = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i - 1])
    repeat = reps / (len(tokens) - 1)
    diversity = len(set(tokens)) / len(tokens)

    pitches = []
    for t in tokens:
        if t.startswith("NOTE_ON_"):
            try:
                pitches.append(int(t.split("_")[2], 16))
            except Exception:
                pass

    if not pitches:
        return {"repeat": repeat, "diversity": diversity, "scale_cov": 0.0}

    pcs = [p % 12 for p in pitches]
    major = {0, 2, 4, 5, 7, 9, 11}
    minor = {0, 2, 3, 5, 7, 8, 10}
    best = 0.0
    for root in range(12):
        maj = {(n + root) % 12 for n in major}
        mino = {(n + root) % 12 for n in minor}
        best = max(best, sum(pc in maj for pc in pcs) / len(pcs), sum(pc in mino for pc in pcs) / len(pcs))

    return {"repeat": repeat, "diversity": diversity, "scale_cov": best}


def evaluate(model, loader, id2token, pad_id):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    total_tokens = 0
    music = {"repeat": 0.0, "diversity": 0.0, "scale_cov": 0.0}
    music_n = 0

    with torch.no_grad():
        for x, y, genre_idx in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            genre_idx = genre_idx.to(DEVICE)

            logits = model(x, role_id=None, genre_id=genre_idx)
            loss, valid = token_loss(logits, y, pad_id)

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

            preds = logits.argmax(dim=-1)
            total_correct += ((preds == y) & valid).sum().item()
            total_tokens += valid.sum().item()

            for i in range(x.size(0)):
                pred_ids = preds[i][valid[i]].detach().cpu().tolist()
                mm = sequence_music_metrics(pred_ids, id2token, pad_id)
                for k in music:
                    music[k] += mm[k]
                music_n += 1

    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 20.0))
    music = {k: v / max(1, music_n) for k, v in music.items()}
    model.train()
    return avg_loss, acc, ppl, music


def save_plots(history):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "loss_full.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(history["val_acc"], label="val_acc")
    plt.title("Val Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "acc_full.png", dpi=200)
    plt.close()


def main():
    set_seed(SEED)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    token2id = vocab["token2id"]
    id2token = {int(k): v for k, v in vocab["id2token"].items()}

    train_ds, val_ds, test_ds, num_genres = build_datasets(token2id)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = TransformerLM(
        vocab_size=len(token2id),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        max_len=MAX_LEN,
        pad_id=token2id.get("<PAD>"),
        num_roles=0,
        num_genres=max(1, num_genres),
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    start_epoch = 0
    best_val = float("inf")

    if RESUME_FROM_CHECKPOINT and RESUME_CHECKPOINT_PATH.exists():
        cp = torch.load(RESUME_CHECKPOINT_PATH, map_location=DEVICE)
        state = cp.get("model_state_dict", cp)
        model.load_state_dict({k.replace("module.", "", 1): v for k, v in state.items()})
        start_epoch = int(cp.get("epoch", 0))
        best_val = float(cp.get("val_loss", best_val))

    print(f"Device: {DEVICE}")
    print(f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"Params: {count_parameters(model):,}")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_ppl": [],
        "music_repeat": [],
        "music_diversity": [],
        "music_scale_cov": [],
    }

    early_counter = 0
    pad_id = token2id["<PAD>"]

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        running = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [TRAIN]")
        for x, y, genre_idx in pbar:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            genre_idx = genre_idx.to(DEVICE)

            logits = model(x, role_id=None, genre_id=genre_idx)
            loss, _ = token_loss(logits, y, pad_id)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / max(1, n)
        val_loss, val_acc, val_ppl, val_music = evaluate(model, val_loader, id2token, pad_id)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_ppl"].append(val_ppl)
        history["music_repeat"].append(val_music["repeat"])
        history["music_diversity"].append(val_music["diversity"])
        history["music_scale_cov"].append(val_music["scale_cov"])

        print(f"✓ Epoch {epoch + 1}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Acc: {val_acc:.4f}")
        print(f"  Val PPL: {val_ppl:.2f}")
        print(f"  Music: repeat={val_music['repeat']:.3f}, diversity={val_music['diversity']:.3f}, scale_cov={val_music['scale_cov']:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            early_counter = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "model_config": {
                        "d_model": D_MODEL,
                        "n_heads": N_HEADS,
                        "n_layers": N_LAYERS,
                        "d_ff": D_FF,
                        "dropout": DROPOUT,
                        "max_len": MAX_LEN,
                        "pad_id": pad_id,
                        "num_roles": 0,
                        "num_genres": max(1, num_genres),
                    },
                },
                CHECKPOINT_DIR / "model_best_full.pth",
            )
            print(f"💾 Best model saved (Val Loss: {val_loss:.4f})\n")
        else:
            early_counter += 1
            print(f"⚠️ Val loss not improved ({early_counter}/{EARLY_STOPPING_PATIENCE})\n")
            if early_counter >= EARLY_STOPPING_PATIENCE:
                print("🛑 Early stopping")
                break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
                "d_ff": D_FF,
                "dropout": DROPOUT,
                "max_len": MAX_LEN,
                "pad_id": pad_id,
                "num_roles": 0,
                "num_genres": max(1, num_genres),
            },
        },
        CHECKPOINT_DIR / "model_final_full.pth",
    )

    with open(CHECKPOINT_DIR / "history_full.json", "w") as f:
        json.dump(history, f, indent=2)

    save_plots(history)

    best_cp = CHECKPOINT_DIR / "model_best_full.pth"
    if best_cp.exists():
        cp = torch.load(best_cp, map_location=DEVICE)
        model.load_state_dict(cp["model_state_dict"])

    test_loss, test_acc, test_ppl, test_music = evaluate(model, test_loader, id2token, pad_id)
    print("\n🧪 Test metrics (best full checkpoint):")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Acc: {test_acc:.4f}")
    print(f"  Test PPL: {test_ppl:.2f}")
    print(f"  Music: repeat={test_music['repeat']:.3f}, diversity={test_music['diversity']:.3f}, scale_cov={test_music['scale_cov']:.3f}")


if __name__ == "__main__":
    main()
