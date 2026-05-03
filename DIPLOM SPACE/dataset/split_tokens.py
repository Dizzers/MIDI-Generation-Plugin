"""File-level train/val/test split (80/10/10) over processed/tokens/full.npy.
Splitting at file granularity (not chunk) prevents leakage of context across
splits during training.

Usage:
    python -m dataset.split_tokens
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
TOKENS_DIR = PROCESSED_DIR / "tokens"
FULL_PATH = TOKENS_DIR / "full.npy"
STATS_PATH = PROCESSED_DIR / "split_stats.json"

SEED = 42
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10


def main() -> int:
    if not FULL_PATH.exists():
        print(f"missing {FULL_PATH}; run dataset/tokenize_midi.py first")
        return 1

    sequences = np.load(FULL_PATH, allow_pickle=True).tolist()
    n = len(sequences)
    if n == 0:
        print("empty token set; nothing to split")
        return 1

    indices = list(range(n))
    random.Random(SEED).shuffle(indices)
    n_val = max(1, int(round(n * VAL_FRACTION))) if n >= 10 else max(0, int(round(n * VAL_FRACTION)))
    n_test = max(1, int(round(n * TEST_FRACTION))) if n >= 10 else max(0, int(round(n * TEST_FRACTION)))
    n_train = max(1, n - n_val - n_test)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val : n_train + n_val + n_test]

    splits = {
        "train": [sequences[i] for i in train_idx],
        "val": [sequences[i] for i in val_idx],
        "test": [sequences[i] for i in test_idx],
    }

    for name, seqs in splits.items():
        np.save(TOKENS_DIR / f"full_{name}.npy", np.array(seqs, dtype=object), allow_pickle=True)

    stats = {
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "test_fraction": TEST_FRACTION,
        "total_sequences": n,
        "train_sequences": len(splits["train"]),
        "val_sequences": len(splits["val"]),
        "test_sequences": len(splits["test"]),
    }
    with open(STATS_PATH, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
