import json
import random
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
TOKENS_DIR = PROCESSED_DIR / "tokens"

SEED = 42
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1


def main():
    input_path = TOKENS_DIR / "full.npy"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}. Run dataset/tokenize_midi.py first.")

    sequences = np.load(input_path, allow_pickle=True).tolist()
    rows = []
    for seq in sequences:
        seq = list(seq)
        genre = "UNKNOWN"
        if seq and isinstance(seq[0], str) and seq[0].startswith("<GENRE_"):
            genre = seq[0][7:-1]
        rows.append({"genre": genre, "tokens": seq})

    by_genre = {}
    for row in rows:
        by_genre.setdefault(row["genre"], []).append(row)

    rng = random.Random(SEED)
    splits = {"train": [], "val": [], "test": []}
    for genre, genre_rows in sorted(by_genre.items()):
        rng.shuffle(genre_rows)
        n = len(genre_rows)
        n_val = max(1, int(round(n * VAL_SPLIT))) if n >= 10 else max(0, int(round(n * VAL_SPLIT)))
        n_test = max(1, int(round(n * TEST_SPLIT))) if n >= 10 else max(0, int(round(n * TEST_SPLIT)))
        n_train = max(1, n - n_val - n_test) if n > 0 else 0

        splits["train"].extend(genre_rows[:n_train])
        splits["val"].extend(genre_rows[n_train:n_train + n_val])
        splits["test"].extend(genre_rows[n_train + n_val:n_train + n_val + n_test])

    for split_name in ["train", "val", "test"]:
        split_tokens = [row["tokens"] for row in splits[split_name]]
        np.save(TOKENS_DIR / f"full_{split_name}.npy", np.array(split_tokens, dtype=object))

    stats = {
        "seed": SEED,
        "val_split": VAL_SPLIT,
        "test_split": TEST_SPLIT,
        "total_sequences": len(rows),
        "train_sequences": len(splits["train"]),
        "val_sequences": len(splits["val"]),
        "test_sequences": len(splits["test"]),
        "genres": {genre: len(genre_rows) for genre, genre_rows in sorted(by_genre.items())},
    }
    with open(PROCESSED_DIR / "full_split_stats.json", "w") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

