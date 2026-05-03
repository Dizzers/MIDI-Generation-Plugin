"""Chunk long sequences from full_{train,val,test}.npy into sliding windows of
MAX_LEN tokens with STRIDE step. The 2-token genre+key prefix is duplicated at
the head of every chunk so the model always sees its conditioning.

The encoded tensor in train.py adds <BOS> at the very front and <EOS> at the
end (handled by model/dataset.py), so chunks here store *prefix + body* only.

Usage:
    python -m dataset.chunk_tokens
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
TOKENS_DIR = PROCESSED_DIR / "tokens"
CHUNKS_DIR = PROCESSED_DIR / "chunks"

MAX_LEN = 1024
STRIDE = 512
PREFIX_LEN = 2  # <GENRE_*>, <KEY_*>


def chunk_one(sequence: list[str]) -> list[list[str]]:
    if len(sequence) <= MAX_LEN:
        return [list(sequence)]

    prefix = sequence[:PREFIX_LEN]
    body = sequence[PREFIX_LEN:]
    body_window = MAX_LEN - PREFIX_LEN

    chunks: list[list[str]] = []
    start = 0
    while start < len(body):
        end = min(start + body_window, len(body))
        chunks.append(prefix + body[start:end])
        if end == len(body):
            break
        start += STRIDE
    return chunks


def main() -> int:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        in_path = TOKENS_DIR / f"full_{split}.npy"
        out_path = CHUNKS_DIR / f"full_chunks_{split}.npy"
        if not in_path.exists():
            print(f"[{split}] missing {in_path}; skipping")
            continue
        sequences = np.load(in_path, allow_pickle=True).tolist()
        all_chunks: list[list[str]] = []
        for seq in sequences:
            all_chunks.extend(chunk_one(list(seq)))
        np.save(out_path, np.array(all_chunks, dtype=object), allow_pickle=True)
        summary[split] = {
            "input_sequences": len(sequences),
            "output_chunks": len(all_chunks),
        }
        print(f"[{split}] {len(sequences)} -> {len(all_chunks)} chunks -> {out_path}")

    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
