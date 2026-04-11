import os
from pathlib import Path

import numpy as np

MAX_LEN = 512
STRIDE = 256
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
TOKENS_DIR = PROCESSED_DIR / "tokens"
CHUNKS_DIR = PROCESSED_DIR / "chunks"

def chunk_sequence(seq, max_len, stride):
    chunks = []
    start = 0
    while start + max_len <= len(seq):
        chunk = seq[start:start + max_len]
        chunks.append(chunk)
        start += stride
    return chunks

def process_role(role):
    input_path = TOKENS_DIR / f"{role}.npy"
    output_path = CHUNKS_DIR / f"{role}_chunks.npy"

    os.makedirs(CHUNKS_DIR, exist_ok=True)

    sequences = np.load(input_path, allow_pickle=True)

    all_chunks = []
    for seq in sequences:
        if len(seq) < MAX_LEN:
            all_chunks.append(seq)
        else:
            all_chunks.extend(chunk_sequence(seq, MAX_LEN, STRIDE))

    np.save(output_path, np.array(all_chunks, dtype=object))
    print(f"{role}: {len(all_chunks)} chunks")

if __name__ == "__main__":
    for role in ["chords", "melody", "bass"]:
        process_role(role)
