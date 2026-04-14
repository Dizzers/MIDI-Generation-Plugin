import os
from pathlib import Path

import numpy as np

MAX_LEN = 256
STRIDE = 128
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
TOKENS_DIR = PROCESSED_DIR / "tokens"
CHUNKS_DIR = PROCESSED_DIR / "chunks"


def chunk_sequence(seq, max_len, stride):
    chunks = []
    start = 0
    while start + max_len <= len(seq):
        chunks.append(seq[start:start + max_len])
        start += stride
    return chunks


def main():
    input_path = TOKENS_DIR / "full.npy"
    output_path = CHUNKS_DIR / "full_chunks.npy"

    os.makedirs(CHUNKS_DIR, exist_ok=True)

    print(f"[full] loading: {input_path}", flush=True)
    sequences = np.load(input_path, allow_pickle=True)
    print(f"[full] sequences: {len(sequences)}", flush=True)

    all_chunks = []
    for i, seq in enumerate(sequences, start=1):
        if len(seq) < MAX_LEN:
            all_chunks.append(seq)
        else:
            all_chunks.extend(chunk_sequence(seq, MAX_LEN, STRIDE))

        if i % 5000 == 0:
            print(f"[full] processed {i}/{len(sequences)} sequences", flush=True)

    np.save(output_path, np.array(all_chunks, dtype=object))
    print(f"[full] done: {len(all_chunks)} chunks -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
