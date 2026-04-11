import os
from pathlib import Path

import numpy as np

MAX_LEN = 256
STRIDE = 128
SHUFFLE_SEED = 42
MAX_CHUNKS_PER_ROLE = {
    "chords": 8000,
    "melody": None,
    "bass": None,
}
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

    print(f"[{role}] loading: {input_path}", flush=True)
    sequences = np.load(input_path, allow_pickle=True)
    print(f"[{role}] sequences: {len(sequences)}", flush=True)

    all_chunks = []
    for idx, seq in enumerate(sequences, start=1):
        if len(seq) < MAX_LEN:
            all_chunks.append(seq)
        else:
            all_chunks.extend(chunk_sequence(seq, MAX_LEN, STRIDE))

        if idx % 5000 == 0:
            print(f"[{role}] processed {idx}/{len(sequences)} sequences", flush=True)

    max_chunks = MAX_CHUNKS_PER_ROLE.get(role)
    if max_chunks is not None and len(all_chunks) > max_chunks:
        rng = np.random.default_rng(SHUFFLE_SEED)
        selected = rng.choice(len(all_chunks), size=max_chunks, replace=False)
        all_chunks = [all_chunks[i] for i in selected]
        print(f"[{role}] capped to {len(all_chunks)} chunks", flush=True)

    np.save(output_path, np.array(all_chunks, dtype=object))
    print(f"[{role}] done: {len(all_chunks)} chunks -> {output_path}", flush=True)

if __name__ == "__main__":
    for role in ["chords", "melody", "bass"]:
        process_role(role)
