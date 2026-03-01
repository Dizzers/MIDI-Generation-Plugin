import numpy as np
import os

MAX_LEN = 512
STRIDE = 256

def chunk_sequence(seq, max_len, stride):
    chunks = []
    start = 0
    while start + max_len <= len(seq):
        chunk = seq[start:start + max_len]
        chunks.append(chunk)
        start += stride
    return chunks

def process_role(role):
    input_path = f"dataset/processed/tokens/{role}.npy"
    output_path = f"dataset/processed/chunks/{role}_chunks.npy"

    os.makedirs("dataset/processed/chunks", exist_ok=True)

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
