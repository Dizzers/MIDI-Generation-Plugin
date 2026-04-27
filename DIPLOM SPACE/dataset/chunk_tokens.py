from pathlib import Path

import numpy as np

MAX_LEN = 512
STRIDE = 256

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
TOKENS_DIR = PROCESSED_DIR / "tokens"
CHUNKS_DIR = PROCESSED_DIR / "chunks"


def chunk_sequence(sequence, max_len, stride):
    prefix_tokens = sequence[:2] if len(sequence) >= 2 else []
    body_tokens = sequence[2:] if len(sequence) >= 2 else sequence
    body_chunk_len = max_len - len(prefix_tokens)

    chunks = []
    start = 0
    while start < len(body_tokens):
        end = min(start + body_chunk_len, len(body_tokens))
        chunk = prefix_tokens + body_tokens[start:end]
        chunks.append(chunk)
        start += stride
    return chunks


def main():
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    def build_chunks(split_name: str, in_name: str, out_name: str):
        input_path = TOKENS_DIR / in_name
        output_path = CHUNKS_DIR / out_name
        if not input_path.exists():
            print(f"[{split_name}] missing: {input_path}")
            return

        sequences = np.load(input_path, allow_pickle=True)
        print(f"[{split_name}] sequences: {len(sequences)}")

        all_chunks = []
        for idx, sequence in enumerate(sequences, start=1):
            if len(sequence) < MAX_LEN:
                all_chunks.append(sequence)
            else:
                all_chunks.extend(chunk_sequence(sequence, MAX_LEN, STRIDE))

            if idx % 5000 == 0:
                print(f"[{split_name}] processed {idx}/{len(sequences)}")

        np.save(output_path, np.array(all_chunks, dtype=object))
        print(f"[{split_name}] done: {len(all_chunks)} chunks -> {output_path}")

    # Keep "full" (all files) for compatibility.
    build_chunks("full", "full.npy", "full_chunks.npy")
    # New: split-by-file chunks to avoid leakage in training.
    build_chunks("train", "full_train.npy", "full_chunks_train.npy")
    build_chunks("val", "full_val.npy", "full_chunks_val.npy")
    build_chunks("test", "full_test.npy", "full_chunks_test.npy")


if __name__ == "__main__":
    main()
