import argparse
import json
import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.generate import generate_best_candidate, load_model_and_vocab


def ngram_repeat_rate(tokens, n=4):
    body = [token for token in tokens if not (token in {"<BOS>", "<EOS>"} or token.startswith("<GENRE_") or token.startswith("<KEY_"))]
    if len(body) < n:
        return 0.0
    grams = [tuple(body[idx:idx + n]) for idx in range(len(body) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def note_on_off_balance(tokens):
    note_on = sum(token.startswith("NOTE_ON_") for token in tokens)
    note_off = sum(token.startswith("NOTE_OFF_") for token in tokens)
    return 1.0 - abs(note_on - note_off) / max(1, note_on + note_off)


def polyphony_ratio(tokens):
    current = 0
    onset_counts = {}
    for token in tokens:
        if token.startswith("TIME_SHIFT_"):
            try:
                current += int(token.split("_")[2], 16)
            except Exception:
                pass
        elif token.startswith("NOTE_ON_"):
            onset_counts[current] = onset_counts.get(current, 0) + 1
    if not onset_counts:
        return 0.0
    return sum(1 for count in onset_counts.values() if count >= 2) / len(onset_counts)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate full-MIDI generation quality")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_best.pth")
    parser.add_argument("--vocab", type=str, default="dataset/processed/vocab.json")
    parser.add_argument("--genre", type=str, default="TRAP")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=18)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.18)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--primer-mode", type=str, default="dataset")
    parser.add_argument("--primer-len", type=int, default=64)
    parser.add_argument("--min-body-tokens", type=int, default=48)
    parser.add_argument("--target-seconds", type=float, default=8.0)
    parser.add_argument("--key", type=str, default="AUTO")
    parser.add_argument("--candidates-per-sample", type=int, default=8)
    parser.add_argument("--diversity-jitter", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, token2id, id2token, genre_to_index = load_model_and_vocab(PROJECT_ROOT / args.checkpoint, PROJECT_ROOT / args.vocab)
    cache = {}
    metrics = {
        "repeat_rate_4gram": 0.0,
        "note_on_off_balance": 0.0,
        "polyphony_ratio": 0.0,
        "eos_rate": 0.0,
    }

    for _ in range(args.samples):
        tokens, ended = generate_best_candidate(
            model=model,
            token2id=token2id,
            id2token=id2token,
            genre=args.genre.upper(),
            max_len=args.max_len,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            genre_to_index=genre_to_index,
            primer_mode=args.primer_mode,
            primer_len=args.primer_len,
            full_sequences_cache=cache,
            min_body_tokens=args.min_body_tokens,
            target_seconds=args.target_seconds,
            key_name=args.key,
            candidates_per_sample=args.candidates_per_sample,
            diversity_jitter=args.diversity_jitter,
        )
        metrics["repeat_rate_4gram"] += ngram_repeat_rate(tokens)
        metrics["note_on_off_balance"] += note_on_off_balance(tokens)
        metrics["polyphony_ratio"] += polyphony_ratio(tokens)
        metrics["eos_rate"] += float(bool(ended))

    for key in metrics:
        metrics[key] /= max(1, args.samples)

    print("Full-MIDI Generation Metrics")
    print(f"  EOS rate: {metrics['eos_rate']:.3f}")
    print(f"  4-gram repeat rate (lower better): {metrics['repeat_rate_4gram']:.3f}")
    print(f"  NOTE_ON/OFF balance (higher better): {metrics['note_on_off_balance']:.3f}")
    print(f"  Polyphony ratio (higher better): {metrics['polyphony_ratio']:.3f}")

    out = {"config": vars(args), **metrics}
    out_path = PROJECT_ROOT / "checkpoints" / "generation_metrics.json"
    with open(out_path, "w") as handle:
        json.dump(out, handle, indent=2)
    print(f"Saved metrics: {out_path}")


if __name__ == "__main__":
    main()
