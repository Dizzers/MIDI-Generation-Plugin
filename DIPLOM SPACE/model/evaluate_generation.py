import argparse
import json
import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.generate import ROLES, generate_tokens, load_model_and_vocab


def ngram_repeat_rate(tokens, n=4):
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def note_on_off_balance(tokens):
    on = sum(tok.startswith("NOTE_ON_") for tok in tokens)
    off = sum(tok.startswith("NOTE_OFF_") for tok in tokens)
    return 1.0 - abs(on - off) / max(1, on + off)


def chord_role_consistency(tokens):
    groups = []
    cur = 0
    for tok in tokens:
        if tok.startswith("NOTE_ON_"):
            cur += 1
        elif tok.startswith("TIME_SHIFT_"):
            groups.append(cur)
            cur = 0
    if cur > 0:
        groups.append(cur)
    if not groups:
        return 0.0
    chord_like = sum(g >= 2 for g in groups)
    return chord_like / len(groups)


def token_to_pitch(tok):
    if tok.startswith("NOTE_ON_"):
        try:
            return int(tok.split("_")[2], 16)
        except Exception:
            return None
    return None


def extract_melody_times(tokens):
    t = 0.0
    points = []
    for tok in tokens:
        if tok.startswith("TIME_SHIFT_"):
            t += int(tok.split("_")[2], 16) * 0.05
        elif tok.startswith("NOTE_ON_"):
            p = token_to_pitch(tok)
            if p is not None:
                points.append((t, p))
    return points


def extract_harmony_times(tokens):
    t = 0.0
    points = []
    for tok in tokens:
        if tok.startswith("TIME_SHIFT_"):
            t += int(tok.split("_")[2], 16) * 0.05
        elif tok.startswith("NOTE_ON_"):
            p = token_to_pitch(tok)
            if p is not None:
                points.append((t, p))
    return points


def cross_track_consonance(melody_tokens, bass_tokens, chord_tokens):
    melody_points = extract_melody_times(melody_tokens)
    harmony_points = extract_harmony_times(bass_tokens) + extract_harmony_times(chord_tokens)
    if not melody_points or not harmony_points:
        return 0.0

    harmony_points.sort(key=lambda x: x[0])
    consonant = {0, 3, 4, 5, 7, 8, 9}
    hit = 0
    total = 0
    idx = 0
    current_harmony_pitch = harmony_points[0][1]

    for mt, mp in melody_points:
        while idx + 1 < len(harmony_points) and harmony_points[idx + 1][0] <= mt:
            idx += 1
            current_harmony_pitch = harmony_points[idx][1]
        interval = abs(mp - current_harmony_pitch) % 12
        hit += interval in consonant
        total += 1

    return hit / max(1, total)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate generation quality metrics")
    p.add_argument("--checkpoint", type=str, default="checkpoints/model_best.pth")
    p.add_argument("--vocab", type=str, default="dataset/processed/vocab.json")
    p.add_argument("--genre", type=str, default="TRAP")
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--max-len", type=int, default=400)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--no-repeat-ngram-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, token2id, id2token, role_to_index, genre_to_index = load_model_and_vocab(
        PROJECT_ROOT / args.checkpoint,
        PROJECT_ROOT / args.vocab,
    )

    eos_hits = 0
    metrics = {
        "repeat_rate_4gram": 0.0,
        "note_on_off_balance": 0.0,
        "chord_role_consistency": 0.0,
        "cross_track_consonance": 0.0,
    }

    for _ in range(args.samples):
        generated = {}
        all_eos = True
        for role in ROLES:
            tokens, ended_by_eos = generate_tokens(
                model=model,
                token2id=token2id,
                id2token=id2token,
                role=role,
                genre=args.genre.upper(),
                max_len=args.max_len,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                role_to_index=role_to_index,
                genre_to_index=genre_to_index,
            )
            generated[role] = tokens
            all_eos = all_eos and ended_by_eos

            metrics["repeat_rate_4gram"] += ngram_repeat_rate(tokens, n=4)
            metrics["note_on_off_balance"] += note_on_off_balance(tokens)
            if role == "CHORDS":
                metrics["chord_role_consistency"] += chord_role_consistency(tokens)

        metrics["cross_track_consonance"] += cross_track_consonance(
            generated["MELODY"], generated["BASS"], generated["CHORDS"]
        )
        eos_hits += int(all_eos)

    denom = float(args.samples)
    metrics["repeat_rate_4gram"] /= (denom * 3.0)
    metrics["note_on_off_balance"] /= (denom * 3.0)
    metrics["chord_role_consistency"] /= denom
    metrics["cross_track_consonance"] /= denom
    eos_rate = eos_hits / denom

    print("🧪 Generation Metrics")
    print(f"  EOS rate: {eos_rate:.3f}")
    print(f"  4-gram repeat rate (lower better): {metrics['repeat_rate_4gram']:.3f}")
    print(f"  NOTE_ON/OFF balance (higher better): {metrics['note_on_off_balance']:.3f}")
    print(f"  CHORDS consistency (higher better): {metrics['chord_role_consistency']:.3f}")
    print(f"  Cross-track consonance (higher better): {metrics['cross_track_consonance']:.3f}")

    out = {
        "eos_rate": eos_rate,
        **metrics,
        "config": vars(args),
    }
    out_path = PROJECT_ROOT / "checkpoints" / "generation_metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved metrics: {out_path}")


if __name__ == "__main__":
    main()
