"""Emit processed/vocab.json in the schema expected by the C++ plugin
(token2id + id2token), built from the canonical token list in
dataset/vocab_contract.py. Order is deterministic and stable across runs.

Layout (matches existing plugin/juce/bin/vocab.json conventions):
    0..3                    <PAD> <BOS> <EOS> <UNK>
    4..(4+G-1)              <GENRE_*> in alphabetical order
    next                    <KEY_*> tokens (12 maj + 12 min + UNKNOWN)
    next                    NOTE_OFF_0xPP for p in 0..127
    next                    NOTE_ON_0xPP for p in 0..127
    next                    TIME_SHIFT_0xSSSS for s in 1..MAX_TIME_STEPS
    last                    VELOCITY_0xVV for v in 0..VELOCITY_BINS-1

Usage:
    python -m dataset.build_vocab
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset.validate_vocab import validate
from dataset.vocab_contract import (
    GENRE_TOKENS,
    KEY_TOKENS,
    MAX_TIME_STEPS,
    NOTE_OFF_TOKENS,
    NOTE_ON_TOKENS,
    SPECIAL_TOKENS,
    TIME_SHIFT_TOKENS,
    VELOCITY_TOKENS,
)

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
VOCAB_PATH = PROCESSED_DIR / "vocab.json"


def build_vocab_dict() -> dict:
    ordered: list[str] = []
    ordered.extend(SPECIAL_TOKENS)
    ordered.extend(sorted(GENRE_TOKENS))
    ordered.extend(KEY_TOKENS)
    ordered.extend(NOTE_OFF_TOKENS)
    ordered.extend(NOTE_ON_TOKENS)
    ordered.extend(TIME_SHIFT_TOKENS)
    ordered.extend(VELOCITY_TOKENS)

    seen = set()
    deduped: list[str] = []
    for tok in ordered:
        if tok in seen:
            raise ValueError(f"duplicate token in canonical list: {tok}")
        seen.add(tok)
        deduped.append(tok)

    token2id = {tok: idx for idx, tok in enumerate(deduped)}
    id2token = {str(idx): tok for tok, idx in token2id.items()}
    return {
        "token2id": token2id,
        "id2token": id2token,
        "size": len(token2id),
        "max_time_steps": MAX_TIME_STEPS,
    }


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    vocab = build_vocab_dict()
    with open(VOCAB_PATH, "w", encoding="utf-8") as handle:
        json.dump(vocab, handle, indent=2, ensure_ascii=False)
    print(f"wrote {VOCAB_PATH} (size={vocab['size']})")

    errors = validate(VOCAB_PATH)
    if errors:
        print(f"validation FAILED ({len(errors)} errors):")
        for line in errors:
            print(f"  - {line}")
        return 1
    print("validation OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
