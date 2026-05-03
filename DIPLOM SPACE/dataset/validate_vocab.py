"""Validate that processed/vocab.json conforms to the contract expected by the
existing C++ plugin (DIPLOM SPACE/plugin/juce/Source/ModelInference.cpp and
MidiGenerator.cpp). Run as a standalone module:

    python -m dataset.validate_vocab
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dataset.vocab_contract import (
    GENRE_RE,
    KEY_RE,
    NOTE_OFF_RE,
    NOTE_ON_RE,
    SPECIAL_TOKENS,
    TIME_SHIFT_RE,
    VELOCITY_RE,
    VELOCITY_BINS,
    parse_hex_suffix,
)

DEFAULT_VOCAB_PATH = Path(__file__).resolve().parent / "processed" / "vocab.json"


def validate(vocab_path: Path) -> list[str]:
    errors: list[str] = []
    if not vocab_path.exists():
        return [f"vocab file not found: {vocab_path}"]

    with open(vocab_path, encoding="utf-8") as handle:
        vocab = json.load(handle)

    if "token2id" not in vocab or "id2token" not in vocab:
        errors.append("vocab.json must contain 'token2id' and 'id2token' fields")
        return errors

    token2id = vocab["token2id"]
    id2token = vocab["id2token"]

    if len(token2id) != len(id2token):
        errors.append(
            f"token2id ({len(token2id)}) and id2token ({len(id2token)}) sizes differ"
        )

    for token, idx in token2id.items():
        back = id2token.get(str(idx))
        if back != token:
            errors.append(f"round-trip failed: token2id[{token!r}] -> {idx} -> {back!r}")

    for required in SPECIAL_TOKENS:
        if required not in token2id:
            errors.append(f"missing required special token: {required}")

    genre_tokens = [t for t in token2id if t.startswith("<GENRE_")]
    if not genre_tokens:
        errors.append("vocab must contain at least one <GENRE_*> token")
    if "<GENRE_TRAP>" not in token2id:
        errors.append(
            "C++ ModelInference.cpp hardcodes <GENRE_TRAP>; it must exist in vocab"
        )

    key_tokens = [t for t in token2id if t.startswith("<KEY_")]
    if "<KEY_UNKNOWN>" not in token2id:
        errors.append("missing <KEY_UNKNOWN> fallback token")

    for token in token2id:
        if token in SPECIAL_TOKENS:
            continue
        if token.startswith("<GENRE_"):
            if not GENRE_RE.match(token):
                errors.append(f"genre token doesn't match regex: {token}")
            continue
        if token.startswith("<KEY_"):
            if not KEY_RE.match(token):
                errors.append(f"key token doesn't match regex: {token}")
            continue
        if token.startswith("NOTE_ON_"):
            if not NOTE_ON_RE.match(token):
                errors.append(f"NOTE_ON token doesn't match regex: {token}")
            value = parse_hex_suffix(token)
            if value is None or not (0 <= value <= 127):
                errors.append(f"NOTE_ON pitch out of [0..127]: {token}")
            continue
        if token.startswith("NOTE_OFF_"):
            if not NOTE_OFF_RE.match(token):
                errors.append(f"NOTE_OFF token doesn't match regex: {token}")
            value = parse_hex_suffix(token)
            if value is None or not (0 <= value <= 127):
                errors.append(f"NOTE_OFF pitch out of [0..127]: {token}")
            continue
        if token.startswith("TIME_SHIFT_"):
            if not TIME_SHIFT_RE.match(token):
                errors.append(f"TIME_SHIFT token doesn't match regex: {token}")
            value = parse_hex_suffix(token)
            if value is None or value < 1:
                errors.append(f"TIME_SHIFT steps must be >=1: {token}")
            continue
        if token.startswith("VELOCITY_"):
            if not VELOCITY_RE.match(token):
                errors.append(f"VELOCITY token doesn't match regex: {token}")
            value = parse_hex_suffix(token)
            if value is None or not (0 <= value < VELOCITY_BINS):
                errors.append(
                    f"VELOCITY bin out of [0..{VELOCITY_BINS - 1}]: {token}"
                )
            continue
        errors.append(f"unknown token class: {token}")

    sorted_genres = sorted(genre_tokens)
    if sorted_genres != sorted(genre_tokens):
        errors.append("internal: genre sorting mismatch (should not happen)")

    return errors


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VOCAB_PATH
    errors = validate(path)
    if errors:
        print(f"vocab.json FAILED ({len(errors)} errors):")
        for line in errors:
            print(f"  - {line}")
        return 1
    print(f"vocab.json OK: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
