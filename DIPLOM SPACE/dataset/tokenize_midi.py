"""Tokenize each kept MIDI from processed/midi_meta.jsonl into a Performance
token sequence with prefix [<BOS>, <GENRE_TRAP>, <KEY_*>] + body events.

Saves processed/tokens/full.npy as a numpy object-array of token-lists.

Usage:
    python -m dataset.tokenize_midi
"""
from __future__ import annotations

import json
import signal
import sys
import warnings
from pathlib import Path

# When run as `python dataset/tokenize_midi.py`, sys.path[0] is `dataset/`, not the
# project root — `import dataset` fails. Same fix as `python -m dataset.tokenize_midi`.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import music21 as m21
import numpy as np
from tqdm import tqdm

from dataset.vocab_contract import (
    GENRE_TOKENS,
    note_off_token,
    note_on_token,
    split_time_shift,
    velocity_token,
)
from dataset.preprocess_midi import (
    PARSE_TIMEOUT_SECONDS,
    ParseTimeoutError,
    estimate_seconds_per_quarter,
    parse_with_timeout,
)

warnings.filterwarnings("ignore", category=m21.midi.translate.TranslateWarning)

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "midi_raw"
PROCESSED_DIR = BASE_DIR / "processed"
META_PATH = PROCESSED_DIR / "midi_meta.jsonl"
TOKENS_DIR = PROCESSED_DIR / "tokens"
FULL_PATH = TOKENS_DIR / "full.npy"
META_OUT_PATH = PROCESSED_DIR / "tokenize_stats.json"

DEFAULT_GENRE_TOKEN = GENRE_TOKENS[0]


def midi_events_to_tokens(score: m21.stream.Score) -> list[str]:
    """Convert a music21 Score to a flat list of body tokens (no prefix)."""
    sec_per_q = estimate_seconds_per_quarter(score)

    raw_events = []
    for item in score.flatten().notes:
        offset_q = float(item.offset)
        duration_q = max(1e-3, float(item.quarterLength))
        if item.isNote:
            pitches = [int(item.pitch.midi)]
            velocity = int(item.volume.velocity or 64)
        elif item.isChord:
            pitches = [int(p.midi) for p in item.pitches]
            velocity = int(item.volume.velocity or 64)
        else:
            continue
        for pitch in pitches:
            on_t = offset_q * sec_per_q
            off_t = (offset_q + duration_q) * sec_per_q
            raw_events.append((on_t, 1, "ON", pitch, velocity))
            raw_events.append((off_t, 0, "OFF", pitch, None))

    raw_events.sort(key=lambda row: (row[0], row[1], row[3]))

    tokens: list[str] = []
    last_t = 0.0
    for t, _ord, kind, pitch, vel in raw_events:
        delta = t - last_t
        if delta > 1e-6:
            tokens.extend(split_time_shift(delta))
        if kind == "ON":
            tokens.append(velocity_token(vel if vel is not None else 64))
            tokens.append(note_on_token(pitch))
        else:
            tokens.append(note_off_token(pitch))
        last_t = t
    return tokens


def main() -> int:
    if not META_PATH.exists():
        print(f"missing {META_PATH}; run dataset/preprocess_midi.py first")
        return 1

    TOKENS_DIR.mkdir(parents=True, exist_ok=True)

    metas: list[dict] = []
    with open(META_PATH, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                metas.append(json.loads(line))

    print(f"Tokenizing {len(metas)} files using genre prefix {DEFAULT_GENRE_TOKEN}")

    sequences: list[list[str]] = []
    sequences_meta: list[dict] = []
    errors: list[dict] = []
    timeouts = 0
    total_body_tokens = 0
    key_counts: dict[str, int] = {}

    for meta in tqdm(metas, desc="tokenize", unit="file"):
        midi_path = RAW_DIR / meta["path"]
        if not midi_path.exists():
            errors.append({"file": meta["path"], "error": "missing on disk"})
            continue
        try:
            score = parse_with_timeout(midi_path)
        except ParseTimeoutError as exc:
            timeouts += 1
            errors.append({"file": meta["path"], "error": str(exc)})
            continue
        except Exception as exc:
            errors.append({"file": meta["path"], "error": str(exc)[:200]})
            continue

        body = midi_events_to_tokens(score)
        if not body:
            errors.append({"file": meta["path"], "error": "no body tokens"})
            continue

        prefix = [DEFAULT_GENRE_TOKEN, meta.get("key_token", "<KEY_UNKNOWN>")]
        seq = prefix + body
        sequences.append(seq)
        sequences_meta.append(
            {
                "path": meta["path"],
                "key_token": meta.get("key_token"),
                "tokens": len(seq),
            }
        )
        total_body_tokens += len(body)
        key_counts[prefix[1]] = key_counts.get(prefix[1], 0) + 1

    np.save(FULL_PATH, np.array(sequences, dtype=object), allow_pickle=True)
    with open(PROCESSED_DIR / "tokenize_index.jsonl", "w", encoding="utf-8") as handle:
        for row in sequences_meta:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "files_seen": len(metas),
        "sequences_kept": len(sequences),
        "errors": len(errors),
        "timeouts": timeouts,
        "avg_tokens_per_sequence": round(total_body_tokens / max(1, len(sequences)), 2),
        "key_distribution_after_filter": dict(
            sorted(key_counts.items(), key=lambda kv: -kv[1])
        ),
        "default_genre": DEFAULT_GENRE_TOKEN,
        "output_full_path": str(FULL_PATH.relative_to(BASE_DIR.parent)),
    }
    with open(META_OUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    if errors:
        with open(PROCESSED_DIR / "tokenize_errors.json", "w", encoding="utf-8") as handle:
            json.dump(errors, handle, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
