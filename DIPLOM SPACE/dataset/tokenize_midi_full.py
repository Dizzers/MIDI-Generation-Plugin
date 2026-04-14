import json
import os
from collections import defaultdict

import music21 as m21
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "midi_raw")
META_DIR = os.path.join(BASE_DIR, "processed", "meta")
TOKENS_DIR = os.path.join(BASE_DIR, "processed", "tokens")
os.makedirs(TOKENS_DIR, exist_ok=True)

TIME_SHIFT_RESOLUTION = 0.05
VELOCITY_BINS = 8

# Keep empty to use all genres found in files.json.
# Set to {"trap"} for trap-only training. было set()
TARGET_GENRES = {"trap"}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def velocity_bin(v):
    b = int((v / 127) * (VELOCITY_BINS - 1))
    return max(0, min(VELOCITY_BINS - 1, b))


def quantize_time(delta):
    steps = int(round(delta / TIME_SHIFT_RESOLUTION))
    return max(1, steps)


def detect_key_token(score):
    try:
        k = score.analyze("Krumhansl")
        tonic = k.tonic.name.replace("-", "b")
        mode = "MAJ" if (k.mode or "major").lower().startswith("maj") else "MIN"
        pc = m21.pitch.Pitch(tonic).pitchClass
        root = NOTE_NAMES[int(pc) % 12]
        return f"<KEY_{root}_{mode}>"
    except Exception:
        return "<KEY_UNKNOWN>"


def extract_events_from_score(score):
    events = []
    for part in score.parts:
        for n in part.flatten().notes:
            start = float(n.offset)
            dur = float(n.quarterLength)

            if n.isNote:
                pitches = [n.pitch.midi]
                vel = n.volume.velocity
            elif n.isChord:
                pitches = [p.midi for p in n.pitches]
                vel = n.volume.velocity
            else:
                continue

            for pitch in pitches:
                events.append((start, 1, "NOTE_ON", int(pitch), vel))
                events.append((start + max(dur, TIME_SHIFT_RESOLUTION), 0, "NOTE_OFF", int(pitch), None))

    events.sort(key=lambda x: (x[0], x[1], x[3]))
    return events


def events_to_tokens(events):
    tokens = []
    last_time = 0.0
    for time, _, etype, pitch, vel in events:
        delta = time - last_time
        if delta > 0:
            tokens.append(f"TIME_SHIFT_{quantize_time(delta):#06x}")

        if etype == "NOTE_ON":
            tokens.extend([f"VELOCITY_{velocity_bin(vel or 64):#02x}", f"NOTE_ON_{pitch:#04x}"])
        else:
            tokens.append(f"NOTE_OFF_{pitch:#04x}")

        last_time = time
    return tokens


def tokenize_dataset_full():
    print(f"tokenize_midi_full.py path: {os.path.abspath(__file__)}")
    print(f"TARGET_GENRES={TARGET_GENRES if TARGET_GENRES else 'ALL'}")

    with open(os.path.join(META_DIR, "files.json")) as f:
        files = json.load(f)

    valid_entries = [
        e for e in files
        if e.get("status") == "ok" and ((not TARGET_GENRES) or e.get("genre", "").strip().lower() in TARGET_GENRES)
    ]

    print(f" Full tokenization: {len(valid_entries)} files\n")

    all_sequences = []
    vocab = set()
    errors = []
    genres_seen = set()
    keys_seen = set()

    for entry in tqdm(valid_entries, desc="Tokenize full", unit="file"):
        genre = str(entry["genre"]).strip().lower()
        genres_seen.add(genre)
        midi_path = os.path.join(RAW_DIR, genre, entry["file"])

        try:
            score = m21.converter.parse(midi_path, forceSource=True)
            key_token = detect_key_token(score)
            keys_seen.add(key_token)
            events = extract_events_from_score(score)
            if not events:
                continue

            tokens = [f"<GENRE_{genre.upper()}>", key_token]
            tokens += events_to_tokens(events)

            all_sequences.append(tokens)
            vocab.update(tokens)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            errors.append({"file": entry.get("file"), "error": str(exc)[:120]})

    np.save(os.path.join(TOKENS_DIR, "full.npy"), np.array(all_sequences, dtype=object))

    key_tokens = sorted(keys_seen) if keys_seen else ["<KEY_UNKNOWN>"]
    genre_tokens = sorted({f"<GENRE_{g.upper()}>" for g in genres_seen}) if genres_seen else ["<GENRE_TRAP>"]

    special_tokens = [
        "<PAD>",
        "<BOS>",
        "<EOS>",
        "<UNK>",
        *genre_tokens,
        *key_tokens,
    ]

    vocab_clean = [t for t in vocab if t not in special_tokens]
    vocab_list = special_tokens + sorted(vocab_clean)

    token2id = {t: i for i, t in enumerate(vocab_list)}
    id2token = {i: t for t, i in token2id.items()}

    with open(os.path.join(BASE_DIR, "processed", "vocab_full.json"), "w") as f:
        json.dump({"token2id": token2id, "id2token": id2token, "size": len(token2id)}, f, indent=2)

    if errors:
        with open(os.path.join(BASE_DIR, "processed", "tokenize_full_errors.json"), "w") as f:
            json.dump(errors, f, indent=2)

    print("\n" + "=" * 50)
    print("FULL TOKENIZATION DONE")
    print("=" * 50)
    print(f"sequences: {len(all_sequences)}")
    print(f"genres:    {len(genre_tokens)} -> {genre_tokens}")
    print(f"keys:      {len(key_tokens)}")
    print(f"vocab:     {len(vocab_list)}")
    print("=" * 50)


if __name__ == "__main__":
    tokenize_dataset_full()
