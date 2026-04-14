import os
import json
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
TARGET_GENRES = {"trap"}

TRACK_LIMIT_PER_FILE = {
    "chords": 1,
    "melody": 3,
    "bass": 2,
}
ROLE_SEQUENCE_CAP = {
    "chords": 1200,
    "melody": None,
    "bass": None,
}
RNG_SEED = 42

KEY_TOKENS = [
    *(f"<KEY_{pc}_MAJ>" for pc in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]),
    *(f"<KEY_{pc}_MIN>" for pc in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]),
    "<KEY_UNKNOWN>",
]

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

        # Normalize to sharp note names for compact vocab.
        pc = m21.pitch.Pitch(tonic).pitchClass
        root = NOTE_NAMES[int(pc) % 12]
        tok = f"<KEY_{root}_{mode}>"
        if tok in KEY_TOKENS:
            return tok
    except Exception:
        pass
    return "<KEY_UNKNOWN>"


def extract_events(part):
    events = []

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
            events.append((start, 1, "NOTE_ON", pitch, vel))
            events.append((start + dur, 0, "NOTE_OFF", pitch, None))

    events.sort(key=lambda x: (x[0], x[1], x[3]))
    return events


def role_quality_pass(role, events):
    note_on = [e for e in events if e[2] == "NOTE_ON"]
    if len(note_on) < 8:
        return False

    pitches = [e[3] for e in note_on]
    avg_pitch = float(np.mean(pitches)) if pitches else 0.0

    onset_counts = defaultdict(int)
    for t, _, etype, _, _ in note_on:
        onset_counts[round(float(t), 3)] += 1
    chord_onset_ratio = (
        sum(1 for c in onset_counts.values() if c >= 2) / max(1, len(onset_counts))
    )

    if role == "chords":
        return chord_onset_ratio >= 0.22

    if role == "melody":
        return chord_onset_ratio <= 0.30 and avg_pitch >= 50

    if role == "bass":
        low_ratio = sum(1 for p in pitches if p <= 55) / max(1, len(pitches))
        return chord_onset_ratio <= 0.32 and low_ratio >= 0.28

    return True


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


def cap_role_sequences(role_tokens):
    rng = np.random.default_rng(RNG_SEED)
    for role in ["chords", "melody", "bass"]:
        cap = ROLE_SEQUENCE_CAP.get(role)
        seqs = role_tokens.get(role, [])
        if cap is None or len(seqs) <= cap:
            continue
        idx = rng.choice(len(seqs), size=cap, replace=False)
        role_tokens[role] = [seqs[i] for i in idx]
        print(f"{role:10s}: capped to {len(role_tokens[role])}")


def tokenize_dataset():
    print(f"tokenize_midi.py path: {os.path.abspath(__file__)}")
    print(f"TARGET_GENRES={TARGET_GENRES}, TRACK_LIMIT_PER_FILE={TRACK_LIMIT_PER_FILE}, ROLE_SEQUENCE_CAP={ROLE_SEQUENCE_CAP}")

    with open(os.path.join(META_DIR, "files.json")) as f:
        files = json.load(f)

    role_tokens = defaultdict(list)
    vocab = set()
    errors = []

    valid_entries = [
        e for e in files
        if e["status"] == "ok" and ((not TARGET_GENRES) or e.get("genre", "").strip().lower() in TARGET_GENRES)
    ]
    print(f" Токенизация {len(valid_entries)} валидных MIDI файлов...\n")

    for entry in tqdm(valid_entries, desc="Токенизация", unit="файл"):
        genre = entry["genre"]
        midi_path = os.path.join(RAW_DIR, genre, entry["file"])

        try:
            score = m21.converter.parse(midi_path, forceSource=True)
            key_token = detect_key_token(score)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            errors.append({"file": entry["file"], "error": str(e)[:100]})
            continue

        try:
            for role in ["melody", "bass", "chords"]:
                tracks = entry[f"{role}_tracks"]
                if not tracks:
                    continue

                lim = TRACK_LIMIT_PER_FILE.get(role)
                if lim is not None and len(tracks) > lim:
                    tracks = tracks[:lim]

                for tid in tracks:
                    part = score.parts[tid]

                    events = extract_events(part)
                    if not events:
                        continue

                    if not role_quality_pass(role, events):
                        continue

                    tokens = [
                        f"<ROLE_{role.upper()}>",
                        f"<GENRE_{genre.upper()}>",
                        key_token,
                    ]
                    tokens += events_to_tokens(events)

                    role_tokens[role].append(tokens)
                    vocab.update(tokens)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            errors.append({"file": entry["file"], "error": f"token_error: {str(e)[:100]}"})
            continue

    cap_role_sequences(role_tokens)

    print("\n" + "=" * 50)
    print("ТОКЕНИЗАЦИЯ ЗАВЕРШЕНА")
    print("=" * 50)

    for role in ["chords", "melody", "bass"]:
        seqs = role_tokens.get(role, [])
        np.save(os.path.join(TOKENS_DIR, f"{role}.npy"), np.array(seqs, dtype=object))
        print(f"{role:10s}: {len(seqs):6d} sequences")

    SPECIAL_TOKENS = [
        "<PAD>",
        "<BOS>",
        "<EOS>",
        "<UNK>",
        "<ROLE_MELODY>",
        "<ROLE_BASS>",
        "<ROLE_CHORDS>",
        "<GENRE_TRAP>",
        *KEY_TOKENS,
    ]

    vocab_clean = [t for t in vocab if t not in SPECIAL_TOKENS]
    vocab_list = SPECIAL_TOKENS + sorted(vocab_clean)

    token2id = {t: i for i, t in enumerate(vocab_list)}
    id2token = {i: t for t, i in token2id.items()}

    vocab_dict = {
        "token2id": token2id,
        "id2token": id2token,
        "size": len(token2id)
    }

    with open(os.path.join(BASE_DIR, "processed", "vocab.json"), "w") as f:
        json.dump(vocab_dict, f, indent=2)

    print(f"\n📊 Vocab size: {len(vocab_list)} tokens")

    if errors:
        print(f" Ошибок при токенизации: {len(errors)}")
        with open(os.path.join(BASE_DIR, "processed", "tokenize_errors.json"), "w") as f:
            json.dump(errors, f, indent=2)

    print("=" * 50)


if __name__ == "__main__":
    tokenize_dataset()
