import json
import signal
from pathlib import Path

import music21 as m21
import numpy as np
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "midi_raw"
PROCESSED_DIR = BASE_DIR / "processed"
TOKENS_DIR = PROCESSED_DIR / "tokens"

TOKENS_DIR.mkdir(parents=True, exist_ok=True)

TIME_SHIFT_RESOLUTION = 0.05
VELOCITY_BINS = 8
TARGET_GENRES = {"trap", "classical"}
MIN_TOTAL_NOTES = 16
MAX_FILES_PER_GENRE = None
PARSE_TIMEOUT_SECONDS = 15
SEED = 42
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

KEY_TOKENS = [
    *(f"<KEY_{pc}_MAJ>" for pc in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]),
    *(f"<KEY_{pc}_MIN>" for pc in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]),
    "<KEY_UNKNOWN>",
]
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class MidiParseTimeoutError(TimeoutError):
    pass


def velocity_bin(value):
    bucket = int((float(value) / 127.0) * (VELOCITY_BINS - 1))
    return max(0, min(VELOCITY_BINS - 1, bucket))


def quantize_time(delta):
    steps = int(round(float(delta) / TIME_SHIFT_RESOLUTION))
    return max(1, steps)


def detect_key_token(score):
    try:
        key = score.analyze("Krumhansl")
        tonic = key.tonic.name.replace("-", "b")
        mode = "MAJ" if (key.mode or "major").lower().startswith("maj") else "MIN"
        pitch_class = m21.pitch.Pitch(tonic).pitchClass
        root = NOTE_NAMES[int(pitch_class) % 12]
        token = f"<KEY_{root}_{mode}>"
        if token in KEY_TOKENS:
            return token
    except Exception:
        pass
    return "<KEY_UNKNOWN>"


def iter_midi_files():
    for genre_dir in sorted(RAW_DIR.iterdir()):
        if not genre_dir.is_dir():
            continue
        genre = genre_dir.name.strip().lower()
        if TARGET_GENRES and genre not in TARGET_GENRES:
            continue

        files = sorted(
            p for p in genre_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".mid", ".midi"}
        )
        if MAX_FILES_PER_GENRE is not None:
            files = files[:MAX_FILES_PER_GENRE]

        for midi_path in files:
            yield genre.upper(), midi_path


def _timeout_handler(signum, frame):
    raise MidiParseTimeoutError(f"timeout_after_{PARSE_TIMEOUT_SECONDS}s")


def parse_score_with_timeout(midi_path):
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(PARSE_TIMEOUT_SECONDS)
    try:
        return m21.converter.parse(str(midi_path), forceSource=True)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def extract_full_events(score):
    events = []
    for item in score.flatten().notes:
        start = float(item.offset)
        duration = max(1e-3, float(item.quarterLength))

        if item.isNote:
            pitches = [int(item.pitch.midi)]
            velocity = int(item.volume.velocity or 64)
        elif item.isChord:
            pitches = [int(p.midi) for p in item.pitches]
            velocity = int(item.volume.velocity or 64)
        else:
            continue

        for pitch in pitches:
            events.append((start, 1, "NOTE_ON", pitch, velocity))
            events.append((start + duration, 0, "NOTE_OFF", pitch, None))

    events.sort(key=lambda row: (row[0], row[1], row[3]))
    return events


def events_to_tokens(events):
    tokens = []
    last_time = 0.0

    for time, _, event_type, pitch, velocity in events:
        delta = time - last_time
        if delta > 0:
            tokens.append(f"TIME_SHIFT_{quantize_time(delta):#06x}")

        if event_type == "NOTE_ON":
            tokens.append(f"VELOCITY_{velocity_bin(velocity or 64):#02x}")
            tokens.append(f"NOTE_ON_{int(pitch):#04x}")
        else:
            tokens.append(f"NOTE_OFF_{int(pitch):#04x}")

        last_time = time

    return tokens


def tokenize_dataset():
    full_sequences = []
    full_sequences_with_meta = []
    vocab = set()
    errors = []
    timeout_files = []
    files_per_genre = {}
    sequences_per_genre = {}
    tokens_per_genre = {}

    midi_files = list(iter_midi_files())
    print(f"Tokenizing {len(midi_files)} full MIDI files...")

    progress = tqdm(midi_files, desc="Full MIDI tokenization", unit="file")
    for genre, midi_path in progress:
        files_per_genre[genre] = files_per_genre.get(genre, 0) + 1
        progress.set_postfix_str(midi_path.name[:48])
        try:
            score = parse_score_with_timeout(midi_path)
            key_token = detect_key_token(score)
            events = extract_full_events(score)
            note_on_count = sum(1 for row in events if row[2] == "NOTE_ON")
            if note_on_count < MIN_TOTAL_NOTES:
                continue

            tokens = [f"<GENRE_{genre}>", key_token]
            tokens.extend(events_to_tokens(events))
            full_sequences.append(tokens)
            full_sequences_with_meta.append({"genre": genre, "path": str(midi_path), "tokens": tokens})
            vocab.update(tokens)
            sequences_per_genre[genre] = sequences_per_genre.get(genre, 0) + 1
            tokens_per_genre[genre] = tokens_per_genre.get(genre, 0) + len(tokens)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            errors.append({"file": str(midi_path), "error": str(exc)[:200]})
            if "timeout_after_" in str(exc):
                timeout_files.append(str(midi_path))

    # Save full set for generation primers / analysis
    np_payload = np.array(full_sequences, dtype=object)
    np.save(TOKENS_DIR / "full.npy", np_payload)

    # Split by FILE (not chunks) to prevent leakage across train/val/test.
    # Stratify by genre to keep both genres represented in each split.
    by_genre = {}
    for row in full_sequences_with_meta:
        by_genre.setdefault(row["genre"], []).append(row)

    rng = __import__("random").Random(SEED)
    splits = {"train": [], "val": [], "test": []}
    for genre, rows in sorted(by_genre.items()):
        rng.shuffle(rows)
        n = len(rows)
        n_val = max(1, int(round(n * VAL_SPLIT))) if n >= 10 else max(0, int(round(n * VAL_SPLIT)))
        n_test = max(1, int(round(n * TEST_SPLIT))) if n >= 10 else max(0, int(round(n * TEST_SPLIT)))
        n_train = max(1, n - n_val - n_test) if n > 0 else 0
        splits["train"].extend(rows[:n_train])
        splits["val"].extend(rows[n_train:n_train + n_val])
        splits["test"].extend(rows[n_train + n_val:n_train + n_val + n_test])

    for split_name in ["train", "val", "test"]:
        split_tokens = [row["tokens"] for row in splits[split_name]]
        np.save(TOKENS_DIR / f"full_{split_name}.npy", np.array(split_tokens, dtype=object))
        with open(PROCESSED_DIR / f"full_{split_name}_files.json", "w") as handle:
            json.dump(
                [{"genre": row["genre"], "path": row["path"]} for row in splits[split_name]],
                handle,
                indent=2,
                ensure_ascii=False,
            )

    special_tokens = [
        "<PAD>",
        "<BOS>",
        "<EOS>",
        "<UNK>",
        "<GENRE_TRAP>",
        "<GENRE_CLASSICAL>",
        *KEY_TOKENS,
    ]
    vocab_clean = [token for token in vocab if token not in special_tokens]
    vocab_list = special_tokens + sorted(vocab_clean)

    token2id = {token: idx for idx, token in enumerate(vocab_list)}
    id2token = {idx: token for token, idx in token2id.items()}

    with open(PROCESSED_DIR / "vocab.json", "w") as handle:
        json.dump({"token2id": token2id, "id2token": id2token, "size": len(token2id)}, handle, indent=2)

    stats = {
        "mode": "full_midi",
        "files_seen": len(midi_files),
        "sequences_kept": len(full_sequences),
        "errors": len(errors),
        "vocab_size": len(token2id),
        "genres": sorted({seq[0] for seq in full_sequences}) if full_sequences else [],
        "split_seed": SEED,
        "train_sequences": len(splits["train"]) if full_sequences else 0,
        "val_sequences": len(splits["val"]) if full_sequences else 0,
        "test_sequences": len(splits["test"]) if full_sequences else 0,
        "files_per_genre": files_per_genre,
        "sequences_per_genre": sequences_per_genre,
        "avg_tokens_per_genre": {
            genre: round(tokens_per_genre[genre] / max(1, sequences_per_genre.get(genre, 0)), 2)
            for genre in sorted(tokens_per_genre)
        },
    }
    with open(PROCESSED_DIR / "full_tokenize_stats.json", "w") as handle:
        json.dump(stats, handle, indent=2)

    if errors:
        with open(PROCESSED_DIR / "tokenize_errors.json", "w") as handle:
            json.dump(errors, handle, indent=2)
    if timeout_files:
        with open(PROCESSED_DIR / "tokenize_timeout_files.txt", "w") as handle:
            handle.write("\n".join(timeout_files) + "\n")

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    tokenize_dataset()
