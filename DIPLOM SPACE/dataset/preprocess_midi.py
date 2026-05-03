"""Walk dataset/midi_raw, parse each MIDI, filter by note count/duration,
dedupe by content hash, estimate musical key via Krumhansl, and write
processed/midi_meta.jsonl.

Usage:
    python -m dataset.preprocess_midi
"""
from __future__ import annotations

import hashlib
import json
import signal
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import music21 as m21
from tqdm import tqdm

warnings.filterwarnings("ignore", category=m21.midi.translate.TranslateWarning)

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "midi_raw"
PROCESSED_DIR = BASE_DIR / "processed"
META_PATH = PROCESSED_DIR / "midi_meta.jsonl"
ERRORS_PATH = PROCESSED_DIR / "preprocess_errors.json"
STATS_PATH = PROCESSED_DIR / "preprocess_stats.json"

PARSE_TIMEOUT_SECONDS = 15
MIN_NOTES = 16
MIN_DURATION_SECONDS = 1.0
MAX_DURATION_SECONDS = 600.0

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class ParseTimeoutError(TimeoutError):
    pass


def _timeout_handler(signum, frame):  # noqa: ARG001
    raise ParseTimeoutError(f"timeout_after_{PARSE_TIMEOUT_SECONDS}s")


def parse_with_timeout(midi_path: Path) -> m21.stream.Score:
    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(PARSE_TIMEOUT_SECONDS)
    try:
        return m21.converter.parse(str(midi_path), forceSource=True)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def krumhansl_key_token(score: m21.stream.Score) -> str:
    """Estimate key with Krumhansl-Schmuckler and map to <KEY_*> token."""
    try:
        key = score.analyze("Krumhansl")
        tonic = key.tonic.name
        if tonic.endswith("-"):
            equivalents = {"D-": "C#", "E-": "D#", "G-": "F#", "A-": "G#", "B-": "A#", "C-": "B", "F-": "E"}
            tonic = equivalents.get(tonic, tonic.replace("-", "b"))
        if tonic not in PITCH_CLASSES:
            try:
                pitch_class = m21.pitch.Pitch(key.tonic.name).pitchClass
                tonic = PITCH_CLASSES[pitch_class % 12]
            except Exception:
                return "<KEY_UNKNOWN>"
        mode = "MAJ" if (key.mode or "major").lower().startswith("maj") else "MIN"
        token = f"<KEY_{tonic}_{mode}>"
        return token
    except Exception:
        return "<KEY_UNKNOWN>"


def collect_notes(score: m21.stream.Score):
    """Return list of (offset_quarters, pitch, velocity, duration_quarters)."""
    events = []
    for item in score.flatten().notes:
        offset = float(item.offset)
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
            events.append((offset, pitch, velocity, duration))
    events.sort(key=lambda row: (row[0], row[1]))
    return events


def file_digest(events) -> str:
    """Stable hash for dedup: pitches and (rounded) onsets only."""
    payload = ",".join(f"{round(o, 3)}:{p}" for o, p, _, _ in events)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@dataclass
class MidiMeta:
    path: str
    note_count: int
    duration_seconds: float
    duration_quarters: float
    avg_tempo_bpm: float
    key_token: str
    digest: str
    pitch_min: int
    pitch_max: int


def estimate_seconds_per_quarter(score: m21.stream.Score) -> float:
    """Average seconds per quarter from MetronomeMark events; fallback 0.5 (=120 BPM)."""
    marks = list(score.flatten().getElementsByClass("MetronomeMark"))
    if not marks:
        return 0.5
    total = 0.0
    n = 0
    for mark in marks:
        try:
            bpm = float(mark.number or 120.0)
            if bpm > 0:
                total += 60.0 / bpm
                n += 1
        except Exception:
            continue
    if n == 0:
        return 0.5
    return total / n


def iter_midi_files() -> Iterator[Path]:
    if not RAW_DIR.exists():
        return
    for path in sorted(RAW_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".mid", ".midi"}:
            yield path


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    files = list(iter_midi_files())
    if not files:
        print(f"No MIDI files under {RAW_DIR}. Place dataset there first.")
        return 1

    print(f"Found {len(files)} MIDI files under {RAW_DIR}")

    seen_digests: set[str] = set()
    errors: list[dict] = []
    kept = 0
    skipped_short = 0
    skipped_long = 0
    skipped_few_notes = 0
    skipped_dup = 0
    timeouts = 0
    key_counts: dict[str, int] = {}

    with open(META_PATH, "w", encoding="utf-8") as out:
        for midi_path in tqdm(files, desc="preprocess", unit="file"):
            rel = str(midi_path.relative_to(RAW_DIR))
            try:
                score = parse_with_timeout(midi_path)
            except ParseTimeoutError as exc:
                timeouts += 1
                errors.append({"file": rel, "error": str(exc)})
                continue
            except Exception as exc:
                errors.append({"file": rel, "error": str(exc)[:200]})
                continue

            events = collect_notes(score)
            if len(events) < MIN_NOTES:
                skipped_few_notes += 1
                continue

            sec_per_q = estimate_seconds_per_quarter(score)
            try:
                duration_quarters = float(score.highestTime)
            except Exception:
                duration_quarters = max(o + d for o, _, _, d in events)
            duration_seconds = duration_quarters * sec_per_q

            if duration_seconds < MIN_DURATION_SECONDS:
                skipped_short += 1
                continue
            if duration_seconds > MAX_DURATION_SECONDS:
                skipped_long += 1
                continue

            digest = file_digest(events)
            if digest in seen_digests:
                skipped_dup += 1
                continue
            seen_digests.add(digest)

            key_token = krumhansl_key_token(score)
            key_counts[key_token] = key_counts.get(key_token, 0) + 1

            pitches = [p for _, p, _, _ in events]
            avg_bpm = 60.0 / sec_per_q if sec_per_q > 0 else 120.0

            meta = MidiMeta(
                path=rel,
                note_count=len(events),
                duration_seconds=round(duration_seconds, 3),
                duration_quarters=round(duration_quarters, 3),
                avg_tempo_bpm=round(avg_bpm, 2),
                key_token=key_token,
                digest=digest,
                pitch_min=min(pitches),
                pitch_max=max(pitches),
            )
            out.write(json.dumps(asdict(meta), ensure_ascii=False) + "\n")
            kept += 1

    stats = {
        "files_seen": len(files),
        "kept": kept,
        "skipped_few_notes": skipped_few_notes,
        "skipped_short": skipped_short,
        "skipped_long": skipped_long,
        "skipped_dup": skipped_dup,
        "errors": len(errors),
        "timeouts": timeouts,
        "key_distribution": dict(sorted(key_counts.items(), key=lambda kv: -kv[1])),
        "min_notes_threshold": MIN_NOTES,
        "min_duration_seconds": MIN_DURATION_SECONDS,
        "max_duration_seconds": MAX_DURATION_SECONDS,
    }
    with open(STATS_PATH, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    if errors:
        with open(ERRORS_PATH, "w", encoding="utf-8") as handle:
            json.dump(errors, handle, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
