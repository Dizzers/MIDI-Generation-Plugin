import os
import json
import warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
import music21 as m21
from tqdm import tqdm
import subprocess
import sys

warnings.filterwarnings("ignore", category=m21.midi.translate.TranslateWarning)

TIMEOUT_SECONDS = 10

MIN_TOTAL_NOTES = 5
MIN_DURATION_SEC = 1
MIN_TRACK_NOTES = 8
TARGET_GENRES = {"trap"}

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUT_DIR = str(BASE_DIR / "processed")


def ensure_dirs():
    os.makedirs(os.path.join(OUTPUT_DIR, "meta"), exist_ok=True)


def _track_note_events(part):
    """Extract per-note events from both Note and Chord objects."""
    events = []
    for obj in part.flatten().notes:
        start = float(obj.offset)
        dur = max(1e-3, float(obj.quarterLength))

        if obj.isNote:
            pitches = [int(obj.pitch.midi)]
        elif obj.isChord:
            pitches = [int(p.midi) for p in obj.pitches]
        else:
            continue

        for pitch in pitches:
            events.append((start, start + dur, pitch))
    return events


def _polyphony_features(events):
    if not events:
        return 0.0, 0.0

    sweep = []
    for start, end, _ in events:
        sweep.append((start, +1))
        sweep.append((end, -1))

    sweep.sort(key=lambda x: (x[0], -x[1]))

    cur = 0
    peak = 0
    weighted_sum = 0.0
    total_time = 0.0
    prev_t = sweep[0][0]

    for t, delta in sweep:
        dt = max(0.0, t - prev_t)
        if dt > 0:
            weighted_sum += cur * dt
            total_time += dt
        cur += delta
        if cur > peak:
            peak = cur
        prev_t = t

    mean_poly = (weighted_sum / total_time) if total_time > 0 else float(peak)
    return float(peak), float(mean_poly)


def _onset_chord_ratio(events):
    if not events:
        return 0.0

    onset_map = defaultdict(int)
    for start, _, _ in events:
        key = round(start, 3)
        onset_map[key] += 1

    if not onset_map:
        return 0.0

    chord_like = sum(1 for c in onset_map.values() if c >= 2)
    return chord_like / len(onset_map)


def analyze_track(part):
    events = _track_note_events(part)
    if len(events) < MIN_TRACK_NOTES:
        return None

    pitches = [p for _, _, p in events]
    if not pitches:
        return None

    peak_poly, mean_poly = _polyphony_features(events)
    chord_onset_ratio = _onset_chord_ratio(events)

    low_ratio = sum(1 for p in pitches if p <= 52) / len(pitches)
    high_ratio = sum(1 for p in pitches if p >= 72) / len(pitches)

    stats = {
        "avg_pitch": float(np.mean(pitches)),
        "min_pitch": int(np.min(pitches)),
        "max_pitch": int(np.max(pitches)),
        "pitch_range": int(np.max(pitches) - np.min(pitches)),
        "note_count": len(pitches),
        "duration": float(part.highestTime),
        "polyphony_peak": peak_poly,
        "polyphony_mean": mean_poly,
        "chord_onset_ratio": float(chord_onset_ratio),
        "low_pitch_ratio": float(low_ratio),
        "high_pitch_ratio": float(high_ratio),
    }
    return stats


def _clamp01(x):
    return float(max(0.0, min(1.0, x)))


def classify_role(stats):
    """
    Return one of: melody, bass, chords, or None.
    We prefer precision over recall to reduce noisy role labels.
    """
    avg_pitch = stats["avg_pitch"]
    pitch_range = stats["pitch_range"]
    poly_peak = stats["polyphony_peak"]
    poly_mean = stats["polyphony_mean"]
    chord_ratio = stats["chord_onset_ratio"]
    low_ratio = stats["low_pitch_ratio"]
    high_ratio = stats["high_pitch_ratio"]

    score_chords = (
        0.45 * _clamp01((poly_mean - 1.10) / 1.2)
        + 0.25 * _clamp01((poly_peak - 2.0) / 4.0)
        + 0.30 * _clamp01((chord_ratio - 0.15) / 0.6)
    )

    score_bass = (
        0.50 * low_ratio
        + 0.20 * _clamp01((62.0 - avg_pitch) / 20.0)
        + 0.20 * _clamp01((1.35 - poly_mean) / 0.6)
        + 0.10 * _clamp01((28.0 - pitch_range) / 24.0)
    )

    score_melody = (
        0.45 * high_ratio
        + 0.25 * _clamp01((avg_pitch - 58.0) / 24.0)
        + 0.20 * _clamp01((1.35 - poly_mean) / 0.6)
        + 0.10 * _clamp01((pitch_range - 6.0) / 24.0)
    )

    scores = {
        "chords": score_chords,
        "bass": score_bass,
        "melody": score_melody,
    }

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_role, best_score = ranked[0]
    second_score = ranked[1][1]

    # Hard gates to avoid obvious role confusion.
    if best_role == "chords":
        if not (poly_mean >= 1.25 and poly_peak >= 2 and chord_ratio >= 0.18):
            return None
    elif best_role == "bass":
        if not (low_ratio >= 0.45 and poly_mean <= 1.35):
            return None
    elif best_role == "melody":
        if not (high_ratio >= 0.30 and poly_mean <= 1.35):
            return None

    # Reject ambiguous tracks.
    if best_score < 0.42:
        return None
    if (best_score - second_score) < 0.08:
        return None

    return best_role


def process_midi_file(path, genre):
    result = {
        "file": os.path.basename(path),
        "genre": genre,
        "melody_tracks": [],
        "bass_tracks": [],
        "chords_tracks": [],
        "status": "ok",
        "reason": None
    }

    try:
        score = m21.converter.parse(path, forceSource=True)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        error_msg = str(e)[:100]
        result["status"] = "error"
        result["reason"] = f"parse_error: {error_msg}"
        return result

    try:
        parts = score.parts
    except Exception as e:
        result["status"] = "error"
        result["reason"] = f"parts_error: {str(e)[:100]}"
        return result

    if not parts:
        result["status"] = "rejected"
        result["reason"] = "no_parts"
        return result

    total_notes = 0
    max_duration = 0

    for idx, part in enumerate(parts):
        stats = analyze_track(part)
        if stats is None:
            continue

        total_notes += stats["note_count"]
        max_duration = max(max_duration, stats["duration"])

        role = classify_role(stats)
        if role == "melody":
            result["melody_tracks"].append(idx)
        elif role == "bass":
            result["bass_tracks"].append(idx)
        elif role == "chords":
            result["chords_tracks"].append(idx)

    if total_notes < MIN_TOTAL_NOTES:
        result["status"] = "rejected"
        result["reason"] = "too_few_notes"
        return result

    if max_duration < MIN_DURATION_SEC:
        result["status"] = "rejected"
        result["reason"] = "too_short"
        return result

    if not (result["melody_tracks"] or result["bass_tracks"] or result["chords_tracks"]):
        result["status"] = "rejected"
        result["reason"] = "no_roles_detected"

    return result


def process_dataset(root_dir):
    ensure_dirs()

    files_log = []
    stats = defaultdict(int)
    errors = []
    timeouts = []

    all_files = []
    for genre in os.listdir(root_dir):
        genre_norm = genre.strip().lower()
        if TARGET_GENRES and genre_norm not in TARGET_GENRES:
            continue

        genre_path = os.path.join(root_dir, genre)
        if not os.path.isdir(genre_path):
            continue

        for fname in os.listdir(genre_path):
            if fname.lower().endswith(".mid"):
                full_path = os.path.join(genre_path, fname)
                all_files.append((full_path, genre_norm))

    print(f" Найдено {len(all_files)} MIDI файлов")
    print(" Начинаем обработку...\n")

    for full_path, genre in tqdm(all_files, desc="Обработка MIDI", unit="файл"):
        try:
            result = subprocess.run(
                [sys.executable, str(BASE_DIR / "_parse_single_midi.py"), full_path],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                cwd=str(PROJECT_ROOT)
            )

            if result.returncode != 0:
                stats["error"] += 1
                errors.append({
                    "file": os.path.basename(full_path),
                    "reason": result.stdout.strip()[:100]
                })
                continue

        except subprocess.TimeoutExpired:
            stats["error"] += 1
            timeouts.append(full_path)
            errors.append({
                "file": os.path.basename(full_path),
                "reason": f"timeout_after_{TIMEOUT_SECONDS}s"
            })
            continue
        except Exception as e:
            stats["error"] += 1
            errors.append({
                "file": os.path.basename(full_path),
                "reason": f"subprocess_error: {str(e)[:100]}"
            })
            continue

        try:
            res = process_midi_file(full_path, genre)
            files_log.append(res)

            if res["status"] == "ok":
                stats["processed"] += 1
                if res["melody_tracks"]:
                    stats["melody"] += 1
                if res["bass_tracks"]:
                    stats["bass"] += 1
                if res["chords_tracks"]:
                    stats["chords"] += 1
            elif res["status"] == "rejected":
                stats["rejected"] += 1
            else:
                stats["error"] += 1
                errors.append(res)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            stats["error"] += 1
            errors.append({"file": os.path.basename(full_path), "error": str(e)[:100]})

    with open(os.path.join(OUTPUT_DIR, "meta", "files.json"), "w") as f:
        json.dump(files_log, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "meta", "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "meta", "errors.json"), "w") as f:
        json.dump(errors, f, indent=2)

    if timeouts:
        with open(os.path.join(OUTPUT_DIR, "meta", "timeout_files.txt"), "w") as f:
            for pf in timeouts:
                f.write(pf + "\n")

    print("\n" + "=" * 50)
    print("ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 50)
    print(f"Обработано:      {stats['processed']:6d}")
    print(f"Отклонено:       {stats['rejected']:6d}")
    print(f"Ошибок (включая timeout): {stats['error']:6d}")
    if timeouts:
        print(f"\n⚠️  Зависших файлов: {len(timeouts)}")
        print("   Сохранены в: dataset/processed/meta/timeout_files.txt")
    print("=" * 50)


if __name__ == "__main__":
    RAW_MIDI_DIR = str(BASE_DIR / "midi_raw")
    process_dataset(RAW_MIDI_DIR)
