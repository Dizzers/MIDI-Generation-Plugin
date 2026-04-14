import os
import json
import warnings
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


def analyze_track(part):
    """Analyze track and return basic statistics without role classification."""
    events = _track_note_events(part)
    if len(events) < MIN_TRACK_NOTES:
        return None

    pitches = [p for _, _, p in events]
    if not pitches:
        return None

    peak_poly, mean_poly = _polyphony_features(events)

    stats = {
        "note_count": len(pitches),
        "duration": float(part.highestTime),
        "avg_pitch": float(np.mean(pitches)),
        "min_pitch": int(np.min(pitches)),
        "max_pitch": int(np.max(pitches)),
        "pitch_range": int(np.max(pitches) - np.min(pitches)),
        "polyphony_peak": peak_poly,
        "polyphony_mean": mean_poly,
    }
    return stats


def process_midi_file_full(path, genre):
    """
    Process MIDI file and extract all tracks without role classification.
    Returns metadata for ALL valid tracks found.
    """
    result = {
        "file": os.path.basename(path),
        "genre": genre,
        "tracks": [],  # List of all valid tracks with their stats
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

        # Store all valid tracks
        track_info = {
            "index": idx,
            "note_count": stats["note_count"],
            "duration": stats["duration"],
            "avg_pitch": stats["avg_pitch"],
            "min_pitch": stats["min_pitch"],
            "max_pitch": stats["max_pitch"],
            "pitch_range": stats["pitch_range"],
            "polyphony_peak": stats["polyphony_peak"],
            "polyphony_mean": stats["polyphony_mean"],
        }
        result["tracks"].append(track_info)

    if total_notes < MIN_TOTAL_NOTES:
        result["status"] = "rejected"
        result["reason"] = "too_few_notes"
        return result

    if max_duration < MIN_DURATION_SEC:
        result["status"] = "rejected"
        result["reason"] = "too_short"
        return result

    if not result["tracks"]:
        result["status"] = "rejected"
        result["reason"] = "no_valid_tracks"

    return result


def process_dataset_full(root_dir):
    """Process all MIDI files in the dataset without role filtering."""
    ensure_dirs()

    files_log = []
    stats = {
        "total_files": 0,
        "processed": 0,
        "rejected": 0,
        "error": 0,
    }
    errors = []
    timeouts = []

    all_files = []
    for genre in os.listdir(root_dir):
        genre_norm = genre.strip().lower()
        genre_path = os.path.join(root_dir, genre)
        if not os.path.isdir(genre_path):
            continue

        for fname in os.listdir(genre_path):
            if fname.lower().endswith(".mid"):
                full_path = os.path.join(genre_path, fname)
                all_files.append((full_path, genre_norm))

    stats["total_files"] = len(all_files)
    print(f" Найдено {len(all_files)} MIDI файлов")
    print(" Начинаем полную обработку (без фильтрации по ролям)...\n")

    for full_path, genre in tqdm(all_files, desc="Полная обработка MIDI", unit="файл"):
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
            res = process_midi_file_full(full_path, genre)
            files_log.append(res)

            if res["status"] == "ok":
                stats["processed"] += 1
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

    with open(os.path.join(OUTPUT_DIR, "meta", "files_full.json"), "w") as f:
        json.dump(files_log, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "meta", "stats_full.json"), "w") as f:
        json.dump(stats, f, indent=2)

    if errors:
        with open(os.path.join(OUTPUT_DIR, "meta", "errors_full.json"), "w") as f:
            json.dump(errors, f, indent=2)

    if timeouts:
        with open(os.path.join(OUTPUT_DIR, "meta", "timeout_files_full.txt"), "w") as f:
            for pf in timeouts:
                f.write(pf + "\n")

    print("\n" + "=" * 50)
    print("ПОЛНАЯ ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 50)
    print(f"Всего файлов:    {stats['total_files']:6d}")
    print(f"Обработано:      {stats['processed']:6d}")
    print(f"Отклонено:       {stats['rejected']:6d}")
    print(f"Ошибок:          {stats['error']:6d}")
    if timeouts:
        print(f"\n⚠️  Зависших файлов: {len(timeouts)}")
        print("   Сохранены в: dataset/processed/meta/timeout_files_full.txt")
    print("=" * 50)


if __name__ == "__main__":
    RAW_MIDI_DIR = str(BASE_DIR / "midi_raw")
    process_dataset_full(RAW_MIDI_DIR)
