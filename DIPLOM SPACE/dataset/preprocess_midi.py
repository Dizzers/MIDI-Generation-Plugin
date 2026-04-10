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
import pickle

warnings.filterwarnings("ignore", category=m21.midi.translate.TranslateWarning)

TIMEOUT_SECONDS = 10

MIN_TOTAL_NOTES = 5
MIN_DURATION_SEC = 1

MELODY_PITCH_MIN = 64
BASS_PITCH_MAX = 75
CHORD_POLYPHONY_MIN = 2.0

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUT_DIR = str(BASE_DIR / "processed")

def ensure_dirs():
    os.makedirs(os.path.join(OUTPUT_DIR, "meta"), exist_ok=True)


def get_note_stats(notes):
    pitches = [n.pitch.midi for n in notes if n.isNote]
    if not pitches:
        return None

    durations = [n.quarterLength for n in notes if n.isNote]

    return {
        "avg_pitch": float(np.mean(pitches)),
        "min_pitch": int(np.min(pitches)),
        "max_pitch": int(np.max(pitches)),
        "pitch_range": int(np.max(pitches) - np.min(pitches)),
        "note_count": len(pitches),
        "avg_duration": float(np.mean(durations))
    }


def estimate_polyphony(part):
    events = []
    for n in part.flat.notes:
        if n.isNote:
            start = float(n.offset)
            end = float(n.offset + n.quarterLength)
            events.append((start, 1))
            events.append((end, -1))

    if not events:
        return 0.0

    events.sort()
    current = 0
    max_poly = 0
    for _, e in events:
        current += e
        max_poly = max(max_poly, current)

    return float(max_poly)


def analyze_track(part):
    notes = list(part.flat.notes)
    if len(notes) < 5:
        return None

    stats = get_note_stats(notes)
    if stats is None:
        return None

    polyphony = estimate_polyphony(part)

    stats["polyphony"] = polyphony
    stats["duration"] = float(part.highestTime)

    return stats


def classify_role(stats):
    """
    Возвращает: "melody", "bass", "chords" или None
    """
    if stats["avg_pitch"] >= MELODY_PITCH_MIN and stats["polyphony"] <= 1.5:
        return "melody"

    if stats["avg_pitch"] <= BASS_PITCH_MAX and stats["polyphony"] <= 1.5:
        return "bass"

    if stats["polyphony"] >= CHORD_POLYPHONY_MIN:
        return "chords"

    return None


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
        genre_path = os.path.join(root_dir, genre)
        if not os.path.isdir(genre_path):
            continue
        
        for fname in os.listdir(genre_path):
            if fname.lower().endswith(".mid"):
                full_path = os.path.join(genre_path, fname)
                all_files.append((full_path, genre))
    
    print(f" Найдено {len(all_files)} MIDI файлов")
    print(f" Начинаем обработку...\n")

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
        print(f"   Сохранены в: dataset/processed/meta/timeout_files.txt")
    print("=" * 50)


if __name__ == "__main__":
    RAW_MIDI_DIR = str(BASE_DIR / "midi_raw")
    process_dataset(RAW_MIDI_DIR)
