import json
from pathlib import Path
from collections import defaultdict
import numpy as np

print("АНАЛИЗ КАЧЕСТВА ДАТАСЕТА\n")


META_DIR = Path("dataset/processed/meta")
VOCAB_PATH = Path("dataset/processed/vocab.json")


print("=" * 50)
print("1️⃣  СТАТИСТИКА ФАЙЛОВ")
print("=" * 50)

with open(META_DIR / "files.json") as f:
    files = json.load(f)

total = len(files)
ok = sum(1 for f in files if f["status"] == "ok")
rejected = sum(1 for f in files if f["status"] == "rejected")
errors = sum(1 for f in files if f["status"] == "error")

print(f"Всего файлов: {total}")
print(f"  ✓ OK: {ok} ({ok/total*100:.1f}%)")
print(f"  ✗ Отклонено: {rejected} ({rejected/total*100:.1f}%)")
print(f"  ✗ Ошибок: {errors} ({errors/total*100:.1f}%)")


print("\n" + "=" * 50)
print("2️⃣  БАЛАНС ДАННЫХ ПО РОЛЯМ")
print("=" * 50)

role_counts = defaultdict(int)
for f in files:
    if f["status"] == "ok":
        for role in ["melody_tracks", "bass_tracks", "chords_tracks"]:
            role_counts[role.replace("_tracks", "")] += len(f[role])

total_tracks = sum(role_counts.values())
for role, count in sorted(role_counts.items()):
    pct = count / total_tracks * 100
    bar = "█" * int(pct / 5)
    print(f"{role:10s}: {count:5d} ({pct:5.1f}%) {bar}")


print("\n" + "=" * 50)
print("3️⃣  РАСПРЕДЕЛЕНИЕ ПО ЖАНРАМ")
print("=" * 50)

genre_counts = defaultdict(int)
for f in files:
    if f["status"] == "ok":
        genre_counts[f["genre"]] += 1

for genre, count in sorted(genre_counts.items(), key=lambda x: -x[1]):
    pct = count / ok * 100
    print(f"{genre:15s}: {count:5d} ({pct:5.1f}%)")


print("\n" + "=" * 50)
print("4️⃣  СТАТИСТИКА СЛОВАРЯ (VOCAB)")
print("=" * 50)

with open(VOCAB_PATH) as f:
    vocab = json.load(f)

token2id = vocab["token2id"]
print(f"Размер vocab: {len(token2id)} токенов")


token_types = defaultdict(int)
for token in token2id.keys():
    if token.startswith("<"):
        token_types["special"] += 1
    elif token.startswith("NOTE_ON"):
        token_types["note_on"] += 1
    elif token.startswith("NOTE_OFF"):
        token_types["note_off"] += 1
    elif token.startswith("VELOCITY"):
        token_types["velocity"] += 1
    elif token.startswith("TIME_SHIFT"):
        token_types["time_shift"] += 1
    else:
        token_types["other"] += 1

print("\nТипы токенов:")
for ttype, count in sorted(token_types.items(), key=lambda x: -x[1]):
    print(f"  {ttype:15s}: {count:5d}")


print("\n" + "=" * 50)
print("💡 РЕКОМЕНДАЦИИ")
print("=" * 50)

if role_counts["melody"] < role_counts["chords"] * 0.1:
    print("ДИСБАЛАНС: Мало MELODY треков")
    print("Решение: Data augmentation, weighted sampling, класс-weighted loss")

if role_counts["bass"] < role_counts["chords"] * 0.1:
    print("ДИСБАЛАНС: Мало BASS треков")
    print("    Решение: Data augmentation, weighted sampling, класс-weighted loss")

if total < 1000:
    print("  НЕДОСТАТОЧНО ДАННЫХ: Меньше 1000 файлов")
    print("    Решение: Добавить больше MIDI файлов в dataset")

if ok < total * 0.8:
    print("  МНОГО ОТКЛОНЕННЫХ ФАЙЛОВ")
    print("    Решение: Проверить критерии отклонения в preprocess_midi.py")

print("\n Анализ завершен!")
