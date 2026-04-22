# MIDI Generation Plugin

## Overview
This diploma project is now built around a `single-stream full MIDI` strategy.

Instead of splitting music into `MELODY / BASS / CHORDS`, the model learns one event stream from the whole MIDI file:
- `<GENRE_...>`
- `<KEY_...>`
- `TIME_SHIFT_*`
- `VELOCITY_*`
- `NOTE_ON_*`
- `NOTE_OFF_*`

The goal is to preserve real musical context and avoid rebuilding an arrangement from weak separately generated roles.

## Current Architecture
- full-MIDI tokenization from complete scores
- chunking into trainable event sequences
- Transformer LM with `genre` conditioning and optional `key` token in the prefix
- single-stream generation of complete MIDI material

Core files:
- [dataset/tokenize_midi.py](/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM%20SPACE/dataset/tokenize_midi.py)
- [dataset/chunk_tokens.py](/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM%20SPACE/dataset/chunk_tokens.py)
- [model/dataset.py](/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM%20SPACE/model/dataset.py)
- [model/train_improved.py](/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM%20SPACE/model/train_improved.py)
- [model/generate.py](/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM%20SPACE/model/generate.py)
- [model/evaluate_generation.py](/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM%20SPACE/model/evaluate_generation.py)

## Why The Strategy Changed
The earlier role-based pipeline had two structural problems:
- role detection threw away too much useful musical material,
- separate role generation made final arrangements sound disconnected.

The current strategy is more realistic for the available data and the one-month deadline:
- use more of the raw dataset,
- train on complete musical texture,
- simplify generation and evaluation.

## Data Pipeline
### 1. Tokenize full MIDI files
```bash
python3 "DIPLOM SPACE/dataset/tokenize_midi.py"
```

Output:
- `dataset/processed/tokens/full.npy`
- `dataset/processed/vocab.json`

### 2. Build training chunks
```bash
python3 "DIPLOM SPACE/dataset/chunk_tokens.py"
```

Output:
- `dataset/processed/chunks/full_chunks.npy`

## Training
```bash
python3 "DIPLOM SPACE/model/train_improved.py"
```

Current training design:
- full train split each epoch
- warmup + cosine LR
- moderate augmentation
- token-level loss and accuracy
- music proxy metrics:
  - `repeat_rate`
  - `unique_token_ratio`
  - `rhythm_diversity`
  - `scale_coverage`

## Generation
```bash
python3 "DIPLOM SPACE/model/generate.py" \
  --genre TRAP \
  --samples 4 \
  --out-dir "DIPLOM SPACE/generated_full"
```

Generation uses:
- dataset primer when available
- top-k / top-p sampling
- repetition penalty
- multi-candidate reranking

## Evaluation
```bash
python3 "DIPLOM SPACE/model/evaluate_generation.py" --genre TRAP --samples 20
```

Tracked generation metrics:
- EOS rate
- 4-gram repeat rate
- NOTE_ON/OFF balance
- polyphony ratio

## Practical Goal
The current target is not a perfect music composer.
It is a stable and defensible diploma result:
- one coherent full-MIDI model,
- reproducible training,
- generation that sounds more musically connected than the old role-based version,
- clean docs so future work and Cursor remain aligned.
