# Cursor Guide For This Project

## What Cursor Should Know First
This project has pivoted from role-based generation to `single-stream full MIDI`.

Do not assume the current training pipeline uses:
- `MELODY`
- `BASS`
- `CHORDS`

Those ideas are now legacy context, not the main architecture.

## Primary Files
- [dataset/tokenize_midi.py](/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM%20SPACE/dataset/tokenize_midi.py)
- [dataset/chunk_tokens.py](/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM%20SPACE/dataset/chunk_tokens.py)
- [model/dataset.py](/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM%20SPACE/model/dataset.py)
- [model/train_improved.py](/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM%20SPACE/model/train_improved.py)
- [model/generate.py](/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM%20SPACE/model/generate.py)
- [model/evaluate_generation.py](/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM%20SPACE/model/evaluate_generation.py)

## Current Truth
- training data comes from whole MIDI files
- tokens begin with:
  - `<GENRE_...>`
  - `<KEY_...>`
- chunks are stored in `dataset/processed/chunks/full_chunks.npy`
- raw token sequences are stored in `dataset/processed/tokens/full.npy`
- the current model uses `genre` conditioning and no role embeddings

## Training Notes
- `train_improved.py` is the source of truth
- model dimensions remain `256 / 8 / 6 / 1024`
- loss is plain token-level CE with light label smoothing
- evaluation focuses on:
  - loss
  - accuracy
  - perplexity
  - repeat / diversity / rhythm / scale coverage

## Generation Notes
- generation is single-stream full-MIDI generation
- one output sample equals one MIDI stream
- generation uses dataset primers and candidate reranking
- current defaults are tuned to reduce repetition and improve coherence

## Before Changing Anything
1. Keep token format aligned with `vocab.json`.
2. Keep training and generation compatible with the same prefix structure.
3. Update docs when architecture changes.
4. Treat role-based code as historical context, not current ground truth.
