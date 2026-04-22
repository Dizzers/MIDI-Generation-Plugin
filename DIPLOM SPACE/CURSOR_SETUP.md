# Cursor Setup

## Purpose
This file keeps Cursor aligned with the active project strategy.

## Current Project State
- diploma project
- one month deadline
- project strategy changed from role-based arrangement generation to full-MIDI single-stream generation

## Current Priorities
1. Build a stable full-MIDI dataset pipeline.
2. Train one coherent Transformer on full event streams.
3. Produce usable MIDI examples for demos and the thesis.
4. Keep documentation synchronized with the new strategy.

## Active Technical Decisions
- full MIDI tokenization from complete files
- no role split in the training loop
- genre-conditioned Transformer LM
- dataset primer for generation
- simple, defensible pipeline over ambitious but fragile architecture

## Files Cursor Should Read First
- [README.md](/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM%20SPACE/README.md)
- [dataset/tokenize_midi.py](/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM%20SPACE/dataset/tokenize_midi.py)
- [dataset/chunk_tokens.py](/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM%20SPACE/dataset/chunk_tokens.py)
- [model/dataset.py](/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM%20SPACE/model/dataset.py)
- [model/train_improved.py](/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM%20SPACE/model/train_improved.py)
- [model/generate.py](/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM%20SPACE/model/generate.py)

## Expected Outputs
- `dataset/processed/tokens/full.npy`
- `dataset/processed/chunks/full_chunks.npy`
- `checkpoints/model_best.pth`
- generated full MIDI samples in a chosen output directory

## If Something Looks Wrong
Check in this order:
1. whether `full.npy` exists
2. whether `full_chunks.npy` exists
3. whether `vocab.json` matches the new token format
4. whether training and generation use the same prefix tokens
