# Music Sequence Generation Plugin (MIDI, Transformer, JUCE-ready)

## Overview
This project is an end-to-end pipeline for **symbolic music generation** based on user-driven constraints, with a target integration into a **DAW plugin** (JUCE).

The model generates MIDI token sequences for three musical roles:
- `MELODY`
- `BASS`
- `CHORDS`

The current architecture is a Transformer language model trained on tokenized MIDI data.  
The generation pipeline supports role/genre conditioning and post-training quality evaluation.

## Project Goals
- Build a controllable generative model for musical sequences.
- Produce harmonically coherent role-based material (melody, bass, chords).
- Support iterative user interaction in a plugin workflow (multiple re-generate options).
- Provide objective evaluation metrics in addition to listening tests.

## Current Status
- MIDI preprocessing and role detection pipeline implemented.
- Tokenization and chunking pipeline implemented.
- Transformer model training/evaluation pipeline implemented.
- Generation pipeline supports:
  - role/genre conditioning,
  - top-k/top-p sampling,
  - repetition penalty,
  - no-repeat n-gram constraints.
- Additional generation-quality evaluation script implemented.

## Repository Structure
```text
dataset/
  preprocess_midi.py        # MIDI parsing and role detection
  tokenize_midi.py          # event-token conversion + vocab build
  chunk_tokens.py           # sequence chunking
  processed/
    vocab.json
    chunks/*.npy

model/
  transformer.py            # TransformerLM architecture
  dataset.py                # PyTorch dataset and sampling
  train_improved.py         # training/validation/test + plots + resume
  generate.py               # controllable MIDI generation
  evaluate_generation.py    # objective generation metrics

checkpoints/               # model checkpoints and training history
```

## Model
`TransformerLM` (`model/transformer.py`) includes:
- token + positional embeddings,
- causal self-attention,
- pre-layer normalization,
- padding mask support,
- weight tying (`token_emb` <-> output projection),
- cached causal masks for efficient decoding.

## Data Pipeline
1. `dataset/preprocess_midi.py`  
   Parses raw MIDI, detects roles, writes metadata.
2. `dataset/tokenize_midi.py`  
   Converts MIDI events to token sequences.
3. `dataset/chunk_tokens.py`  
   Splits long sequences into trainable chunks.

## Training
Main script: `model/train_improved.py`  
Configuration is intentionally defined in-code (single source of truth).

### Key training features
- train/val/test split,
- weighted role loss,
- ReduceLROnPlateau scheduler,
- gradient clipping,
- early stopping,
- checkpointing (`model_best.pth`, `model_final.pth`),
- resume from checkpoint,
- metric plots.

### Run
```bash
./venv/bin/python model/train_improved.py
```

## Evaluation
### 1) Validation/Test metrics during training
- Loss
- Accuracy
- Perplexity
- Role-wise accuracy (`MELODY`, `BASS`, `CHORDS`)
- Music proxy metrics:
  - `repeat_rate`
  - `unique_token_ratio`
  - `note_density`
  - `pitch_range`
  - `rhythm_diversity`
  - `scale_coverage`

### 2) Generation-specific metrics
Script: `model/evaluate_generation.py`
- `EOS rate`
- `4-gram repeat rate`
- `NOTE_ON/OFF balance`
- `CHORDS consistency`
- `cross-track consonance`

Run:
```bash
./venv/bin/python model/evaluate_generation.py --genre TRAP --samples 30
```

## Generation
Script: `model/generate.py`

Supports:
- role conditioning (`MELODY`, `BASS`, `CHORDS`, `ALL`),
- genre conditioning,
- top-k + top-p,
- repetition penalty,
- no-repeat n-gram.

Example:
```bash
./venv/bin/python model/generate.py \
  --role ALL \
  --genre TRAP \
  --samples 10 \
  --temperature 0.95 \
  --top-k 20 \
  --top-p 0.95 \
  --repetition-penalty 1.1 \
  --no-repeat-ngram-size 4 \
  --out-dir generated_eval_trap
```

## Running on Kaggle
See: `kaggle_bundle/README_KAGGLE.md`

## Versioning and Reproducibility
- Use Git for code/config evolution.
- Keep large binary artifacts out of Git (`.gitignore` configured).
- Keep checkpoints per experiment with clear naming (recommended extension).
- Keep training history (`checkpoints/history.json`) and plots for reports.

## Roadmap
- Improve CHORDS role quality and cross-role coherence.
- Add structured conditioning (e.g., chord-first generation pipeline).
- Add plugin integration layer (JUCE bridge, low-latency inference path).
- Add human listening test protocol + statistical reporting.

## License
Define project license before public release (MIT/Apache-2.0 recommended for code).
