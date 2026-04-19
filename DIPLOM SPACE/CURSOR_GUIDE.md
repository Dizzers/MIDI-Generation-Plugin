# MIDI Generation Plugin - Quick Reference Guide

## What to Focus On When Reviewing Code

### Must-Know Files
1. **model/transformer.py** - Core architecture
   - TransformerLM class with role/genre conditioning
   - Position embeddings, token embeddings, causal masking
   - Output vocabulary size matches training data

2. **model/train_improved.py** - Training pipeline
   - DataLoader setup, loss computation, checkpoint saving
   - Validation metrics, early stopping, plot generation
   - Resume functionality for long training runs

3. **model/generate.py** - Generation interface
   - Supported parameters: role, genre, temperature, top_k, top_p
   - Token sampling strategies
   - Output: MIDI file with generated sequences

4. **dataset/tokenize_midi.py** - Token vocabulary
   - Token types: NOTE_ON, NOTE_OFF, TIME_SHIFT, CONTROL_CHANGE
   - Vocabulary mapping (string → ID)
   - Special tokens: PAD, EOS, BOS

## Code Review Checklist

When reviewing changes:
- [ ] Role classification logic (pitch > X = melody, etc.)
- [ ] Token vocabulary consistency across pipeline
- [ ] Batch size × chunk length doesn't exceed model.max_len
- [ ] Checkpoint paths use pathlib.Path
- [ ] Device handling (CPU vs CUDA) is consistent
- [ ] Stats.json validates output quality
- [ ] Test with both single-file and batch processing

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| OOM during training | Reduce batch_size, enable gradient_checkpointing |
| Poor generation quality | Increase n_layers, check vocab integrity |
| Inconsistent roles | Verify role detection thresholds in preprocess |
| Token ID mismatch | Regenerate vocab.json, clear old checkpoints |

## Tokenization Reference
```
Vocabulary example:
- NOTE_ON:72 → "note_on_72" (MIDI note 72 = C5)
- TIME_SHIFT:0.2 → "time_shift_1" (200ms shift)
- CONTROL_CHANGE → "cc_*"
```

## Role Detection Heuristic
```
- MELODY: pitch median > 60, low note density
- BASS: pitch median < 48, sparse notes
- CHORDS: multiple simultaneous notes, harmonic function
```

## Hyperparameter Defaults
- Learning rate: 1e-4 (Adam)
- Warmup steps: 1000
- Dropout: 0.2
- Weight decay: 0.01
- Gradient clip norm: 1.0

## File Input/Output Formats
```
Input: MIDI files from classical/ folder
Processing: 
  .mid → EventList → Tokens → Chunks (256-length with stride 128)
Output: 
  vocab.json (token vocabulary)
  chunks.npy (token ID arrays)
  stats.json (processing metrics)
```

## Dataset Statistics Location
After processing, check `dataset/processed/stats.json`:
```json
{
  "total_files": 12345,
  "total_chunks": 567890,
  "vocab_size": 2048,
  "avg_chunk_length": 250,
  "role_distribution": { "melody": 0.4, "bass": 0.3, "chords": 0.3 }
}
```

## Quick Commands
```bash
# One-time setup
cd dataset && python preprocess_midi.py && python tokenize_midi.py && python chunk_tokens.py

# Train from scratch
python model/train_improved.py --epochs 50 --batch_size 16

# Resume training
python model/train_improved.py --resume --checkpoint "checkpoints/best.pt"

# Generate 256-token melody
python model/generate.py --role melody --length 256 --temperature 0.8

# Full pipeline test
python model/evaluate_generation.py --checkpoint "checkpoints/best.pt"
```

## When Adding New Features
1. Update .cursorrules with new patterns
2. Add docstring with parameter descriptions
3. Log statistics to facilitate debugging
4. Test with both regular and full dataset modes
5. Verify backward compatibility with existing checkpoints

## Token Budget Tips for Cursor Pro
1. ✅ Use .cursorignore to exclude processed/*.npy (saves 80% of tokens)
2. ✅ Reference .cursorrules instead of repeating architecture details
3. ✅ Keep context window focused on 2-3 files at a time
4. ✅ Use "View File" feature instead of "Explain Full File"
5. ✅ Save context with cmd+shift+s before large refactors
