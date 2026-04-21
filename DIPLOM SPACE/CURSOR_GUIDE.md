# MIDI Generation Plugin - Quick Reference Guide

## Recent Improvements (Solution 1 + 2)

### Solution 1: Detection Threshold Optimization
- **File**: dataset/preprocess_midi.py (lines 186-194)
- **Changes**: Lowered detection gates to catch more melody/bass tracks
  - Bass: low_pitch_ratio 0.36 → 0.20
  - Melody: high_pitch_ratio 0.22 → 0.12
- **Result**: Better role distribution (~1:0.4:6.4 ratio for melody:bass:chords)

### Solution 2: Weighted Loss Training
- **File**: model/train_improved.py
- **Key parameters**:
  - ROLE_WEIGHT_ALPHA: 1.2 (higher weight for rare roles)
  - MAX_ROLE_WEIGHT: 15.0 (stronger multipliers)
  - LEARNING_RATE: 5e-5 (fine-tuning rate)
  - NUM_EPOCHS: 40 (resume from checkpoint)
- **How it works**: Rare roles (melody/bass) get higher loss weights during backward pass
  - Melody weight ≈ 1.8x (max_count / count)^1.2
  - Bass weight ≈ 2.5x
  - Chords weight ≈ 1.0x (normalized)

## Training Monitoring (Weighted Loss Phase - April 20, 2026)

When running `python model/train_improved.py`:

### Current Dataset Status
- **Total MIDI**: 15,529 files (Trap + Classical)
- **Sequences**: 1200 chords, 298 melody, 140 bass
- **Chunks**: 5000 chords (capped), 1981 melody, 588 bass
- **Ratio**: 8.5:3.4:1 (excellent balance!)
- **Fixed**: Conditioning tokens now present in all chunks
- **Weighted Loss**: ALPHA=0.9, MAX_WEIGHT=15.0

### Training Progress (Epochs 1-3)
```
Epoch 1: Val Loss=4.8444, Acc=0.1600, Role Acc M=0.174 B=0.158 C=0.146
Epoch 2: Val Loss=4.5216, Acc=0.1891, Role Acc M=0.219 B=0.186 C=0.159
Epoch 3: In progress... (train loss=1.1031 - improving!)
```

### Key Metrics to Watch
1. **train_loss** (should decrease each epoch)
   - Epoch 1: ~2.69 → Epoch 2: ~2.13 → Epoch 3: ~1.10 = ✅ Excellent!

2. **val_loss** (main metric - watch for overfitting)
   - Epoch 1: 4.8444 → Epoch 2: 4.5216 = ✅ Improving
   - Target: < 4.0 by epoch 10

3. **role_metrics** (new! all roles now learning)
   ```
   Epoch 1: M=0.174, B=0.158, C=0.146
   Epoch 2: M=0.219, B=0.186, C=0.159  ← All improving!
   ```
   - Chords accuracy was 0.000 before fix!

4. **music_metrics** (generation quality indicators)
   - repeat_rate: 0.729 → 0.721 (still high, but improving)
   - diversity: 0.017 (low, needs more epochs)
   - scale_cov: 0.000 (issue with metric or model, investigate later)

4. **music_metrics** (generation quality indicators)
   - repeat_rate < 0.25: Good (low repetition)
   - unique_token_ratio > 0.6: Good (diverse tokens)
   - pitch_range > 0.4: Good (wide note range)
   - scale_coverage > 0.6: Good (harmonic coherence)

### Expected Values by Epoch
```
Epoch 1:   train_loss=4.5, val_loss=4.2, repeat_rate=0.40
Epoch 10:  train_loss=3.8, val_loss=3.5, repeat_rate=0.32
Epoch 20:  train_loss=3.0, val_loss=3.2, repeat_rate=0.28
Epoch 30:  train_loss=2.7, val_loss=3.2, repeat_rate=0.26 ← Plateau OK
```

### When to Stop Training
- Val loss hasn't improved for 5+ epochs → early stopping will trigger
- Epoch 40 reached (max configured)
- repeat_rate plateaus < 0.25 = quality acceptable

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
