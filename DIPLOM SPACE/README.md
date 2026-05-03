# MIDI Generation Plugin - Training Pipeline

From-scratch causal Transformer LM that produces a TorchScript module +
`vocab.json` consumed by the JUCE plugin in `plugin/juce/` (no plugin C++
changes required).

## Project layout

```
DIPLOM SPACE/
├── dataset/
│   ├── vocab_contract.py     # single source of truth for token format
│   ├── validate_vocab.py     # checks vocab.json against C++ regex contract
│   ├── preprocess_midi.py    # raw MIDI -> processed/midi_meta.jsonl
│   ├── tokenize_midi.py      # MIDI -> Performance tokens with prefix
│   ├── build_vocab.py        # canonical vocab.json (size = 326 tokens)
│   ├── split_tokens.py       # file-level 80/10/10 split
│   ├── chunk_tokens.py       # max_len=1024, stride=512 chunks
│   ├── midi_raw/             # PUT YOUR 10k MIDI FILES HERE
│   └── processed/            # all generated artifacts
├── model/
│   ├── transformer.py        # TorchScript-friendly TransformerLM (~25M)
│   ├── dataset.py            # PyTorch Dataset + augmentation
│   ├── train.py              # AMP fp16, AdamW, cosine LR, early-stop
│   ├── music_metrics.py      # scale-coverage, repeat-rate, etc.
│   ├── export_torchscript.py # .pth -> .ts.pt + roundtrip check
│   └── generate.py           # Python sanity-check generator (.mid out)
├── notebooks/
│   └── kaggle_train.ipynb    # end-to-end run on Kaggle T4
├── checkpoints/              # (gitignored) model_best.pth, history, plots
├── generated/                # (gitignored) .mid samples
├── plugin/juce/              # JUCE plugin (don't touch)
└── requirements.txt
```

## Token contract (locked)

Documented in [`dataset/vocab_contract.py`](dataset/vocab_contract.py); enforced by
[`dataset/validate_vocab.py`](dataset/validate_vocab.py). Mirrors the regex /
parser used by [`plugin/juce/Source/ModelInference.cpp`](plugin/juce/Source/ModelInference.cpp)
and [`plugin/juce/Source/MidiGenerator.cpp`](plugin/juce/Source/MidiGenerator.cpp):

| Class       | Format                  | Range                   | Example              |
|-------------|-------------------------|-------------------------|----------------------|
| Special     | literal                 | `<PAD> <BOS> <EOS> <UNK>` | `<BOS>`            |
| Genre       | `<GENRE_NAME>`          | >= 1                    | `<GENRE_TRAP>`       |
| Key         | `<KEY_PC_{MAJ\|MIN}>`   | 24 + `<KEY_UNKNOWN>`    | `<KEY_C#_MAJ>`       |
| Note On     | `NOTE_ON_0xPP`          | 128 (0..127)            | `NOTE_ON_0x3c`       |
| Note Off    | `NOTE_OFF_0xPP`         | 128 (0..127)            | `NOTE_OFF_0x3c`      |
| Time Shift  | `TIME_SHIFT_0xSSSS`     | 1..32 steps x 0.05 s    | `TIME_SHIFT_0x0010`  |
| Velocity    | `VELOCITY_0xVV`         | 0..7 (8 bins)           | `VELOCITY_0x05`      |

Total vocab = 4 + 1 + 25 + 128 + 128 + 32 + 8 = **326 tokens**.

The TorchScript module signature matches the C++ call site exactly:
```cpp
auto out = module.forward({x, g}).toTensor();   // x: [B,T] long, g: [B] long
```

## Quick start (local sanity check)

```bash
cd "DIPLOM SPACE"
pip install -r requirements.txt

# put a few MIDIs into dataset/midi_raw/ first

python -m dataset.preprocess_midi
python -m dataset.tokenize_midi
python -m dataset.build_vocab
python -m dataset.split_tokens
python -m dataset.chunk_tokens

# 2-epoch dry run on CPU/MPS just to verify the pipeline
python -m model.train --num_epochs 2 --batch_size 4

python -m model.export_torchscript --copy-to-bin
python -m model.generate --key C_MAJOR --target_seconds 4.0 --bpm 120 \
    --out generated/sample.mid
```

## Training on Kaggle (T4 / P100, ~2-4 h)

Open [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) on Kaggle,
turn on the GPU + Internet, attach your raw-MIDI dataset, edit
`RAW_DATASET_DIR`, and Run All. The final cells write
`/kaggle/working/MIDI-Generation-Plugin/DIPLOM SPACE/plugin/juce/bin/`:
- `model_best.ts.pt` (TorchScript module, ~100 MB)
- `vocab.json`        (~22 KB)

Download both and drop them into your local `plugin/juce/bin/`. The JUCE
plugin loads them automatically the next time you launch it (look for
"Model ready" in the plugin status text).

## Where the 14 UI parameters live

- **Train-time conditioning (visible to the model)**: KEY (via `<KEY_*>` prefix
  token). Genre is technically conditioning too, but the C++ plugin hardcodes
  `<GENRE_TRAP>` so it acts as a fixed style id.
- **Inference-time logit ops (already in
  [`ModelInference.cpp`](plugin/juce/Source/ModelInference.cpp))**: temperature,
  top-K, top-P, repetition-penalty, no-repeat-ngram, harmony-bias, harmony-mode,
  groove-feel, velocity-feel, max-polyphony, max-len, target-seconds,
  min-body-tokens, primer-mode/len.
- **Post-processing (already in
  [`MidiPostProcessor.h`](plugin/juce/Source/MidiPostProcessor.h))**: BPM,
  quantize-grid, quantize-amount, swing-amount, humanize-time-ms,
  humanize-velocity, velocity-min/max.

Note: BPM is not a real train-time parameter for this tokenization (TIME_SHIFT
is in absolute seconds, not in beats). It is correctly handled as a post-export
time-scaling factor, which is what the plugin already does.

## Reproducing for the diploma

After training:
- `checkpoints/history.json` - per-epoch train/val loss, accuracy, perplexity, music metrics
- `checkpoints/plots/loss_curve.png` - training curves
- `checkpoints/test_metrics.json` - final metrics on the held-out test split

For an A/B comparison vs the C++ plugin, run `model/generate.py` with the same
seed + parameters, then compare the two `.mid` files (event order should match
up to the sampling step).
