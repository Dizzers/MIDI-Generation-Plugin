# MIDI Generation Plugin (SkyTNT-powered)

Diplom-level project: a JUCE MIDI-effect plugin that generates MIDI fully
**offline** using the [SkyTNT/midi-model](https://github.com/SkyTNT/midi-model)
architecture. The training, ONNX export and inference all share the same
SkyTNT tokenizer/model contract, so the C++ plugin behaves byte-for-byte
like the upstream Python `app_onnx.py`.

```
seed/empty -> SkyTNT tokenizer -> model_base.onnx -> hidden state ->
              model_token.onnx (per-token loop, KV cache) ->
              SkyTNT detokenizer -> juce::MidiMessageSequence -> DAW
```

## Repository layout

```
DIPLOM SPACE/
├── skytnt_adapter/                 ← Python adapter on top of SkyTNT
│   ├── midi_model_upstream/        ← Vendored SkyTNT/midi-model snapshot
│   ├── finetune_skytnt.py          ← Fine-tune wrapper (Lightning)
│   ├── export_skytnt_onnx.py       ← Export ckpt -> 2 ONNX files + tokenizer JSON
│   └── sample_generate.py          ← Smoke test (PyTorch and ONNX modes)
│
├── plugin/juce/
│   ├── CMakeLists.txt              ← USE_ONNXRUNTIME=ON + FetchContent ORT
│   ├── Source/
│   │   ├── ModelInference.{h,cpp}  ← Plugin-side facade
│   │   ├── SkyTNTTokenizer.{h,cpp} ← C++ port of SkyTNT MIDITokenizerV2
│   │   ├── SkyTNTRuntime.{h,cpp}   ← ONNX-Runtime two-stage generator + KV cache
│   │   ├── SmokeTest.cpp           ← Standalone CLI (skytnt_smoke_test)
│   │   ├── GeneratorThread.{h,cpp} ← Async wrapper around ModelInference
│   │   └── …all original JUCE plugin sources…
│   └── JUCE/                        ← JUCE submodule (unchanged)
│
├── artifacts/
│   ├── onnx/model_base.onnx         ← Produced by export step
│   ├── onnx/model_token.onnx
│   ├── tokenizer/tokenizer_config.json
│   ├── tokenizer/config.json        ← Full MIDIModelConfig dump
│   └── examples/sample_*.mid        ← Smoke-test outputs
│
├── notebooks/
│   ├── kaggle_skytnt_finetune.ipynb ← Kaggle T4 end-to-end recipe
│   └── kaggle_train.ipynb           ← (legacy, custom transformer)
│
├── dataset/                         ← Old custom-vocab pipeline (kept for reference)
├── model/                           ← Old TransformerLM (kept; not the active path)
├── checkpoints/                     ← Training output (gitignored except dirs)
├── requirements.txt
└── README.md  (this file)
```

The legacy `dataset/` + `model/` + custom `vocab.json` pipeline is **no longer
the active path**. It's left intact so you can reproduce the old experiments,
but the plugin now consumes only `artifacts/onnx/*` + `tokenizer_config.json`.

## Quick install (local)

```bash
cd "DIPLOM SPACE"
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## End-to-end pipeline

### 1) Prepare your MIDI corpus

Drop raw MIDI files into `dataset/midi_raw/` (recursive subfolders are fine).

```bash
ls dataset/midi_raw/ | head
```

No tokenization or vocab building is needed any more — SkyTNT does it on the
fly during training (using the V2 tokenizer baked into the upstream code).

### 2) Fine-tune SkyTNT on your corpus

GPU with bf16 strongly recommended (T4, P100, A100). For laptop CPU you can
use `--accelerator cpu --precision 32-true --max-step 100` just to verify
that the pipeline works.

```bash
python -m skytnt_adapter.finetune_skytnt \
    --data dataset/midi_raw \
    --pretrained skytnt/midi-model-tv2o-medium \
    --output checkpoints/skytnt \
    --config tv2o-medium \
    --max-len 2048 --batch-size 1 --acc-grad 4 \
    --max-step 4000 --warmup-step 200 --val-step 400 \
    --lr 1e-4 --precision bf16-mixed
```

Outputs into `checkpoints/skytnt/`:
- `lightning_ckpt/best-…ckpt`  — best-by-val-loss checkpoint
- `model.ckpt`                  — final state-dict (used by export)
- `model_best.ckpt`             — copy of the best epoch
- `tokenizer_config.json`       — convenience copy

### 3) Export to ONNX

```bash
python -m skytnt_adapter.export_skytnt_onnx \
    --ckpt checkpoints/skytnt \
    --config tv2o-medium \
    --out-dir artifacts/onnx \
    --tokenizer-out artifacts/tokenizer/tokenizer_config.json
```

Produces `artifacts/onnx/model_base.onnx`,
`artifacts/onnx/model_token.onnx`, and the tokenizer JSON. The exporter
internally calls SkyTNT's own `export.py` helpers so the I/O graph (past_kv
list, dynamic axes, opset 14, onnxsim simplification) matches the upstream
Python runtime exactly.

### 4) Smoke-test from Python

```bash
python -m skytnt_adapter.sample_generate \
    --mode onnx \
    --base artifacts/onnx/model_base.onnx \
    --token artifacts/onnx/model_token.onnx \
    --tokenizer artifacts/tokenizer/tokenizer_config.json \
    --out artifacts/examples \
    --num 2 --max-len 256
```

Two `.mid` files appear in `artifacts/examples/`. Drag them into your DAW to
verify they're musically meaningful.

### 5) Build the JUCE plugin

The plugin auto-fetches ONNX Runtime 1.17.3 via CMake `FetchContent` (you can
override with `-DONNXRUNTIME_DIR=/path/to/onnxruntime` or
`-DONNXRUNTIME_VERSION=1.17.3`). Pre-extracted runtimes work too.

```bash
cd plugin/juce
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

Built artifacts:
- `build/MIDIGenerationPlugin_artefacts/Release/AU/MIDI Generation.component`
- `build/MIDIGenerationPlugin_artefacts/Release/VST3/MIDI Generation.vst3`
- `build/skytnt_smoke_test`  — CLI you can run before installing into the DAW

### 6) Plugin smoke-test (CLI)

```bash
./plugin/juce/build/skytnt_smoke_test \
    --base artifacts/onnx/model_base.onnx \
    --token artifacts/onnx/model_token.onnx \
    --tokenizer artifacts/tokenizer/tokenizer_config.json \
    --out artifacts/examples/cli_test.mid \
    --max-len 128 --bpm 120
```

Open `artifacts/examples/cli_test.mid` in any MIDI editor to confirm the
C++ runtime produces the same kind of MIDI as the Python smoke test.

### 7) Install plugin

macOS:

```bash
cp -r "plugin/juce/build/MIDIGenerationPlugin_artefacts/Release/AU/MIDI Generation.component" \
    ~/Library/Audio/Plug-Ins/Components/
```

The plugin looks for ONNX models in this order:

1. `$MIDIGEN_ARTIFACT_DIR/onnx/model_base.onnx` (set in Logic / your DAW)
2. `<plugin-bundle>/.../artifacts/onnx/model_base.onnx`
3. Walks up to 8 directories from the executable looking for `artifacts/`,
   `plugin/juce/bin/`, or `bin/`

So either drop your `artifacts/` next to the plugin or set the env var.

## Kaggle T4 recipe

Open `notebooks/kaggle_skytnt_finetune.ipynb` on Kaggle.

1. **Add Data → Notebook input → Kaggle Datasets**: attach a dataset
   containing your MIDIs (e.g. `your-username/my-midi-corpus`).
2. **Settings → Internet ON, Accelerator: GPU T4 ×1**.
3. (Optional) attach this repo as a Kaggle dataset and set `REPO_INPUT` in
   cell 1, *or* `git clone` the upstream and copy `skytnt_adapter/` over.
4. **Run all cells**.
5. After the run download `/kaggle/working/artifacts.zip` from the right-hand
   "Output" panel, unzip into `artifacts/` locally — done.

The notebook's training command:

```bash
python -m skytnt_adapter.finetune_skytnt \
    --data "$MIDI_INPUT" \
    --pretrained "$pretrained_dir" \
    --output checkpoints/skytnt \
    --config tv2o-medium \
    --max-len 2048 \
    --batch-size 1 --batch-size-val 1 \
    --workers 2 --workers-val 2 --acc-grad 4 \
    --max-step 4000 --warmup-step 200 --val-step 400 \
    --lr 1e-4 --precision bf16-mixed \
    --accelerator gpu --devices 1
```

Export step:

```bash
python -m skytnt_adapter.export_skytnt_onnx \
    --ckpt checkpoints/skytnt --config tv2o-medium \
    --out-dir /kaggle/working/artifacts/onnx \
    --tokenizer-out /kaggle/working/artifacts/tokenizer/tokenizer_config.json
```

## What's MVP and what is full-fidelity

| Concern                              | Status                                             |
|--------------------------------------|----------------------------------------------------|
| Two-stage `model_base` + `model_token` generation with KV cache (C++) | ✅ full port of `app_onnx.generate` |
| Top-p / top-k sampling, temperature  | ✅ matches upstream numpy implementation           |
| Event-mask (`event_ids`/`parameter_ids`) at every position | ✅ exact, V1 + V2                |
| `disable_patch_change`, `disable_control_change`, `disable_channels` | ✅ honoured in mask     |
| Tokenizer config: vocab/event tables, parameter sizes, max_token_seq | ✅ rebuilt from JSON dump |
| `tokens2event` + `detokenize` (token grid → MIDI events with seconds) | ✅ V1 and V2, mirrors `MIDITokenizer.detokenize` |
| `tokenize` (user MIDI → tokens) in C++                               | 🟡 **MVP**: empty-prompt + setup events (BPM, key sig, patch) only. To extend into a full seed-MIDI encoder, port the `tokenize()` function from `skytnt_adapter/midi_model_upstream/midi_tokenizer.py` (~300 lines) into `SkyTNTTokenizer.cpp`. |
| Render audio (FluidSynth)                                            | ❌ out of scope (DAW handles audio) |

The MVP intentionally keeps the C++ encoder narrow because the SkyTNT
tokenizer's `tokenize()` is highly stateful (track/channel remapping, empty
channel pruning, key-signature inference, etc.). All the **decoding** logic
is fully implemented, so generation always produces valid MIDI; only seeding
from a user-provided MIDI clip in the DAW is "lite" today.

## Troubleshooting

- **"ONNX models not found"** in the plugin status: the plugin couldn't find
  `artifacts/onnx/*.onnx`. Set `MIDIGEN_ARTIFACT_DIR` to the parent of
  `artifacts/`, or copy the folder next to the `.au`/`.vst3`.
- **`vocab_size mismatch`**: your `tokenizer_config.json` was produced by a
  different SkyTNT version than the C++ port. Recreate it by re-running
  `export_skytnt_onnx.py`.
- **Build error: `onnxruntime_cxx_api.h not found`**: pass
  `-DONNXRUNTIME_DIR=…` to CMake or check that FetchContent finished
  downloading (look for `_deps/onnxruntime-src/include/`).
- **Out-of-memory during finetune on T4**: drop `--batch-size 1`, increase
  `--acc-grad`, or use `--max-len 1024`.

## License

Original SkyTNT/midi-model code is MIT (see `skytnt_adapter/midi_model_upstream/LICENSE`).
The rest of this repository follows the project's existing license.
