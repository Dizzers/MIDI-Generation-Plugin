# JUCE AU MIDI Generation Plugin - Architecture & Design

## System Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                 FL STUDIO (DAW)                         │
│  Hosts AU Plugin, receives MIDI on track               │
└─────────────────────────────────────────────────────────┘
                            ↑
                       MIDI Buffer
                            ↓
┌─────────────────────────────────────────────────────────┐
│         PluginProcessor (AudioProcessor)                │
│  ├─ 14 Parameters (stored in ValueTreeState)           │
│  ├─ processBlock() → outputs MIDI to DAW               │
│  ├─ startGeneration() → fires GeneratorThread          │
│  └─ updateParameterDisplay() → notify UI               │
└─────────────────────────────────────────────────────────┘
           ↑                                    ↑
      Parameter Updates              Generation Results
      (User clicks Slider)           (Tokens → MIDI)
           ↓                                    ↓
┌──────────────────────┐         ┌────────────────────────┐
│  PluginEditor (UI)   │         │ GeneratorThread        │
│                      │         │                        │
│ ┌──────────────────┐ │   ┌────→├─ ModelInference       │
│ │ Main Parameter   │ │   │     │  (PyTorch wrapper)    │
│ │ Panel            │ │   │     │                        │
│ │ - Crankcase      │ │   │     │ ┌─────────────────┐   │
│ │   Knobs (temp,   │ │   │     │ │ model_best.pt   │   │
│ │   leap, penalty) │ │   │     │ │ (inference)     │   │
│ │ - ComboBox       │ │   │     │ └─────────────────┘   │
│ │   (role, key)    │ │   │     │                        │
│ │ - Sliders        │ │   │     │ ┌─────────────────┐   │
│ │ (top-k, top-p)   │ │   │     │ │ vocab.json      │   │
│ │                  │ │   │     │ │ (token→id)      │   │
│ │ ┌──────────────┐ │ │   │     │ └─────────────────┘   │
│ │ │ GENERATE btn │─┼─┼───┘     │                        │
│ │ └──────────────┘ │ │         │ ↓ generateTokens()   │
│ │                  │ │         │ [vector<int>]         │
│ │ ┌──────────────┐ │ │         │                        │
│ │ │Output Window │─┼─┼────────→├─ MidiGenerator       │
│ │ │  button      │ │ │         │                        │
│ │ └──────────────┘ │ │         │ (tokens → MIDI)       │
│ └──────────────────┘ │         │                        │
│                      │         │ ↓ convertToMidi()     │
│ Status: Generating.. │         │ [MidiBuffer]          │
└──────────────────────┘         └────────────────────────┘
           ↑                              ↓
      Updates UI                    Writes to Queue
           ↑                              ↓
           └──────────────────────────────┘
                (Async communication)

                    ┌───────────────────────┐
                    │ OutputWindow          │
                    │ (Separate Window)     │
                    │                       │
                    │ ┌─────────────────┐   │
                    │ │ MidiVisualizer  │   │  
                    │ │ (Piano Roll)    │   │
                    │ │ ┌─────────────┐ │   │
                    │ │ │ MIDI Events │ │   │
                    │ │ │ Drawn as    │ │   │
                    │ │ │ Rectangles  │ │   │
                    │ │ └─────────────┘ │   │
                    │ └─────────────────┘   │
                    │                       │
                    │ Info: Dur, Notes,     │
                    │ PitchRange            │
                    │                       │
                    │ [Play] [Export]       │
                    │ [Drag-Drop] [Regen]   │
                    └───────────────────────┘
```

## Class Hierarchy

```
PluginProcessor (extends AudioProcessor)
  - holds: 14 AudioParameterFloat/Choice objects
  - holds: GeneratorThread (async worker)
  - holds: OutputWindow* (float window)
  - methods:
    * processBlock(buffer, midiMessages) → outputs to DAW
    * setParameter(id, value) → updates state
    * startGeneration() → fires thread

PluginEditor (extends AudioProcessorEditor)
  - holds: Sliders, RotaryKnobs, ComboBoxes
  - holds: "Output Window" button
  - callbacks: slider changes → processor.setParameter()

OutputWindow (extends DocumentWindow)
  - holds: MidiVisualizer component
  - holds: Play, Export, Drag-Drop buttons
  - updates when MIDI generated

MidiVisualizer (extends Component)
  - holds: vector<MidiMessage> to draw
  - draws piano roll with JUCE Graphics
  - time axis = horizontal, pitch = vertical

GeneratorThread (extends Thread)
  - holds: ModelInference instance
  - loop: wait request → generateTokens() → convert to MIDI → post UI

ModelInference
  - singleton pattern
  - holds: torch::jit::Module (loaded model)
  - holds: vocab mapping
  - method: generateTokens(role, key, temp, ...)
  - calls: generate.py logic in C++

MidiGenerator
  - static utilities
  - parseTokens(vector<int>) → vector<MidiMessage>
  - uses: vocab.json for token names
  - handles: NOTE_ON, NOTE_OFF, TIME_SHIFT

RotaryKnob (extends Slider or custom Component)
  - renders as rotating dial
  - mouse drag = rotation
  - displays value as number + text
```

## Key Data Flow: "Generate Button Click"

1. **User clicks GENERATE**
   - Extract params from UI (role, key, temperature, etc)
   - PluginProcessor::startGeneration(params)
   
2. **GeneratorThread wakes up**
   - Calls ModelInference::generateTokens(role, key, temp, ...)
   
3. **ModelInference (PyTorch)**
   - Load context: [<BOS>, <ROLE_MELODY>, <GENRE_TRAP>, <KEY_A_MINOR>]
   - Forward pass through transformer
   - Apply sampling:
     * temperature scaling
     * top_k filter
     * top_p (nucleus)
     * repetition penalty
     * no-repeat n-gram
   - Return: `vector<int>` token_ids
   
4. **MidiGenerator converts**
   - Lookup vocab.json: token_id → "NOTE_ON_48", "TIME_SHIFT_0x3c"
   - Build MidiBuffer with NOTE_ON, NOTE_OFF, time offsets
   - Return vector<MidiMessage>
   
5. **Output to UI + DAW**
   - Post result to UI thread (atomic queue)
   - PluginEditor updates OutputWindow piano roll
   - PluginProcessor queues MIDI in processBlock()
   - FL Studio receives MIDI on track
   
6. **User can**
   - PLAY: Trigger playback
   - EXPORT: Save as .mid file
   - DRAG-DROP: Drag MIDI to FL track
   - REGENERATE: Use same params, new seed

## Parameter Persistence

- All 14 params in **AudioProcessorValueTreeState**
- `getStateInformation()` → XML blob
- `setStateInformation()` ← restores on reload
- FL Studio .flp project saves plugin state
- User settings preserved across sessions

## Threading Model

```
Audio Thread (Real-time, ~44.1kHz)
├─ processBlock() every ~11ms
├─ reads MIDI queue from GeneratorThread
├─ outputs to DAW
└─ MUST NOT block!

UI Thread (Event-driven)
├─ slider movements
├─ button clicks
├─ window redraws
└─ updates piano roll

Inference Thread (Background)
├─ awakened by Generate click
├─ calls torch forward pass
├─ MIDI conversion
├─ posts results to UI
└─ sleeps until next request
```

## Performance Targets

- Model Load Time: <2 seconds (once on startup)
- Generation Time: 2-5 seconds per 256 tokens
- UI Latency: <100ms responsiveness
- Audio Thread: No blocking (async queue)
- CPU during Generation: <30%
- Memory: ~500MB (LibTorch) + model

## File I/O Paths

- Checkpoint: `plugin/juce/bin/model_best.pt`
- Vocab: `plugin/juce/bin/vocab.json`
- Export MIDI: `~/Downloads/midi_gen_<timestamp>.mid`
- Temp MIDI: `/tmp/juce_midi_<random>.mid` (drag-drop)
- Plugin State: Embedded in .flp (DAW handles)

## UI Color Scheme

```
Background:     #1a1a1a (very dark gray)
Text:           #ffffff (white)
Label:          #aaaaaa (medium gray)

Accent Primary:   #00ff80 (neon green) - GENERATE button
Accent Secondary: #00d4ff (neon cyan)  - Export, Play

Piano Roll:
  - MELODY notes: #00ff80 (green)
  - BASS notes:   #0080ff (blue)
  - CHORDS notes: #9d4edd (purple)
  - Grid:         #333333
  - Playhead:     #ffff00 (yellow)

Hover: brighten +30%
Disabled: desaturate 50%
```

## Testing Checklist

- [ ] PluginProcessor loads in FL Studio
- [ ] All 14 params change without crashes
- [ ] GeneratorThread runs async (UI responsive)
- [ ] PyTorch model loads (<2s)
- [ ] GenerateTokens returns valid IDs
- [ ] MidiGenerator converts to NOTE_ON/OFF
- [ ] OutputWindow piano roll visualizes
- [ ] Play button triggers notes
- [ ] Export MIDI readable in FL Studio
- [ ] Drag-drop works
- [ ] Parameters saved/restored
- [ ] No memory leaks (1+ hour)
- [ ] CPU acceptable (<30%)

---

**Generated from**: plan.md  
**Status**: Ready for Phase 1 implementation
