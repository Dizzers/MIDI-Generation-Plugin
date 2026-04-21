# MIDI Generation Plugin - Phase 1 Setup

## Overview
This is the **Phase 1 implementation** of the JUCE AU MIDI Generation Plugin.
- ✅ Basic JUCE structure ready
- ✅ 14 parameters defined
- ✅ UI layout with rotary knobs, sliders, comboboxes
- ✅ Async GeneratorThread skeleton
- ✅ OutputWindow frame
- 🔲 PyTorch integration (Phase 2)
- 🔲 Actual MIDI generation (Phase 2)

## Prerequisites

### macOS Requirements
- Xcode 12.0 or later
- CMake 3.20+
- LibTorch binary (pre-built)

### Get JUCE
JUCE must be available in the parent directory or specified in CMakeLists.txt:
```bash
cd plugin/juce
git clone https://github.com/juce-framework/JUCE.git
```

### Get LibTorch (macOS)
Download from [PyTorch.org](https://pytorch.org/get-started/locally/):

1. Visit https://pytorch.org/get-started/locally/
2. Select:
   - PyTorch: Stable
   - OS: macOS
   - Package: LibTorch
   - Language: C++/Java
   - Compute Platform: CPU or MPS (Mac Neural Engine)

3. Download and extract:
```bash
cd ~/Downloads
unzip libtorch-macos-*.zip
# Extract creates ~/Downloads/libtorch/
```

## Build Instructions

### Step 1: Configure CMake
```bash
cd plugin/juce
mkdir build && cd build

# Replace /path/to/libtorch with actual path
cmake .. -DTORCH_INSTALL_PREFIX=/Users/YOUR_USERNAME/Downloads/libtorch
```

### Step 2: Compile
```bash
cmake --build . --config Release -j4
```

Expected output:
```
[100%] Building CXX object CMakeFiles/MIDIGenerationPlugin.dir/Source/PluginEditor.cpp.o
[100%] Linking CXX shared module MIDIGenerationPlugin_artefacts/Release/AU/MIDI Generation.component
Plugin compiled successfully!
```

### Step 3: Install AU Plugin
```bash
cp -r "MIDIGenerationPlugin_artefacts/Release/AU/MIDI Generation.component" \
      "~/Library/Audio/Plug-Ins/Components/"
```

## Test in FL Studio

1. **Restart FL Studio** (it re-scans AU plugins on launch)
2. Open FL Studio
3. In the instrument menu, look for "MIDI Generation"
4. Click to insert as AU instrument
5. The plugin interface should appear with:
   - Parameter controls (Role, Key, Temperature, etc)
   - GENERATE button
   - OUTPUT button (for piano roll window)
   - Status display

## Troubleshooting

### Build Error: "torch not found"
```
CMake Error at CMakeLists.txt:XX (find_package):
  Could not find a package configuration file provided by "Torch"
```
**Solution**: Ensure TORCH_INSTALL_PREFIX points to extracted libtorch directory:
```bash
cmake .. -DTORCH_INSTALL_PREFIX=/Users/username/Downloads/libtorch -DCMAKE_BUILD_TYPE=Release
```

### Build Error: "JUCE not found"
```
CMake Error: JUCE source path not found
```
**Solution**: Clone JUCE into plugin/juce/JUCE:
```bash
cd plugin/juce
git clone https://github.com/juce-framework/JUCE.git JUCE
```

Or update CMakeLists.txt:
```cmake
set(JUCE_SOURCE_PATH "/path/to/JUCE")
add_subdirectory(${JUCE_SOURCE_PATH} EXCLUDE_FROM_ALL)
```

### Plugin not appearing in FL Studio
1. Check installation path: `~/Library/Audio/Plug-Ins/Components/`
2. Restart FL Studio completely
3. In FL Studio settings, manually rescan plugins
4. Check macOS System Preferences > Security & Privacy (if notarization needed)

### Linking error: "undefined reference to torch symbols"
**Solution**: Set RPATH in CMakeLists.txt (already done):
```cmake
set_target_properties(MIDIGenerationPlugin PROPERTIES
    BUILD_RPATH "@loader_path"
    INSTALL_RPATH "@loader_path"
)
```

## Next Steps (Phase 2)

Once Phase 1 compiles successfully:

1. **Integrate PyTorch C++ API**
   - Load model checkpoint in ModelInference::loadCheckpoint()
   - Implement token generation logic

2. **Test token generation**
   - Click Generate button
   - Verify GeneratorThread runs async (no UI hang)

3. **Connect tokens to MIDI**
   - Implement vocab.json parsing
   - Convert token IDs → NOTE_ON/NOTE_OFF events

## File Structure
```
plugin/juce/
├── CMakeLists.txt                  ← Build config
├── JUCE/                           ← JUCE framework (clone needed)
├── Source/
│   ├── PluginProcessor.h/cpp       ← Main AudioProcessor
│   ├── PluginEditor.h/cpp          ← UI with 14 parameters
│   ├── GeneratorThread.h/cpp       ← Async inference
│   ├── ModelInference.h/cpp        ← PyTorch wrapper (Phase 2)
│   ├── MidiGenerator.h/cpp         ← Tokens→MIDI (Phase 2)
│   ├── OutputWindow.h/cpp          ← Piano roll window
│   ├── MidiVisualizer.h/cpp        ← Piano roll drawing
│   ├── MidiFileExporter.h/cpp      ← MIDI save (Phase 5)
│   └── RotaryKnob.h/cpp            ← Custom UI element
└── bin/
    ├── model_best.pt               ← Will be added in Phase 2
    └── vocab.json                  ← Will be added in Phase 2
```

## Status
- **Current Phase**: 1 (Skeleton)
- **Estimated Compilation Time**: 3-5 minutes
- **Plugin Size**: ~50MB (includes LibTorch)
- **Test Status**: Ready for audio thread testing after compilation

---

**Generated**: April 20, 2026 | **Status**: Ready for Phase 1 compilation
