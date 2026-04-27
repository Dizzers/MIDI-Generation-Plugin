cmake_minimum_required(VERSION 3.20)
project(MIDIGenerationPlugin VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# === Options ===
# Default OFF so the plugin can build/run without a local LibTorch install.
option(USE_TORCH "Build with LibTorch inference enabled" OFF)

# === JUCE Configuration ===
add_subdirectory(JUCE)  # Ensure JUCE is available

juce_add_plugin(MIDIGenerationPlugin
    PRODUCT_NAME "MIDI Generation"
    COMPANY_NAME "DiplomaProject"
    PLUGIN_MANUFACTURER_CODE "MIDI"
    PLUGIN_CODE "MIDI"
    IS_SYNTH FALSE
    IS_MIDI_EFFECT TRUE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT TRUE
    NEEDS_TRANSPORT_AUDIO FALSE
    FORMATS AU VST3 Standalone
    # MIDI processor AU (generates/forwards MIDI)
    AU_MAIN_TYPE kAudioUnitType_MIDIProcessor
    VST3_CATEGORIES "Instrument" "Synth"
    COPY_PLUGIN_AFTER_BUILD FALSE
    PLUGIN_AU_EXPORT_PREFIX "MIDIGenerationAU"
)

# === Source Files ===
target_sources(MIDIGenerationPlugin PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
    Source/AnalogLookAndFeel.cpp
    Source/ModelInference.cpp
    Source/MidiGenerator.cpp
    Source/MidiPostProcessor.cpp
    Source/GeneratorThread.cpp
    Source/AnimatedKnob.h
    Source/RotaryKnob.cpp
    Source/OutputWindow.cpp
    Source/MidiVisualizer.cpp
    Source/MidiFileExporter.cpp
)

# === JUCE Dependencies ===
target_link_libraries(MIDIGenerationPlugin PRIVATE
    juce::juce_audio_processors
    juce::juce_audio_formats
    juce::juce_audio_utils
    juce::juce_core
    juce::juce_data_structures
    juce::juce_events
    juce::juce_graphics
    juce::juce_gui_basics
    juce::juce_gui_extra
)

# We don't ship VST2, so disable JUCE's VST2/VST3 migration checks.
target_compile_definitions(MIDIGenerationPlugin PRIVATE
    JUCE_VST3_CAN_REPLACE_VST2=0
)

# === LibTorch / PyTorch C++ ===
# IMPORTANT: Download LibTorch binary for macOS from https://pytorch.org/get-started/locally/
# Then set TORCH_INSTALL_PREFIX to the extracted directory
# Example: -DTORCH_INSTALL_PREFIX=/path/to/libtorch

if (USE_TORCH)
    set(TORCH_INSTALL_PREFIX "" CACHE STRING "Path to LibTorch installation (optional; required only when USE_TORCH=ON)")

    if(NOT TORCH_INSTALL_PREFIX OR NOT EXISTS "${TORCH_INSTALL_PREFIX}")
        message(WARNING "USE_TORCH=ON but TORCH_INSTALL_PREFIX is missing/invalid. Building without LibTorch.")
        set(USE_TORCH OFF)
    endif()
endif()

if (USE_TORCH)
    set(CMAKE_PREFIX_PATH ${TORCH_INSTALL_PREFIX} ${CMAKE_PREFIX_PATH})
    find_package(Torch REQUIRED)

    target_compile_definitions(MIDIGenerationPlugin PRIVATE MIDI_GEN_USE_TORCH=1)
    target_link_libraries(MIDIGenerationPlugin PRIVATE ${TORCH_LIBRARIES})
    target_include_directories(MIDIGenerationPlugin PRIVATE ${TORCH_INCLUDE_DIRS})
else()
    target_compile_definitions(MIDIGenerationPlugin PRIVATE MIDI_GEN_USE_TORCH=0)
endif()

# === Compiler Flags ===
if(MSVC)
    target_compile_options(MIDIGenerationPlugin PRIVATE /W4)
else()
    target_compile_options(MIDIGenerationPlugin PRIVATE -Wall -Wextra -Wpedantic)
endif()

# === Include Directories ===
target_include_directories(MIDIGenerationPlugin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Source
)

# === RPATH Configuration (macOS dylib loading) ===
set_target_properties(MIDIGenerationPlugin PROPERTIES
    BUILD_RPATH "@loader_path"
    INSTALL_RPATH "@loader_path"
)

# === Build Output ===
message(STATUS "Building MIDI Generation Plugin")
message(STATUS "USE_TORCH: ${USE_TORCH}")
if (USE_TORCH)
    message(STATUS "TORCH_INSTALL_PREFIX: ${TORCH_INSTALL_PREFIX}")
endif()
message(STATUS "Output directory: ${CMAKE_BINARY_DIR}/MIDIGenerationPlugin_artefacts/")

# Post-build: Display target plugin path
add_custom_command(TARGET MIDIGenerationPlugin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Plugin compiled successfully!"
    COMMAND ${CMAKE_COMMAND} -E echo "Copy .au to: ~/Library/Audio/Plug-Ins/Components/"
)
