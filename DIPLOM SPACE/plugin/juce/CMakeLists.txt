cmake_minimum_required(VERSION 3.20)
project(MIDIGenerationPlugin VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# === JUCE Configuration ===
add_subdirectory(JUCE)  # Ensure JUCE is available

juce_add_plugin(MIDIGenerationPlugin
    PRODUCT_NAME "MIDI Generation"
    COMPANY_NAME "DiplomaProject"
    PLUGIN_MANUFACTURER_CODE "MIDI"
    PLUGIN_CODE "MIDI"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT TRUE
    NEEDS_TRANSPORT_AUDIO FALSE
    AU_MAIN_TYPE "aumu"
    VST3_CATEGORIES "Instrument" "Synth"
    COPY_PLUGIN_AFTER_BUILD FALSE
    PLUGIN_AU_EXPORT_PREFIX "MIDIGenerationAU"
)

# === Source Files ===
target_sources(MIDIGenerationPlugin PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
    Source/ModelInference.cpp
    Source/MidiGenerator.cpp
    Source/GeneratorThread.cpp
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

# === LibTorch / PyTorch C++ ===
# IMPORTANT: Download LibTorch binary for macOS from https://pytorch.org/get-started/locally/
# Then set TORCH_INSTALL_PREFIX to the extracted directory
# Example: -DTORCH_INSTALL_PREFIX=/path/to/libtorch

set(TORCH_INSTALL_PREFIX "" CACHE STRING "Path to LibTorch installation (required)")

if(NOT TORCH_INSTALL_PREFIX)
    message(FATAL_ERROR "TORCH_INSTALL_PREFIX not set. Please provide path to LibTorch binary.")
endif()

set(CMAKE_PREFIX_PATH ${TORCH_INSTALL_PREFIX} ${CMAKE_PREFIX_PATH})

find_package(Torch REQUIRED)

target_link_libraries(MIDIGenerationPlugin PRIVATE
    ${TORCH_LIBRARIES}
)

# === Compiler Flags ===
if(MSVC)
    target_compile_options(MIDIGenerationPlugin PRIVATE /W4)
else()
    target_compile_options(MIDIGenerationPlugin PRIVATE -Wall -Wextra -Wpedantic)
endif()

# === Include Directories ===
target_include_directories(MIDIGenerationPlugin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Source
    ${TORCH_INCLUDE_DIRS}
)

# === RPATH Configuration (macOS dylib loading) ===
set_target_properties(MIDIGenerationPlugin PROPERTIES
    BUILD_RPATH "@loader_path"
    INSTALL_RPATH "@loader_path"
)

# === Build Output ===
message(STATUS "Building MIDI Generation Plugin")
message(STATUS "TORCH_INSTALL_PREFIX: ${TORCH_INSTALL_PREFIX}")
message(STATUS "Output directory: ${CMAKE_BINARY_DIR}/MIDIGenerationPlugin_artefacts/")

# Post-build: Display target plugin path
add_custom_command(TARGET MIDIGenerationPlugin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Plugin compiled successfully!"
    COMMAND ${CMAKE_COMMAND} -E echo "Copy .au to: ~/Library/Audio/Plug-Ins/Components/"
)
