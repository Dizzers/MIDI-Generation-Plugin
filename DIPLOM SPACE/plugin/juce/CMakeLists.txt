cmake_minimum_required(VERSION 3.20)
project(MIDIGenerationPlugin VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# === Options ============================================================
option(USE_ONNXRUNTIME       "Build with ONNX Runtime inference (SkyTNT)" ON)
option(BUILD_SMOKE_TEST_CLI  "Build a tiny CLI that exercises the runtime" ON)
# Optional override: set ONNXRUNTIME_DIR to an unpacked onnxruntime-osx*-1.x.y/
# directory if you don't want CMake to download it for you. The directory must
# contain include/ and lib/ subfolders.
set(ONNXRUNTIME_DIR "" CACHE PATH "Pre-extracted ONNX Runtime root (with include/ and lib/)")
set(ONNXRUNTIME_VERSION "1.17.3" CACHE STRING "ONNX Runtime release tag to fetch when ONNXRUNTIME_DIR is empty")

# === JUCE ===============================================================
add_subdirectory(JUCE)

juce_add_plugin(MIDIGenerationPlugin
    PRODUCT_NAME            "MIDI Generation"
    COMPANY_NAME            "DiplomaProject"
    PLUGIN_MANUFACTURER_CODE "MIDI"
    PLUGIN_CODE             "MIDI"
    IS_SYNTH                FALSE
    IS_MIDI_EFFECT          TRUE
    NEEDS_MIDI_INPUT        TRUE
    NEEDS_MIDI_OUTPUT       TRUE
    NEEDS_TRANSPORT_AUDIO   FALSE
    FORMATS                 AU VST3 Standalone
    AU_MAIN_TYPE            kAudioUnitType_MIDIProcessor
    VST3_CATEGORIES         "Instrument" "Synth"
    COPY_PLUGIN_AFTER_BUILD FALSE
    PLUGIN_AU_EXPORT_PREFIX "MIDIGenerationAU"
)

set(PLUGIN_SOURCES
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
    Source/AnalogLookAndFeel.cpp
    Source/ModelInference.cpp
    Source/MidiPostProcessor.cpp
    Source/GeneratorThread.cpp
    Source/RotaryKnob.cpp
    Source/OutputWindow.cpp
    Source/MidiVisualizer.cpp
    Source/MidiFileExporter.cpp
    Source/SkyTNTTokenizer.cpp
    Source/SkyTNTRuntime.cpp
)

target_sources(MIDIGenerationPlugin PRIVATE ${PLUGIN_SOURCES})

target_link_libraries(MIDIGenerationPlugin PRIVATE
    juce::juce_audio_processors
    juce::juce_audio_formats
    juce::juce_audio_utils
    juce::juce_audio_basics
    juce::juce_core
    juce::juce_data_structures
    juce::juce_events
    juce::juce_graphics
    juce::juce_gui_basics
    juce::juce_gui_extra)

target_compile_definitions(MIDIGenerationPlugin PRIVATE
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_USE_CURL=0
    JUCE_WEB_BROWSER=0)

target_include_directories(MIDIGenerationPlugin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Source)

# === ONNX Runtime =======================================================
function(_setup_onnxruntime _root)
    set(ORT_INCLUDE "${_root}/include")
    if(EXISTS "${_root}/include/onnxruntime")
        set(ORT_INCLUDE "${_root}/include/onnxruntime")
    endif()
    if(NOT EXISTS "${ORT_INCLUDE}/onnxruntime_cxx_api.h")
        # Some upstream archives nest headers under <root>/include/onnxruntime/core/session/
        if(EXISTS "${_root}/include/onnxruntime/core/session/onnxruntime_cxx_api.h")
            set(ORT_INCLUDE "${_root}/include/onnxruntime/core/session")
        endif()
    endif()

    if(WIN32)
        find_library(ORT_LIB onnxruntime PATHS "${_root}/lib" REQUIRED NO_DEFAULT_PATH)
    elseif(APPLE)
        find_library(ORT_LIB onnxruntime PATHS "${_root}/lib" REQUIRED NO_DEFAULT_PATH)
    else()
        find_library(ORT_LIB onnxruntime PATHS "${_root}/lib" "${_root}/lib64" REQUIRED NO_DEFAULT_PATH)
    endif()

    set(ORT_INCLUDE "${ORT_INCLUDE}" PARENT_SCOPE)
    set(ORT_LIB     "${ORT_LIB}"     PARENT_SCOPE)
    set(ORT_ROOT    "${_root}"       PARENT_SCOPE)
endfunction()

if(USE_ONNXRUNTIME)
    if(ONNXRUNTIME_DIR)
        if(NOT EXISTS "${ONNXRUNTIME_DIR}")
            message(FATAL_ERROR "ONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} does not exist")
        endif()
        _setup_onnxruntime("${ONNXRUNTIME_DIR}")
    else()
        include(FetchContent)
        if(WIN32)
            set(_ort_pkg "onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip")
        elseif(APPLE)
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
                set(_ort_pkg "onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}.tgz")
            else()
                set(_ort_pkg "onnxruntime-osx-x86_64-${ONNXRUNTIME_VERSION}.tgz")
            endif()
        else()
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
                set(_ort_pkg "onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz")
            else()
                set(_ort_pkg "onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
            endif()
        endif()
        set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${_ort_pkg}")
        message(STATUS "Fetching ONNX Runtime: ${_ort_url}")
        FetchContent_Declare(onnxruntime URL ${_ort_url})
        FetchContent_MakeAvailable(onnxruntime)
        _setup_onnxruntime("${onnxruntime_SOURCE_DIR}")
    endif()
    message(STATUS "ONNXRuntime include: ${ORT_INCLUDE}")
    message(STATUS "ONNXRuntime lib:     ${ORT_LIB}")
    target_compile_definitions(MIDIGenerationPlugin PRIVATE MIDI_GEN_USE_ONNX=1)
    target_include_directories(MIDIGenerationPlugin PRIVATE "${ORT_INCLUDE}")
    target_link_libraries(MIDIGenerationPlugin PRIVATE "${ORT_LIB}")
else()
    target_compile_definitions(MIDIGenerationPlugin PRIVATE MIDI_GEN_USE_ONNX=0)
endif()

# === Compile flags ======================================================
if(MSVC)
    target_compile_options(MIDIGenerationPlugin PRIVATE /W4)
else()
    target_compile_options(MIDIGenerationPlugin PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_target_properties(MIDIGenerationPlugin PROPERTIES
    BUILD_RPATH   "@loader_path;@executable_path"
    INSTALL_RPATH "@loader_path;@executable_path")

# === Smoke-test CLI =====================================================
if(BUILD_SMOKE_TEST_CLI)
    add_executable(skytnt_smoke_test
        Source/SmokeTest.cpp
        Source/SkyTNTTokenizer.cpp
        Source/SkyTNTRuntime.cpp)
    target_link_libraries(skytnt_smoke_test PRIVATE
        juce::juce_audio_basics
        juce::juce_audio_formats
        juce::juce_core
        juce::juce_events)
    target_compile_definitions(skytnt_smoke_test PRIVATE
        JUCE_USE_CURL=0
        JUCE_WEB_BROWSER=0)
    target_include_directories(skytnt_smoke_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Source)
    if(USE_ONNXRUNTIME)
        target_compile_definitions(skytnt_smoke_test PRIVATE MIDI_GEN_USE_ONNX=1)
        target_include_directories(skytnt_smoke_test PRIVATE "${ORT_INCLUDE}")
        target_link_libraries(skytnt_smoke_test PRIVATE "${ORT_LIB}")
    else()
        target_compile_definitions(skytnt_smoke_test PRIVATE MIDI_GEN_USE_ONNX=0)
    endif()
    set_target_properties(skytnt_smoke_test PROPERTIES
        BUILD_RPATH   "@loader_path;@executable_path"
        INSTALL_RPATH "@loader_path;@executable_path")
endif()

# === Build summary ======================================================
message(STATUS "MIDI Generation Plugin")
message(STATUS "  USE_ONNXRUNTIME:      ${USE_ONNXRUNTIME}")
message(STATUS "  BUILD_SMOKE_TEST_CLI: ${BUILD_SMOKE_TEST_CLI}")
message(STATUS "  Output dir:           ${CMAKE_BINARY_DIR}/MIDIGenerationPlugin_artefacts/")

add_custom_command(TARGET MIDIGenerationPlugin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Plugin compiled successfully!"
    COMMAND ${CMAKE_COMMAND} -E echo "Copy .au to: ~/Library/Audio/Plug-Ins/Components/")
