set -e

cd "/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM SPACE/plugin/juce/build"
/opt/homebrew/bin/ccmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
