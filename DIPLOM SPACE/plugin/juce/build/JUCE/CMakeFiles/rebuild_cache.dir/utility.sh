set -e

cd "/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM SPACE/plugin/juce/build/JUCE"
/opt/homebrew/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
