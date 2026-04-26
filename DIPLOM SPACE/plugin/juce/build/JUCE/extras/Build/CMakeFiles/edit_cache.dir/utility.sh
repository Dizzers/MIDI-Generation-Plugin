set -e

cd "/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM SPACE/plugin/juce/build/JUCE/extras/Build"
/opt/homebrew/bin/ccmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
