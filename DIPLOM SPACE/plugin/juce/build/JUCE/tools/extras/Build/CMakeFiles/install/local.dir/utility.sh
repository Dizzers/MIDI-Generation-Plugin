set -e

cd "/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM SPACE/plugin/juce/build/JUCE/tools/extras/Build"
/opt/homebrew/bin/cmake -DCMAKE_INSTALL_LOCAL_ONLY=1 -P cmake_install.cmake
