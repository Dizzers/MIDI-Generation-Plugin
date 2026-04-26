set -e

cd "/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM SPACE/plugin/juce/build/JUCE/tools"
/opt/homebrew/bin/cmake -DCMAKE_INSTALL_DO_STRIP=1 -P cmake_install.cmake
