set -e

cd "/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM SPACE/plugin/juce/build"
/opt/homebrew/bin/cmake -E copy "/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM SPACE/plugin/juce/build/MIDIGenerationPlugin_artefacts/JuceLibraryCode/MIDIGenerationPlugin_VST3/PkgInfo" "/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM SPACE/plugin/juce/build/MIDIGenerationPlugin_artefacts/Release/VST3/MIDI Generation.vst3/Contents"
cd "/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM SPACE/plugin/juce/build"
/opt/homebrew/bin/cmake "-Dsrc=/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM SPACE/plugin/juce/build/MIDIGenerationPlugin_artefacts/Release/VST3/MIDI Generation.vst3" -P "/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM SPACE/plugin/juce/JUCE/extras/Build/CMake/checkBundleSigning.cmake"
cd "/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM SPACE/plugin/juce/build"
/opt/homebrew/bin/cmake -E echo removing\ moduleinfo.json
/opt/homebrew/bin/cmake -E remove -f /Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM\ SPACE/plugin/juce/build/MIDIGenerationPlugin_artefacts/Release/VST3/MIDI\ Generation.vst3/Contents/moduleinfo.json /Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM\ SPACE/plugin/juce/build/MIDIGenerationPlugin_artefacts/Release/VST3/MIDI\ Generation.vst3/Contents/Resources/moduleinfo.json
cd "/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM SPACE/plugin/juce/build"
/opt/homebrew/bin/cmake -E echo creating\ /Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM\ SPACE/plugin/juce/build/MIDIGenerationPlugin_artefacts/Release/VST3/MIDI\ Generation.vst3
/opt/homebrew/bin/cmake -E make_directory /Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM\ SPACE/plugin/juce/build/MIDIGenerationPlugin_artefacts/Release/VST3/MIDI\ Generation.vst3/Contents/Resources
"/Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM SPACE/plugin/juce/build/MIDIGenerationPlugin_vst3_helper" > /Volumes/T7/Университет/PythonProject/MIDI-Generation-Plugin/DIPLOM\ SPACE/plugin/juce/build/MIDIGenerationPlugin_artefacts/Release/VST3/MIDI\ Generation.vst3/Contents/Resources/moduleinfo.json
