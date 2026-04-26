#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <vector>
#include <unordered_map>

/**
 * Convert token IDs to MIDI messages
 */
class MidiGenerator
{
public:
    /**
     * Convert token sequence to MIDI events
     * Uses vocab.json for token→MIDI mapping
     *
     * Important: returned MidiMessage objects have their timeStamp set (seconds since start),
     * based on TIME_SHIFT tokens produced by the Python tokenizer.
     */
    static std::vector<juce::MidiMessage> convertTokensToMidi(
        const std::vector<int>& tokenIds);

private:
    static bool ensureVocabLoaded();
    static juce::File findVocabJson();
    static bool tokenIdToString(int id, juce::String& tokenName);
    static bool parseHexSuffix(const juce::String& token, int& valueOut);

    static std::unordered_map<int, juce::String> idToToken;
    static bool vocabLoaded;
};
