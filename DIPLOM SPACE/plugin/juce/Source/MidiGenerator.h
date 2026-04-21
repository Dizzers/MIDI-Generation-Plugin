#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <vector>

/**
 * Convert token IDs to MIDI messages
 */
class MidiGenerator
{
public:
    struct MidiEvent
    {
        int timeMs = 0;
        juce::MidiMessage message;
    };

    /**
     * Convert token sequence to MIDI events
     * Uses vocab.json for token→MIDI mapping
     */
    static std::vector<juce::MidiMessage> convertTokensToMidi(
        const std::vector<int>& tokenIds);

private:
    static bool parseTokenId(int id, juce::String& tokenName);
};
