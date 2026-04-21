#include "MidiGenerator.h"

std::vector<juce::MidiMessage> MidiGenerator::convertTokensToMidi(const std::vector<int>& tokenIds)
{
    std::vector<juce::MidiMessage> midiEvents;
    
    // TODO: Implement token→MIDI conversion with vocab.json lookup
    // For Phase 1: Generate dummy MIDI to test audio thread
    int currentTime = 0;
    for (size_t i = 0; i < tokenIds.size(); ++i) {
        if (i % 3 == 0) {
            // NOTE_ON
            int note = 60 + (i % 12);
            midiEvents.push_back(juce::MidiMessage::noteOn(1, note, 100));
        } else if (i % 3 == 1) {
            // TIME_SHIFT
            currentTime += 100;
        } else {
            // NOTE_OFF
            int note = 60 + ((i - 1) % 12);
            midiEvents.push_back(juce::MidiMessage::noteOff(1, note, 0));
        }
    }
    
    return midiEvents;
}

bool MidiGenerator::parseTokenId(int id, juce::String& tokenName)
{
    // TODO: Lookup vocab.json for token name
    tokenName = "TOKEN_" + juce::String(id);
    return true;
}
