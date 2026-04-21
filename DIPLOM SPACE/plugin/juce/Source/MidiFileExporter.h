#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <vector>

/**
 * Export MIDI messages to .mid file
 */
class MidiFileExporter
{
public:
    /**
     * Save MIDI messages to file
     */
    static bool saveMidiFile(
        const juce::File& targetFile,
        const std::vector<juce::MidiMessage>& midiMessages,
        double tempo = 120.0);

private:
    MidiFileExporter() = default;
};
