#pragma once
#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>

/**
 * Piano roll visualizer for MIDI events
 * Displays notes as rectangles on a time/pitch grid
 */
class MidiVisualizer : public juce::Component
{
public:
    MidiVisualizer();
    ~MidiVisualizer() override;

    void paint(juce::Graphics& g) override;
    void setMidiMessages(const std::vector<juce::MidiMessage>& messages);

private:
    struct NoteRect
    {
        int pitch = 60;
        double start = 0.0;
        double end = 0.0;
        int velocity = 100;
    };

    void rebuildNotes();

    std::vector<juce::MidiMessage> midiMessages;
    std::vector<NoteRect> notes;
};
