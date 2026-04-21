#pragma once
#include <juce_gui_extra/juce_gui_extra.h>

class PluginProcessor;
class MidiVisualizer;

/**
 * Separate output window showing MIDI piano roll and controls
 * Displays generated MIDI visually and allows playback/export
 */
class OutputWindow : public juce::DocumentWindow
{
public:
    OutputWindow(PluginProcessor& proc);
    ~OutputWindow() override;

    void closeButtonPressed() override;
    void updateMidiDisplay(const std::vector<juce::MidiMessage>& midiMessages);

private:
    PluginProcessor& processor;
    std::unique_ptr<MidiVisualizer> midiVisualizer;
};
