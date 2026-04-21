#include "MidiVisualizer.h"

MidiVisualizer::MidiVisualizer()
{
    setSize(800, 400);
}

MidiVisualizer::~MidiVisualizer()
{
}

void MidiVisualizer::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0x1a1a1a));  // Dark background
    
    // Draw grid
    g.setColour(juce::Colour(0x333333));
    for (int i = 0; i < getWidth(); i += 50) {
        g.drawVerticalLine(i, 0, getHeight());
    }
    
    // Draw piano keys on left
    g.setColour(juce::Colours::white);
    g.setFont(10.0f);
    for (int octave = 2; octave < 6; ++octave) {
        for (int note = 0; note < 12; ++note) {
            int noteValue = octave * 12 + note;
            int y = getHeight() - ((noteValue - 24) * getHeight() / 60);
            g.drawText("C" + juce::String(octave), 5, y - 5, 20, 10, juce::Justification::centred);
        }
    }
    
    // Draw MIDI notes as rectangles (TODO: Phase 5)
    g.setColour(juce::Colour(0x00ff80));  // Green for notes
    for (const auto& msg : midiMessages) {
        if (msg.isNoteOn()) {
            int note = msg.getNoteNumber();
            int y = getHeight() - ((note - 24) * getHeight() / 60);
            g.fillRect(100, y - 5, 100, 10);
        }
    }
    
    // Status text
    g.setColour(juce::Colours::white);
    g.drawText("Piano Roll Preview (Phase 5)", 10, getHeight() - 30, 300, 20, juce::Justification::topLeft);
}

void MidiVisualizer::setMidiMessages(const std::vector<juce::MidiMessage>& messages)
{
    midiMessages = messages;
    repaint();
}
