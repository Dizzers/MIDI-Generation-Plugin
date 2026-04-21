#include "OutputWindow.h"
#include "MidiVisualizer.h"
#include "PluginProcessor.h"

OutputWindow::OutputWindow(PluginProcessor& proc)
    : DocumentWindow("MIDI Output", juce::Desktop::getInstance().getDefaultLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId), juce::DocumentWindow::allButtons),
      processor(proc)
{
    setSize(800, 400);
    setResizable(true, true);
    
    midiVisualizer = std::make_unique<MidiVisualizer>();
    setContentOwned(midiVisualizer.get(), false);
    
    setVisible(true);
    DBG("OutputWindow created");
}

OutputWindow::~OutputWindow()
{
    DBG("OutputWindow destroyed");
}

void OutputWindow::closeButtonPressed()
{
    setVisible(false);
}

void OutputWindow::updateMidiDisplay(const std::vector<juce::MidiMessage>& midiMessages)
{
    if (midiVisualizer) {
        midiVisualizer->setMidiMessages(midiMessages);
    }
}
