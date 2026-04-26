#include "OutputWindow.h"
#include "MidiVisualizer.h"
#include "PluginProcessor.h"
#include "MidiFileExporter.h"

namespace
{
    class OutputContent : public juce::Component, public juce::DragAndDropContainer
    {
    };
}

OutputWindow::OutputWindow(PluginProcessor& proc)
    : DocumentWindow("MIDI Output", juce::Desktop::getInstance().getDefaultLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId), juce::DocumentWindow::allButtons),
      processor(proc)
{
    setSize(800, 400);
    setResizable(true, true);
    
    midiVisualizer = std::make_unique<MidiVisualizer>();
    content = std::make_unique<OutputContent>();

    regenButton = std::make_unique<juce::TextButton>("RE-GENERATE");
    regenButton->setColour(juce::TextButton::buttonColourId, juce::Colour(0x00D4FF));
    regenButton->setColour(juce::TextButton::textColourOffId, juce::Colours::black);
    regenButton->onClick = [this]() { processor.regenerateLast(); };

    exportButton = std::make_unique<juce::TextButton>("EXPORT .MID");
    exportButton->setColour(juce::TextButton::buttonColourId, juce::Colour(0x00ff80));
    exportButton->setColour(juce::TextButton::textColourOffId, juce::Colours::black);
    exportButton->onClick = [this]()
    {
        if (lastMessages.empty())
            return;
        juce::FileChooser chooser("Export MIDI", juce::File::getSpecialLocation(juce::File::userDocumentsDirectory), "*.mid");
        if (chooser.browseForFileToSave(true))
        {
            auto file = chooser.getResult().withFileExtension(".mid");
            const bool ok = MidiFileExporter::saveMidiFile(file, lastMessages, 120.0);
            DBG(ok ? "MIDI exported OK" : "MIDI export failed");
        }
    };

    dragDropButton = std::make_unique<juce::TextButton>("DRAG&DROP");
    dragDropButton->setColour(juce::TextButton::buttonColourId, juce::Colour(0x666666));
    dragDropButton->setColour(juce::TextButton::textColourOffId, juce::Colours::white);
    dragDropButton->onClick = [this]()
    {
        if (lastMessages.empty())
            return;
        auto tmp = juce::File::getSpecialLocation(juce::File::tempDirectory)
                       .getChildFile("juce_midi_gen_drag.mid");
        if (!MidiFileExporter::saveMidiFile(tmp, lastMessages, 120.0))
            return;

        auto* dnd = dynamic_cast<juce::DragAndDropContainer*>(content.get());
        if (dnd != nullptr)
        {
            juce::StringArray files;
            files.add(tmp.getFullPathName());
            dnd->performExternalDragDropOfFiles(files, true, nullptr);
        }
    };

    infoLabel = std::make_unique<juce::Label>("info", "No clip yet");
    infoLabel->setColour(juce::Label::textColourId, juce::Colours::white);
    infoLabel->setJustificationType(juce::Justification::centredLeft);

    content->addAndMakeVisible(*midiVisualizer);
    content->addAndMakeVisible(*regenButton);
    content->addAndMakeVisible(*exportButton);
    content->addAndMakeVisible(*dragDropButton);
    content->addAndMakeVisible(*infoLabel);

    content->onResize = [this]()
    {
        auto area = content->getLocalBounds().reduced(10);
        auto top = area.removeFromTop(32);
        regenButton->setBounds(top.removeFromLeft(140));
        top.removeFromLeft(10);
        exportButton->setBounds(top.removeFromLeft(120));
        top.removeFromLeft(10);
        dragDropButton->setBounds(top.removeFromLeft(120));
        top.removeFromLeft(10);
        infoLabel->setBounds(top);
        area.removeFromTop(8);
        midiVisualizer->setBounds(area);
    };

    setContentOwned(content.get(), false);
    
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
    lastMessages = midiMessages;
    if (midiVisualizer) {
        midiVisualizer->setMidiMessages(midiMessages);
    }

    int noteOn = 0;
    double maxT = 0.0;
    for (const auto& m : midiMessages)
    {
        if (m.isNoteOn())
            ++noteOn;
        maxT = std::max(maxT, m.getTimeStamp());
    }
    if (infoLabel)
        infoLabel->setText("Notes: " + juce::String(noteOn) + " | Duration: " + juce::String(maxT, 2) + "s", juce::dontSendNotification);
}
