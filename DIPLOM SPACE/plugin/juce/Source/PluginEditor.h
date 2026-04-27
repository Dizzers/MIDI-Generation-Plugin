#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <unordered_map>

#include "AnalogLookAndFeel.h"

class PluginProcessor;
class MidiVisualizer;

/**
 * Main Editor UI for MIDI Generation Plugin
 * Displays:
 * - Parameter controls (14 parameters)
 * - Rotary knobs for Temperature, Melody Leap, Repetition Penalty
 * - ComboBoxes for Role, Key, Harmony Mode
 * - Sliders for numeric parameters
 * - Generate button and status display
 * - Output Window button
 */
class PluginEditor : public juce::AudioProcessorEditor,
                     public juce::DragAndDropContainer,
                     private juce::Timer
{
public:
    explicit PluginEditor(PluginProcessor&);
    ~PluginEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;
    void updateStatusDisplay();
    void createControls();
    void applyPageVisibility();

    // === REFERENCES ===
    PluginProcessor& processor;
    juce::AudioProcessorValueTreeState& apvts;

    class ControlsPanel;
    std::unique_ptr<ControlsPanel> controlsPanel;
    std::unique_ptr<juce::Viewport> controlsViewport;

    class ToolbarBlock;
    std::unique_ptr<ToolbarBlock> actionsBlock;
    std::unique_ptr<ToolbarBlock> bpmBlock;
    std::unique_ptr<juce::Label> bpmTopLabel;
    std::unique_ptr<juce::TextEditor> bpmTopEditor;
    juce::RangedAudioParameter* bpmParam = nullptr;
    void commitBpmFromText();

    std::unique_ptr<juce::Label> goLabel;
    std::unique_ptr<juce::Label> regenLabel;
    std::unique_ptr<juce::Label> exportLabel;
    std::unique_ptr<juce::Label> dragLabel;

    enum class Page { Gen, Feel, Post };
    Page currentPage = Page::Gen;

    std::unique_ptr<juce::TextButton> pageGenButton;
    std::unique_ptr<juce::TextButton> pageFeelButton;
    std::unique_ptr<juce::TextButton> pagePostButton;

    // Captions for compact labeled UI
    std::vector<std::unique_ptr<juce::Label>> captions;
    std::unordered_map<juce::Component*, juce::Label*> captionByComponent;
    juce::Label& addCaption(juce::Component& target, const juce::String& text);

    class Divider;
    std::vector<std::unique_ptr<Divider>> dividers;
    Divider& addDivider();

    class GroupBlock;
    std::vector<std::unique_ptr<GroupBlock>> groupBlocks;
    GroupBlock& addGroupBlock(const juce::String& title);

    // === UI COMPONENTS ===
    // Basic Settings
    std::unique_ptr<juce::ComboBox> keyBox;
    std::unique_ptr<juce::Slider> maxLenSlider;
    std::unique_ptr<juce::Slider> targetDurationSlider;
    std::unique_ptr<juce::Slider> seedSlider;

    // Rotary Knobs (Main Parameters)
    std::unique_ptr<juce::Slider> temperatureKnob;
    std::unique_ptr<juce::Slider> melodyLeapKnob;
    std::unique_ptr<juce::Slider> repetitionPenaltyKnob;
    std::unique_ptr<juce::Slider> velocityFeelKnob;
    std::unique_ptr<juce::Slider> grooveFeelKnob;

    // Sampling Control
    std::unique_ptr<juce::Slider> topKSlider;
    std::unique_ptr<juce::Slider> topPSlider;
    std::unique_ptr<juce::Slider> ngramSlider;

    // Musical Constraints
    std::unique_ptr<juce::ComboBox> harmonyModeBox;
    std::unique_ptr<juce::Slider> harmonyBiasSlider;
    std::unique_ptr<juce::ComboBox> primerModeBox;
    std::unique_ptr<juce::Slider> primerLenSlider;
    std::unique_ptr<juce::Slider> maxPolyphonySlider;
    std::unique_ptr<juce::Slider> minBodyTokensSlider;

    // Performance (post)
    std::unique_ptr<juce::Slider> bpmSlider;
    std::unique_ptr<juce::ComboBox> quantizeGridBox;
    std::unique_ptr<juce::Slider> quantizeAmountSlider;
    std::unique_ptr<juce::Slider> swingAmountSlider;
    std::unique_ptr<juce::Slider> humanizeTimeSlider;
    std::unique_ptr<juce::Slider> humanizeVelocitySlider;
    std::unique_ptr<juce::Slider> velocityMinSlider;
    std::unique_ptr<juce::Slider> velocityMaxSlider;

    // Buttons
    std::unique_ptr<juce::TextButton> generateButton;
    std::unique_ptr<juce::TextButton> randomizeButton;
    std::unique_ptr<juce::TextButton> regenButton;
    std::unique_ptr<juce::TextButton> exportButton;
    std::unique_ptr<juce::TextButton> dragDropButton;

    // Status Display
    std::unique_ptr<juce::Label> statusLabel;
    std::unique_ptr<juce::Label> generationProgressLabel;
    std::unique_ptr<juce::Label> midiInfoLabel;


    std::unique_ptr<MidiVisualizer> midiVisualizer;
    std::vector<juce::MidiMessage> lastMidiMessages;
    uint64_t lastMidiVersion = 0;

    AnalogLookAndFeel analogLookAndFeel;

    // === ATTACHMENTS ===
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>> sliderAttachments;
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment>> comboAttachments;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginEditor)
};
