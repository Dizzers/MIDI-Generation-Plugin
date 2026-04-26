#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>

class PluginProcessor;

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

    // === REFERENCES ===
    PluginProcessor& processor;
    juce::AudioProcessorValueTreeState& apvts;

    // === UI COMPONENTS ===
    // Basic Settings
    std::unique_ptr<juce::ComboBox> roleBox;
    std::unique_ptr<juce::ComboBox> keyBox;
    std::unique_ptr<juce::Slider> maxLenSlider;
    std::unique_ptr<juce::Slider> targetDurationSlider;
    std::unique_ptr<juce::Slider> seedSlider;

    // Rotary Knobs (Main Parameters)
    std::unique_ptr<juce::Slider> temperatureKnob;
    std::unique_ptr<juce::Slider> melodyLeapKnob;
    std::unique_ptr<juce::Slider> repetitionPenaltyKnob;

    // Sampling Control
    std::unique_ptr<juce::Slider> topKSlider;
    std::unique_ptr<juce::Slider> topPSlider;
    std::unique_ptr<juce::Slider> ngramSlider;

    // Musical Constraints
    std::unique_ptr<juce::ComboBox> harmonyModeBox;
    std::unique_ptr<juce::Slider> harmonyBiasSlider;
    std::unique_ptr<juce::ComboBox> primerModeBox;
    std::unique_ptr<juce::Slider> primerLenSlider;

    // Buttons
    std::unique_ptr<juce::TextButton> generateButton;
    std::unique_ptr<juce::TextButton> randomizeButton;
    std::unique_ptr<juce::TextButton> outputWindowButton;

    // Status Display
    std::unique_ptr<juce::Label> statusLabel;
    std::unique_ptr<juce::Label> generationProgressLabel;

    // === ATTACHMENTS ===
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>> sliderAttachments;
    std::vector<std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment>> comboAttachments;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginEditor)
};
