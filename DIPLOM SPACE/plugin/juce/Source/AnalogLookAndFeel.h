#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

class AnalogLookAndFeel final : public juce::LookAndFeel_V4
{
public:
    AnalogLookAndFeel();
    ~AnalogLookAndFeel() override = default;

    void drawButtonBackground(juce::Graphics& g,
                              juce::Button& button,
                              const juce::Colour& backgroundColour,
                              bool shouldDrawButtonAsHighlighted,
                              bool shouldDrawButtonAsDown) override;

    void drawLinearSlider(juce::Graphics& g,
                          int x, int y, int width, int height,
                          float sliderPos,
                          float minSliderPos,
                          float maxSliderPos,
                          const juce::Slider::SliderStyle style,
                          juce::Slider& slider) override;

    void drawRotarySlider(juce::Graphics& g,
                          int x, int y, int width, int height,
                          float sliderPosProportional,
                          float rotaryStartAngle,
                          float rotaryEndAngle,
                          juce::Slider& slider) override;

    void drawComboBox(juce::Graphics& g,
                      int width, int height,
                      bool isButtonDown,
                      int buttonX, int buttonY, int buttonW, int buttonH,
                      juce::ComboBox& box) override;

    juce::Font getTextButtonFont(juce::TextButton&, int buttonHeight) override;
    juce::Font getComboBoxFont(juce::ComboBox&) override;
    juce::Font getLabelFont(juce::Label&) override;

    // Simple design tokens
    juce::Colour panelBg;
    juce::Colour panelBorder;
    juce::Colour metalHi;
    juce::Colour metalLo;
    juce::Colour woodA;
    juce::Colour woodB;
    juce::Colour accent;
    juce::Colour text;
};

