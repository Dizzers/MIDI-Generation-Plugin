#pragma once

#include <juce_gui_basics/juce_gui_basics.h>

// Small helper: smooth hover/drag animation for rotary knobs.
class AnimatedKnob : public juce::Slider,
                     private juce::Timer
{
public:
    AnimatedKnob()
        : juce::Slider(juce::Slider::RotaryVerticalDrag, juce::Slider::TextBoxBelow)
    {
        setMouseCursor(juce::MouseCursor::PointingHandCursor);
        // Smaller textbox to avoid overlaps in compact grids
        setTextBoxStyle(juce::Slider::TextBoxBelow, false, 44, 14);
        setTextBoxIsEditable(false);
        setPopupDisplayEnabled(true, true, nullptr);
    }

    ~AnimatedKnob() override { stopTimer(); }

    float getAnimAmount() const { return anim; } // 0..1

    void mouseEnter(const juce::MouseEvent& e) override
    {
        juce::Slider::mouseEnter(e);
        target = 1.0f;
        startTimerHz(60);
    }

    void mouseExit(const juce::MouseEvent& e) override
    {
        juce::Slider::mouseExit(e);
        target = isMouseButtonDown() ? 1.0f : 0.0f;
        startTimerHz(60);
    }

    void mouseDown(const juce::MouseEvent& e) override
    {
        juce::Slider::mouseDown(e);
        target = 1.0f;
        startTimerHz(60);
    }

    void mouseUp(const juce::MouseEvent& e) override
    {
        juce::Slider::mouseUp(e);
        target = isMouseOver() ? 1.0f : 0.0f;
        startTimerHz(60);
    }

private:
    void timerCallback() override
    {
        // Critically damped-ish smoothing
        const float speed = 0.22f;
        anim += (target - anim) * speed;
        if (std::abs(target - anim) < 0.01f)
        {
            anim = target;
            stopTimer();
        }
        repaint();
    }

    float anim = 0.0f;
    float target = 0.0f;
};

