#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <vector>

class MidiPostProcessor
{
public:
    enum class QuantizeGrid
    {
        Off = 0,
        Grid1_4,
        Grid1_8,
        Grid1_16,
        Grid1_32
    };

    struct Params
    {
        float bpm = 120.0f;

        QuantizeGrid quantizeGrid = QuantizeGrid::Off;
        float quantizeAmount = 0.0f;   // 0..1

        float swingAmount = 0.0f;      // 0..1 (applied to off-beat 8ths)

        float humanizeTimeMs = 0.0f;   // 0..50
        int humanizeVelocity = 0;      // 0..30

        int velocityMin = 1;           // 1..127
        int velocityMax = 127;         // 1..127

        int seed = 42;                 // deterministic post
    };

    static std::vector<juce::MidiMessage> process(const std::vector<juce::MidiMessage>& in, const Params& p);

private:
    static double gridSeconds(QuantizeGrid grid, double bpm);
    static double quantizeTowards(double t, double grid, double amount);
    static double applySwing(double t, double bpm, double swingAmount);
};

