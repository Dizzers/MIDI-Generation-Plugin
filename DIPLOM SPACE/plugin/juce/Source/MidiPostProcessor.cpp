#include "MidiPostProcessor.h"

namespace
{
    double clampBpm(double bpm)
    {
        if (bpm < 20.0) return 20.0;
        if (bpm > 400.0) return 400.0;
        return bpm;
    }

    int clampVel(int v)
    {
        if (v < 1) return 1;
        if (v > 127) return 127;
        return v;
    }
}

double MidiPostProcessor::gridSeconds(QuantizeGrid grid, double bpm)
{
    const double q = 60.0 / clampBpm(bpm); // quarter note in seconds
    switch (grid)
    {
        case QuantizeGrid::Grid1_4:  return q;
        case QuantizeGrid::Grid1_8:  return q * 0.5;
        case QuantizeGrid::Grid1_16: return q * 0.25;
        case QuantizeGrid::Grid1_32: return q * 0.125;
        case QuantizeGrid::Off:
        default: return 0.0;
    }
}

double MidiPostProcessor::quantizeTowards(double t, double grid, double amount)
{
    if (grid <= 1e-9 || amount <= 1e-6)
        return t;
    const double q = std::round(t / grid) * grid;
    return t + (q - t) * juce::jlimit(0.0, 1.0, (double)amount);
}

double MidiPostProcessor::applySwing(double t, double bpm, double swingAmount)
{
    const double s = juce::jlimit(0.0, 1.0, (double)swingAmount);
    if (s <= 1e-6)
        return t;

    // Apply swing to off-beat 8ths: delay the “and” of each beat.
    // Beat grid is 1/8; we move odd 8th indices later by up to 50% of an 1/8 step.
    const double eighth = (60.0 / clampBpm(bpm)) * 0.5;
    if (eighth <= 1e-9)
        return t;

    const double idx = std::floor(t / eighth + 1e-9);
    const bool isOffBeat = (std::fmod(idx, 2.0) >= 1.0); // 1,3,5,...
    if (!isOffBeat)
        return t;

    const double maxDelay = eighth * 0.5; // 50% of an 1/8 step
    return t + s * maxDelay;
}

std::vector<juce::MidiMessage> MidiPostProcessor::process(const std::vector<juce::MidiMessage>& in, const Params& p)
{
    if (in.empty())
        return {};

    const double bpm = clampBpm(p.bpm);
    const double grid = gridSeconds(p.quantizeGrid, bpm);

    const int velMin = clampVel(std::min(p.velocityMin, p.velocityMax));
    const int velMax = clampVel(std::max(p.velocityMin, p.velocityMax));

    juce::Random rng((juce::int64)p.seed);

    std::vector<juce::MidiMessage> out;
    out.reserve(in.size());

    for (auto msg : in)
    {
        double t = msg.getTimeStamp();

        // Swing first (musical feel), then quantize towards grid, then humanize back a little if requested.
        t = applySwing(t, bpm, p.swingAmount);
        t = quantizeTowards(t, grid, p.quantizeAmount);

        if (p.humanizeTimeMs > 1e-6f)
        {
            const double ms = juce::jlimit(0.0, 80.0, (double)p.humanizeTimeMs);
            const double jitter = (rng.nextDouble() * 2.0 - 1.0) * (ms / 1000.0);
            t = std::max(0.0, t + jitter);
        }

        msg.setTimeStamp(t);

        if (msg.isNoteOn())
        {
            int v = (int)msg.getVelocity();
            if (p.humanizeVelocity > 0)
            {
                const int hv = juce::jlimit(0, 60, p.humanizeVelocity);
                v += rng.nextInt(hv * 2 + 1) - hv;
            }
            v = juce::jlimit(velMin, velMax, v);
            msg = juce::MidiMessage::noteOn(msg.getChannel(), msg.getNoteNumber(), (juce::uint8)clampVel(v));
            msg.setTimeStamp(t);
        }

        out.push_back(msg);
    }

    // Sort by time to keep the sequence sane after jitter.
    std::sort(out.begin(), out.end(),
              [](const juce::MidiMessage& a, const juce::MidiMessage& b) { return a.getTimeStamp() < b.getTimeStamp(); });

    return out;
}

