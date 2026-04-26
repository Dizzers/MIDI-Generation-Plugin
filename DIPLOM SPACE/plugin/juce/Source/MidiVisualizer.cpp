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
    
    const int leftMargin = 60;
    const int topMargin = 10;
    const int bottomMargin = 24;
    const auto drawArea = getLocalBounds().reduced(0).withTrimmedTop(topMargin).withTrimmedBottom(bottomMargin).withTrimmedLeft(leftMargin);

    // Draw grid (time)
    g.setColour(juce::Colour(0x333333));
    for (int x = drawArea.getX(); x < drawArea.getRight(); x += 60) {
        g.drawVerticalLine(x, (float)drawArea.getY(), (float)drawArea.getBottom());
    }
    
    // Determine pitch range
    int minPitch = 48;
    int maxPitch = 84;
    for (const auto& n : notes)
    {
        minPitch = std::min(minPitch, n.pitch);
        maxPitch = std::max(maxPitch, n.pitch);
    }
    minPitch = juce::jlimit(0, 127, minPitch - 2);
    maxPitch = juce::jlimit(0, 127, maxPitch + 2);
    if (maxPitch <= minPitch)
        maxPitch = minPitch + 1;

    // Draw piano labels on left (octaves)
    g.setColour(juce::Colours::white);
    g.setFont(10.0f);
    for (int pitch = minPitch; pitch <= maxPitch; ++pitch)
    {
        if (pitch % 12 == 0) // C
        {
            const int octave = (pitch / 12) - 1;
            const float frac = (float)(pitch - minPitch) / (float)(maxPitch - minPitch);
            const int y = drawArea.getBottom() - (int)std::lround(frac * drawArea.getHeight());
            g.drawText("C" + juce::String(octave), 5, y - 6, leftMargin - 10, 12, juce::Justification::centredLeft);
        }
    }
    
    // Determine time range
    double maxT = 0.0;
    for (const auto& n : notes)
        maxT = std::max(maxT, n.end);
    if (maxT <= 1e-6)
        maxT = 1.0;

    // Draw MIDI notes as rectangles
    for (const auto& n : notes)
    {
        const float t0 = (float)(n.start / maxT);
        const float t1 = (float)(n.end / maxT);
        const int x0 = drawArea.getX() + (int)std::lround(t0 * (float)drawArea.getWidth());
        const int x1 = drawArea.getX() + (int)std::lround(t1 * (float)drawArea.getWidth());

        const float pFrac = (float)(n.pitch - minPitch) / (float)(maxPitch - minPitch);
        const int y = drawArea.getBottom() - (int)std::lround(pFrac * (float)drawArea.getHeight());

        const int w = std::max(2, x1 - x0);
        const int h = 10;

        const float vel = (float)juce::jlimit(1, 127, n.velocity) / 127.0f;
        auto c = juce::Colour(0x00ff80).withBrightness(0.55f + 0.45f * vel);
        g.setColour(c);
        g.fillRoundedRectangle((float)x0, (float)(y - h / 2), (float)w, (float)h, 2.5f);
    }
    
    // Status text
    g.setColour(juce::Colours::white);
    g.drawText("Piano Roll Preview", 10, getHeight() - 20, 300, 16, juce::Justification::topLeft);
}

void MidiVisualizer::setMidiMessages(const std::vector<juce::MidiMessage>& messages)
{
    midiMessages = messages;
    rebuildNotes();
    repaint();
}

void MidiVisualizer::rebuildNotes()
{
    notes.clear();
    std::unordered_map<int, std::pair<double, int>> active; // pitch -> (start, velocity)

    for (const auto& msg : midiMessages)
    {
        const double t = msg.getTimeStamp();
        if (msg.isNoteOn())
        {
            active[msg.getNoteNumber()] = {t, (int)msg.getVelocity()};
        }
        else if (msg.isNoteOff())
        {
            const int pitch = msg.getNoteNumber();
            auto it = active.find(pitch);
            if (it == active.end())
                continue;
            const double start = it->second.first;
            const int vel = it->second.second;
            const double end = std::max(start + 0.01, t);
            notes.push_back(NoteRect{pitch, start, end, vel});
            active.erase(it);
        }
    }

    // Close any hanging notes
    double lastT = 0.0;
    for (const auto& msg : midiMessages)
        lastT = std::max(lastT, msg.getTimeStamp());
    lastT = std::max(lastT, 0.1);

    for (const auto& kv : active)
        notes.push_back(NoteRect{kv.first, kv.second.first, std::max(kv.second.first + 0.05, lastT), kv.second.second});
}
