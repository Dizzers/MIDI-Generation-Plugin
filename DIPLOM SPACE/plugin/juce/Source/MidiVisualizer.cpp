#include "MidiVisualizer.h"

MidiVisualizer::MidiVisualizer()
{
    setSize(800, 400);
    setWantsKeyboardFocus(true);
}

MidiVisualizer::~MidiVisualizer()
{
}

void MidiVisualizer::paint(juce::Graphics& g)
{
    // Subtle depth like reference (slight vertical gradient)
    {
        auto r = getLocalBounds().toFloat();
        juce::ColourGradient bg(juce::Colour::fromRGB(18, 18, 18), r.getX(), r.getY(),
                                juce::Colour::fromRGB(10, 10, 10), r.getX(), r.getBottom(), false);
        g.setGradientFill(bg);
        g.fillAll();
    }

    const auto drawArea = getDrawArea();
    if (drawArea.isEmpty())
        return;

    const int minPitch = getMinPitch();
    const int maxPitch = getMaxPitch();
    const int pitches = std::max(1, maxPitch - minPitch + 1);

    const double maxT = getMaxTimeSeconds();
    const float contentW = (float)drawArea.getWidth() * zoomX;
    const float contentH = (float)drawArea.getHeight();
    const float pxPerSec = (maxT > 1e-6) ? (contentW / (float)maxT) : (float)drawArea.getWidth();

    auto timeToX = [&](double t) -> float { return (float)drawArea.getX() - scrollX + (float)t * pxPerSec; };
    auto pitchToY = [&](int pitch) -> float
    {
        const float frac = (float)(pitch - minPitch) / (float)(pitches - 1);
        return (float)drawArea.getBottom() - scrollY - frac * (float)contentH;
    };

    // Background rows (white/black keys)
    for (int p = minPitch; p <= maxPitch; ++p)
    {
        const int pc = p % 12;
        const bool black = (pc == 1 || pc == 3 || pc == 6 || pc == 8 || pc == 10);
        const float y = pitchToY(p);
        const float yNext = (p < maxPitch) ? pitchToY(p + 1) : (float)drawArea.getY();
        const auto row = juce::Rectangle<float>((float)drawArea.getX(), yNext, (float)drawArea.getWidth(), y - yNext);
        g.setColour(black ? juce::Colour::fromRGB(20, 20, 20) : juce::Colour::fromRGB(24, 24, 24));
        g.fillRect(row);
    }

    // Grid: time (bar/beat strength like reference)
    {
        const float safeBpm = juce::jlimit(40.0f, 200.0f, bpm);
        const float quarter = 60.0f / safeBpm;
        const float beat = quarter;       // 4/4 assumption
        const float eighth = beat * 0.5f;
        const float bar = beat * 4.0f;

        const int eighthLines = (int)std::ceil((float)maxT / eighth);
        for (int i = 0; i <= eighthLines; ++i)
        {
            const float t = i * eighth;
            const float x = timeToX(t);
            if (x < drawArea.getX() - 1 || x > drawArea.getRight() + 1)
                continue;

            const bool isBeat = (i % 2) == 0;
            const bool isBar = (std::fmod(t, bar) < 1e-4f);

            const float a = isBar ? 0.55f : (isBeat ? 0.35f : 0.18f);
            g.setColour(juce::Colour::fromRGB(90, 90, 90).withAlpha(a));
            g.drawVerticalLine((int)std::lround(x), (float)drawArea.getY(), (float)drawArea.getBottom());
        }
    }

    // Grid: octave separators
    {
        g.setColour(juce::Colour::fromRGB(255, 255, 255).withAlpha(0.08f));
        for (int p = minPitch; p <= maxPitch; ++p)
        {
            if (p % 12 != 0) // C
                continue;
            const float y = pitchToY(p);
            if (y < drawArea.getY() || y > drawArea.getBottom())
                continue;
            g.drawHorizontalLine((int)std::lround(y), (float)drawArea.getX(), (float)drawArea.getRight());
        }
    }

    // Keyboard strip (left)
    {
        const int kbW = 58;
        auto kb = getLocalBounds().withTrimmedTop(10).withTrimmedBottom(10).removeFromLeft(kbW);
        g.setColour(juce::Colours::black.withAlpha(0.25f));
        g.fillRoundedRectangle(kb.toFloat().reduced(4.0f, 0.0f), 6.0f);

        for (int p = minPitch; p <= maxPitch; ++p)
        {
            const int pc = p % 12;
            const bool black = (pc == 1 || pc == 3 || pc == 6 || pc == 8 || pc == 10);
            const float y = pitchToY(p);
            const float yNext = (p < maxPitch) ? pitchToY(p + 1) : (float)drawArea.getY();
            const float h = y - yNext;
            juce::Rectangle<float> key((float)kb.getX() + 6.0f, yNext, (float)kb.getWidth() - 12.0f, h);
            g.setColour(black ? juce::Colour::fromRGB(28, 28, 28) : juce::Colour::fromRGB(230, 230, 230).withAlpha(0.06f));
            g.fillRect(key);
            if (!black)
            {
                g.setColour(juce::Colours::white.withAlpha(0.05f));
                g.drawRect(key, 1.0f);
            }

            if (p % 12 == 0)
            {
                const int octave = (p / 12) - 1;
                g.setColour(juce::Colours::white.withAlpha(0.55f));
                g.setFont(juce::Font(juce::FontOptions(10.5f)));
                g.drawText("C" + juce::String(octave),
                           kb.getX() + 8, (int)std::lround(yNext + 2.0f), kb.getWidth() - 16, 12,
                           juce::Justification::centredLeft);
            }
        }
    }

    // Notes (reference-like green palette)
    for (const auto& n : notes)
    {
        const float x0 = timeToX(n.start);
        const float x1 = timeToX(n.end);
        if (x1 < (float)drawArea.getX() || x0 > (float)drawArea.getRight())
            continue;

        const float y = pitchToY(n.pitch);
        const float yNext = (n.pitch < maxPitch) ? pitchToY(n.pitch + 1) : y - 10.0f;
        const float rowH = std::max(8.0f, y - yNext);

        const float vel = (float)juce::jlimit(1, 127, n.velocity) / 127.0f;
        auto c = juce::Colour::fromRGB(138, 255, 116).withAlpha(0.22f + 0.62f * vel);
        auto glow = juce::Colour::fromRGB(180, 255, 170).withAlpha(0.08f + 0.22f * vel);

        const float w = std::max(2.0f, x1 - x0);
        const float h = std::min(14.0f, rowH - 2.0f);
        auto rr = juce::Rectangle<float>(x0, y - h, w, h).reduced(0.5f);

        g.setColour(glow);
        g.fillRoundedRectangle(rr.expanded(2.0f, 1.5f), 4.0f);
        g.setColour(c);
        g.fillRoundedRectangle(rr, 3.5f);
        g.setColour(juce::Colours::black.withAlpha(0.25f));
        g.drawRoundedRectangle(rr, 3.5f, 1.0f);
    }

    // Footer hint (less intrusive)
    g.setColour(juce::Colours::white.withAlpha(0.32f));
    g.setFont(juce::Font(juce::FontOptions(10.5f)));
    g.drawText("Wheel: scroll  |  Ctrl+Wheel: zoom  |  Drag: pan  |  Double-click: fit",
               10, getHeight() - 18, getWidth() - 20, 14, juce::Justification::centredLeft);
}

void MidiVisualizer::setBpm(float newBpm)
{
    bpm = newBpm;
    repaint();
}

void MidiVisualizer::resized()
{
    clampScroll();
}

void MidiVisualizer::setMidiMessages(const std::vector<juce::MidiMessage>& messages)
{
    midiMessages = messages;
    rebuildNotes();
    clampScroll();
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

double MidiVisualizer::getMaxTimeSeconds() const
{
    double maxT = 0.0;
    for (const auto& n : notes)
        maxT = std::max(maxT, n.end);
    return (maxT <= 1e-6) ? 1.0 : maxT;
}

int MidiVisualizer::getMinPitch() const
{
    int minPitch = 36;
    for (const auto& n : notes)
        minPitch = std::min(minPitch, n.pitch);
    return juce::jlimit(0, 127, minPitch - 2);
}

int MidiVisualizer::getMaxPitch() const
{
    int maxPitch = 84;
    for (const auto& n : notes)
        maxPitch = std::max(maxPitch, n.pitch);
    return juce::jlimit(0, 127, maxPitch + 2);
}

juce::Rectangle<int> MidiVisualizer::getDrawArea() const
{
    const int leftMargin = 68;
    const int topMargin = 10;
    const int bottomMargin = 24;
    return getLocalBounds()
        .withTrimmedLeft(leftMargin)
        .withTrimmedTop(topMargin)
        .withTrimmedBottom(bottomMargin);
}

void MidiVisualizer::mouseWheelMove(const juce::MouseEvent& e, const juce::MouseWheelDetails& wheel)
{
    const bool zoom = e.mods.isCtrlDown() || e.mods.isCommandDown();
    const float delta = (std::abs(wheel.deltaY) > 0.0f) ? wheel.deltaY : wheel.deltaX;

    if (zoom)
    {
        const float prev = zoomX;
        zoomX = juce::jlimit(1.0f, 12.0f, zoomX * (delta > 0.0f ? 1.12f : 0.89f));
        if (std::abs(zoomX - prev) > 1e-4f)
        {
            // Keep the mouse position stable in time-space.
            const auto drawArea = getDrawArea();
            const double maxT = getMaxTimeSeconds();
            const float contentWPrev = (float)drawArea.getWidth() * prev;
            const float contentWNew = (float)drawArea.getWidth() * zoomX;
            const float pxPerSecPrev = (maxT > 1e-6) ? (contentWPrev / (float)maxT) : (float)drawArea.getWidth();
            const float pxPerSecNew = (maxT > 1e-6) ? (contentWNew / (float)maxT) : (float)drawArea.getWidth();

            const float mouseX = (float)e.position.x;
            const float tAtMouse = (mouseX - (float)drawArea.getX() + scrollX) / pxPerSecPrev;
            scrollX = (float)juce::jmax(0.0, (double)(tAtMouse * pxPerSecNew - (mouseX - (float)drawArea.getX())));
        }
    }
    else
    {
        const float speed = 50.0f;
        if (std::abs(wheel.deltaX) > 0.0f)
            scrollX += wheel.deltaX * speed * -1.0f;
        else
            scrollY += wheel.deltaY * speed;
    }

    clampScroll();
    repaint();
}

void MidiVisualizer::mouseDown(const juce::MouseEvent& e)
{
    dragAnchor = e.getPosition();
    dragScrollAnchor = { scrollX, scrollY };
}

void MidiVisualizer::mouseDrag(const juce::MouseEvent& e)
{
    const auto delta = e.getPosition() - dragAnchor;
    scrollX = dragScrollAnchor.x - (float)delta.x;
    scrollY = dragScrollAnchor.y - (float)delta.y;
    clampScroll();
    repaint();
}

void MidiVisualizer::mouseDoubleClick(const juce::MouseEvent&)
{
    zoomX = 1.0f;
    scrollX = 0.0f;
    scrollY = 0.0f;
    clampScroll();
    repaint();
}

void MidiVisualizer::clampScroll()
{
    const auto drawArea = getDrawArea();
    const double maxT = getMaxTimeSeconds();

    const float contentW = (float)drawArea.getWidth() * zoomX;
    const float maxScrollX = std::max(0.0f, contentW - (float)drawArea.getWidth());

    scrollX = juce::jlimit(0.0f, maxScrollX, scrollX);
    scrollY = juce::jmax(0.0f, scrollY);
    (void)maxT;
}
