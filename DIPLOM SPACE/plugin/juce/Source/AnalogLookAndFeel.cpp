#include "AnalogLookAndFeel.h"
#include "AnimatedKnob.h"

namespace
{
    juce::Colour mix(juce::Colour a, juce::Colour b, float t)
    {
        return a.interpolatedWith(b, juce::jlimit(0.0f, 1.0f, t));
    }

    void fillMetal(juce::Graphics& g, juce::Rectangle<float> r, juce::Colour hi, juce::Colour lo)
    {
        juce::ColourGradient grad(hi, r.getX(), r.getY(), lo, r.getX(), r.getBottom(), false);
        grad.addColour(0.55, mix(hi, lo, 0.45f));
        g.setGradientFill(grad);
        g.fillRoundedRectangle(r, 8.0f);
    }

    void fillWood(juce::Graphics& g, juce::Rectangle<float> r, juce::Colour a, juce::Colour b)
    {
        // Lightweight “wood” suggestion: warm gradient + subtle stripes (no bitmaps).
        juce::ColourGradient grad(a, r.getX(), r.getY(), b, r.getRight(), r.getBottom(), false);
        g.setGradientFill(grad);
        g.fillRoundedRectangle(r, 8.0f);

        g.setColour(juce::Colours::black.withAlpha(0.06f));
        for (int i = 0; i < (int)r.getWidth(); i += 14)
            g.drawLine(r.getX() + (float)i, r.getY(), r.getX() + (float)i + 6.0f, r.getBottom(), 1.0f);
    }

    juce::Rectangle<float> inner(juce::Rectangle<float> r, float pad)
    {
        return r.reduced(pad);
    }
}

AnalogLookAndFeel::AnalogLookAndFeel()
{
    // Reference-like darker palette with green accent
    panelBg = juce::Colour::fromRGB(16, 16, 16);
    panelBorder = juce::Colour::fromRGB(58, 58, 58);
    metalHi = juce::Colour::fromRGB(68, 68, 70);
    metalLo = juce::Colour::fromRGB(26, 26, 27);
    woodA = juce::Colour::fromRGB(86, 54, 34);
    woodB = juce::Colour::fromRGB(52, 33, 22);
    accent = juce::Colour::fromRGB(138, 255, 116); // green accent like reference
    text = juce::Colour::fromRGB(230, 230, 230);

    setColour(juce::ResizableWindow::backgroundColourId, panelBg);
    setColour(juce::Label::textColourId, text);
    setColour(juce::ComboBox::textColourId, text);
    setColour(juce::ComboBox::backgroundColourId, panelBg.darker(0.2f));
    setColour(juce::ComboBox::outlineColourId, panelBorder);
    setColour(juce::TextButton::textColourOffId, text);
}

juce::Font AnalogLookAndFeel::getComboBoxFont(juce::ComboBox&)
{
    return juce::Font(juce::FontOptions(13.0f));
}

juce::Font AnalogLookAndFeel::getLabelFont(juce::Label&)
{
    return juce::Font(juce::FontOptions(12.5f));
}

juce::Font AnalogLookAndFeel::getTextButtonFont(juce::TextButton& b, int buttonHeight)
{
    const auto id = b.getComponentID();
    if (id == "toolbar")
        return juce::Font(juce::FontOptions((float)juce::jmin(18, buttonHeight - 6))).boldened();
    if (id == "tab")
        return juce::Font(juce::FontOptions((float)juce::jmin(13, buttonHeight - 8))).boldened();
    return juce::Font(juce::FontOptions((float)juce::jmin(14, buttonHeight - 6)));
}

void AnalogLookAndFeel::drawButtonBackground(juce::Graphics& g,
                                             juce::Button& button,
                                             const juce::Colour& /*backgroundColour*/,
                                             bool highlighted,
                                             bool down)
{
    auto r = button.getLocalBounds().toFloat().reduced(0.5f);

    const bool toggled = button.getToggleState();
    const auto id = button.getComponentID();
    const bool isTab = id == "tab";

    const float glow = (down ? 0.22f : highlighted ? 0.14f : 0.0f) + (toggled ? 0.18f : 0.0f);
    const auto metalTop = mix(metalHi, accent, glow);
    const auto metalBottom = mix(metalLo, accent.darker(0.4f), glow);

    // More reference-like: tighter corners, flatter metal
    const float cr = (id == "toolbar") ? 6.0f : 7.0f;
    juce::ColourGradient grad(metalTop, r.getX(), r.getY(), metalBottom, r.getX(), r.getBottom(), false);
    g.setGradientFill(grad);
    g.fillRoundedRectangle(r, cr);

    // Thin bevel + border
    g.setColour(juce::Colours::white.withAlpha(0.06f));
    g.drawRoundedRectangle(inner(r, 1.0f), cr - 1.0f, 1.0f);

    g.setColour(panelBorder.withAlpha(down ? 0.9f : 0.7f));
    g.drawRoundedRectangle(r, cr, 1.0f);

    if (isTab)
    {
        // Accent underline for active tab (reference-style)
        g.setColour(accent.withAlpha(toggled ? 0.85f : 0.0f));
        if (toggled)
            g.fillRoundedRectangle(r.withTrimmedTop(r.getHeight() - 3.0f).reduced(10.0f, 0.0f), 2.0f);
    }
}

void AnalogLookAndFeel::drawLinearSlider(juce::Graphics& g,
                                         int x, int y, int width, int height,
                                         float sliderPos,
                                         float /*minSliderPos*/,
                                         float /*maxSliderPos*/,
                                         const juce::Slider::SliderStyle style,
                                         juce::Slider& slider)
{
    // Render BPM top control as a pure text field (no track/thumb)
    if (slider.getComponentID() == "bpmField")
        return;

    auto r = juce::Rectangle<float>((float)x, (float)y, (float)width, (float)height);
    const bool isHorizontal = (style == juce::Slider::LinearHorizontal || style == juce::Slider::LinearBar);

    auto track = isHorizontal
        ? juce::Rectangle<float>(r.getX(), r.getCentreY() - 3.0f, r.getWidth(), 6.0f)
        : juce::Rectangle<float>(r.getCentreX() - 3.0f, r.getY(), 6.0f, r.getHeight());

    g.setColour(panelBg.brighter(0.12f));
    g.fillRoundedRectangle(track, 3.0f);
    g.setColour(panelBorder.withAlpha(0.9f));
    g.drawRoundedRectangle(track, 3.0f, 1.0f);

    auto fill = track;
    if (isHorizontal)
        fill.setWidth(juce::jlimit(0.0f, track.getWidth(), sliderPos - track.getX()));
    else
        fill.setY(sliderPos), fill.setBottom(track.getBottom());

    g.setColour(accent.withAlpha(0.85f));
    g.fillRoundedRectangle(fill, 3.0f);

    // Thumb
    const float thumbR = 7.0f;
    const float cx = isHorizontal ? sliderPos : track.getCentreX();
    const float cy = isHorizontal ? track.getCentreY() : sliderPos;
    auto thumb = juce::Rectangle<float>(cx - thumbR, cy - thumbR, thumbR * 2.0f, thumbR * 2.0f);
    fillMetal(g, thumb, metalHi.brighter(0.05f), metalLo);
    g.setColour(accent.withAlpha(slider.isMouseOverOrDragging() ? 0.35f : 0.2f));
    g.drawEllipse(thumb, 1.2f);
}

void AnalogLookAndFeel::drawRotarySlider(juce::Graphics& g,
                                         int x, int y, int width, int height,
                                         float sliderPosProportional,
                                         float rotaryStartAngle,
                                         float rotaryEndAngle,
                                         juce::Slider& slider)
{
    float anim = slider.isMouseOverOrDragging() ? 1.0f : 0.0f;
    if (auto* k = dynamic_cast<AnimatedKnob*>(&slider))
        anim = k->getAnimAmount();

    auto r = juce::Rectangle<float>((float)x, (float)y, (float)width, (float)height).reduced(6.0f);
    const float radius = juce::jmin(r.getWidth(), r.getHeight()) * 0.5f;
    const auto centre = r.getCentre();

    // Main knob body
    auto knob = juce::Rectangle<float>(centre.x - radius, centre.y - radius, radius * 2.0f, radius * 2.0f);
    fillMetal(g, knob, metalHi, metalLo);

    // Inner shadow
    g.setColour(juce::Colours::black.withAlpha(0.35f));
    g.drawEllipse(knob.reduced(1.0f), 1.0f);

    const float angle = rotaryStartAngle + sliderPosProportional * (rotaryEndAngle - rotaryStartAngle);

    // Background arc (shows range)
    {
        juce::Path bgArc;
        bgArc.addCentredArc(centre.x, centre.y, radius - 3.0f, radius - 3.0f, 0.0f,
                            rotaryStartAngle, rotaryEndAngle, true);
        g.setColour(juce::Colour::fromRGB(95, 95, 95).withAlpha(0.22f));
        g.strokePath(bgArc, juce::PathStrokeType(2.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
    }

    // Tick arc (LED)
    juce::Path arc;
    arc.addCentredArc(centre.x, centre.y, radius - 3.0f, radius - 3.0f, 0.0f, rotaryStartAngle, angle, true);
    g.setColour(accent.withAlpha(0.65f + 0.30f * anim));
    g.strokePath(arc, juce::PathStrokeType(2.2f + 0.6f * anim, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

    // Pointer (use centre-relative vectors to avoid stray lines)
    juce::Point<float> p1(0.0f, -radius + 6.0f);
    juce::Point<float> p2(0.0f, -radius + 16.0f);
    p1 = p1.rotatedAboutOrigin(angle).translated(centre.x, centre.y);
    p2 = p2.rotatedAboutOrigin(angle).translated(centre.x, centre.y);
    g.setColour(text.withAlpha(0.9f));
    g.drawLine({p1, p2}, 2.0f);

    // Subtle indicator dot at the tip (avoid “wonky” feel)
    {
        juce::Point<float> dot(0.0f, -radius + 6.0f);
        dot = dot.rotatedAboutOrigin(angle).translated(centre.x, centre.y);
        g.setColour(accent.withAlpha(0.55f + 0.25f * anim));
        g.fillEllipse(dot.x - 1.9f, dot.y - 1.9f, 3.8f, 3.8f);
    }

    // Soft glow when hovered/dragged (keep subtle)
    if (anim > 0.001f)
    {
        g.setColour(accent.withAlpha(0.10f * anim));
        g.fillEllipse(knob.expanded(4.0f + 6.0f * anim));
    }

    // Wood cap (subtle) for “analog” feel
    auto cap = knob.reduced(radius * 0.45f);
    fillWood(g, cap, woodA.withAlpha(0.85f), woodB.withAlpha(0.85f));
    g.setColour(panelBorder.withAlpha(0.8f));
    g.drawRoundedRectangle(cap, 8.0f, 1.0f);
}

void AnalogLookAndFeel::drawComboBox(juce::Graphics& g,
                                     int width, int height,
                                     bool isButtonDown,
                                     int /*buttonX*/, int /*buttonY*/, int /*buttonW*/, int /*buttonH*/,
                                     juce::ComboBox& box)
{
    (void)box;
    auto r = juce::Rectangle<float>(0, 0, (float)width, (float)height).reduced(0.5f);
    fillMetal(g, r, metalHi.darker(isButtonDown ? 0.2f : 0.0f), metalLo);
    g.setColour(panelBorder.withAlpha(0.8f));
    g.drawRoundedRectangle(r, 7.0f, 1.0f);

    // dropdown chevron
    const float cx = r.getRight() - 14.0f;
    const float cy = r.getCentreY();
    juce::Path p;
    p.startNewSubPath(cx - 5.0f, cy - 2.0f);
    p.lineTo(cx, cy + 3.0f);
    p.lineTo(cx + 5.0f, cy - 2.0f);
    g.setColour(text.withAlpha(0.8f));
    g.strokePath(p, juce::PathStrokeType(1.6f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
}

