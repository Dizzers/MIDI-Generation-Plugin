#include "PluginEditor.h"
#include "PluginProcessor.h"
#include "MidiVisualizer.h"
#include "MidiFileExporter.h"
// AnalogLookAndFeel is included via PluginEditor.h
#include "AnimatedKnob.h"

class PluginEditor::ControlsPanel final : public juce::Component
{
public:
    ControlsPanel() = default;

    void paint(juce::Graphics& g) override
    {
        // Subtle “rack panel” behind controls
        g.fillAll(juce::Colours::transparentBlack);

        auto r = getLocalBounds().toFloat();
        auto panel = r.reduced(4.0f);

        juce::Colour hi = juce::Colour::fromRGB(30, 30, 31);
        juce::Colour lo = juce::Colour::fromRGB(16, 16, 16);
        juce::ColourGradient grad(hi, panel.getX(), panel.getY(), lo, panel.getX(), panel.getBottom(), false);
        g.setGradientFill(grad);
        g.fillRoundedRectangle(panel, 10.0f);

        g.setColour(juce::Colour::fromRGB(70, 70, 70).withAlpha(0.65f));
        g.drawRoundedRectangle(panel, 10.0f, 1.0f);
    }
};

class PluginEditor::ToolbarBlock final : public juce::Component
{
public:
    explicit ToolbarBlock(juce::String t = {}) : title(std::move(t)) {}

    void setTitle(juce::String t)
    {
        title = std::move(t);
        repaint();
    }

    void paint(juce::Graphics& g) override
    {
        auto r = getLocalBounds().toFloat().reduced(0.5f);
        juce::Colour hi = juce::Colour::fromRGB(34, 34, 34);
        juce::Colour lo = juce::Colour::fromRGB(16, 16, 16);
        juce::ColourGradient grad(hi, r.getX(), r.getY(), lo, r.getX(), r.getBottom(), false);
        g.setGradientFill(grad);
        g.fillRoundedRectangle(r, 8.0f);

        g.setColour(juce::Colour::fromRGB(255, 255, 255).withAlpha(0.05f));
        g.drawRoundedRectangle(r.reduced(1.0f), 7.0f, 1.0f);
        g.setColour(juce::Colour::fromRGB(80, 80, 80).withAlpha(0.70f));
        g.drawRoundedRectangle(r, 8.0f, 1.0f);

        if (title.isNotEmpty())
        {
            auto a = getLocalBounds().removeFromTop(14).reduced(10, 0);
            g.setColour(juce::Colour::fromRGB(235, 235, 235).withAlpha(0.62f));
            g.setFont(juce::Font(juce::FontOptions(10.0f)).boldened());
            g.drawText(title, a, juce::Justification::centredLeft);
        }
    }

private:
    juce::String title;
};

class PluginEditor::Divider final : public juce::Component
{
public:
    void paint(juce::Graphics& g) override
    {
        auto r = getLocalBounds().toFloat();
        g.setColour(juce::Colours::white.withAlpha(0.06f));
        g.drawLine(r.getX(), r.getCentreY(), r.getRight(), r.getCentreY(), 1.0f);
        g.setColour(juce::Colours::black.withAlpha(0.25f));
        g.drawLine(r.getX(), r.getCentreY() + 1.0f, r.getRight(), r.getCentreY() + 1.0f, 1.0f);
    }
};

class PluginEditor::GroupBlock final : public juce::Component
{
public:
    explicit GroupBlock(juce::String t) : title(std::move(t)) {}

    void setTitle(juce::String t)
    {
        title = std::move(t);
        repaint();
    }

    void paint(juce::Graphics& g) override
    {
        auto r = getLocalBounds().toFloat().reduced(1.0f);

        // Metal panel fill
        juce::Colour hi = juce::Colour::fromRGB(32, 32, 33);
        juce::Colour lo = juce::Colour::fromRGB(14, 14, 14);
        juce::ColourGradient grad(hi, r.getX(), r.getY(), lo, r.getX(), r.getBottom(), false);
        g.setGradientFill(grad);
        g.fillRoundedRectangle(r, 10.0f);

        // Inner top highlight + outer border (Instacomposer-like)
        g.setColour(juce::Colour::fromRGB(255, 255, 255).withAlpha(0.04f));
        g.drawRoundedRectangle(r.reduced(1.0f), 9.0f, 1.0f);
        g.setColour(juce::Colour::fromRGB(78, 78, 78).withAlpha(0.70f));
        g.drawRoundedRectangle(r, 10.0f, 1.0f);

        // Title
        if (title.isNotEmpty())
        {
            auto titleArea = getLocalBounds().removeFromTop(16).reduced(10, 0);
            g.setColour(juce::Colour::fromRGB(235, 235, 235).withAlpha(0.62f));
            g.setFont(juce::Font(juce::FontOptions(10.5f)).boldened());
            g.drawText(title, titleArea, juce::Justification::centredLeft);
        }
    }

private:
    juce::String title;
};

PluginEditor::Divider& PluginEditor::addDivider()
{
    auto d = std::make_unique<Divider>();
    auto& ref = *d;
    dividers.push_back(std::move(d));
    controlsPanel->addAndMakeVisible(ref);
    ref.setInterceptsMouseClicks(false, false);
    ref.toBack();
    return ref;
}

PluginEditor::GroupBlock& PluginEditor::addGroupBlock(const juce::String& title)
{
    auto b = std::make_unique<GroupBlock>(title);
    auto& ref = *b;
    groupBlocks.push_back(std::move(b));
    controlsPanel->addAndMakeVisible(ref);
    ref.setInterceptsMouseClicks(false, false);
    ref.toBack();
    return ref;
}

juce::Label& PluginEditor::addCaption(juce::Component& target, const juce::String& text)
{
    auto l = std::make_unique<juce::Label>();
    l->setText(text, juce::dontSendNotification);
    l->setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 235, 235).withAlpha(0.72f));
    l->setJustificationType(juce::Justification::centredLeft);
    l->setInterceptsMouseClicks(false, false);
    l->setFont(juce::Font(juce::FontOptions(11.0f)));

    auto& ref = *l;
    captions.push_back(std::move(l));
    controlsPanel->addAndMakeVisible(ref);
    captionByComponent[&target] = &ref;
    return ref;
}

void PluginEditor::applyPageVisibility()
{
    const auto showGen = currentPage == Page::Gen;
    const auto showFeel = currentPage == Page::Feel;
    const auto showPost = currentPage == Page::Post;

    auto set = [](juce::Component* c, bool v) { if (c) c->setVisible(v); };

    // GEN
    set(keyBox.get(), showGen);
    set(maxLenSlider.get(), showGen);
    set(targetDurationSlider.get(), showGen);
    set(seedSlider.get(), showGen);
    set(temperatureKnob.get(), showGen);
    set(repetitionPenaltyKnob.get(), showGen);
    set(topKSlider.get(), showGen);
    set(topPSlider.get(), showGen);
    set(ngramSlider.get(), showGen);
    set(harmonyModeBox.get(), showGen);
    set(harmonyBiasSlider.get(), showGen);
    set(primerModeBox.get(), showGen);
    set(primerLenSlider.get(), showGen);

    // FEEL
    set(velocityFeelKnob.get(), showFeel);
    set(grooveFeelKnob.get(), showFeel);
    set(maxPolyphonySlider.get(), showFeel);
    set(minBodyTokensSlider.get(), showFeel);

    // POST (BPM is shown in the top bar only)
    set(bpmSlider.get(), false);
    set(quantizeGridBox.get(), showPost);
    set(quantizeAmountSlider.get(), showPost);
    set(swingAmountSlider.get(), showPost);
    set(humanizeTimeSlider.get(), showPost);
    set(humanizeVelocitySlider.get(), showPost);
    set(velocityMinSlider.get(), showPost);
    set(velocityMaxSlider.get(), showPost);

    // Always visible controls (toolbar/status)
    set(generateButton.get(), true);
    set(randomizeButton.get(), true);
    set(regenButton.get(), true);
    set(exportButton.get(), true);
    set(dragDropButton.get(), true);
    set(statusLabel.get(), true);
    set(generationProgressLabel.get(), true);
    set(bpmTopEditor.get(), true);
    set(bpmTopLabel.get(), true);
    set(actionsBlock.get(), true);
    set(bpmBlock.get(), true);
    set(goLabel.get(), true);
    set(regenLabel.get(), true);
    set(exportLabel.get(), true);
    set(dragLabel.get(), true);

    // Captions: default hide all, we will show a subset in resized() for current page.
    for (auto& c : captions)
        if (c) c->setVisible(false);

    for (auto& d : dividers)
        if (d) d->setVisible(false);

    // Blocks: we toggle in resized() per-page; default visible to avoid flicker
    for (auto& b : groupBlocks)
        if (b) b->setVisible(true);

    // Enable captions for visible components
    for (const auto& kv : captionByComponent)
        if (kv.second)
            kv.second->setVisible(kv.first && kv.first->isVisible());

    repaint();
    resized();
}

//==============================================================================
PluginEditor::PluginEditor(PluginProcessor& p)
    : AudioProcessorEditor(&p), processor(p), apvts(p.getValueTreeState())
{
    setLookAndFeel(&analogLookAndFeel);
    bpmParam = apvts.getParameter("bpm");
    createControls();

    // setSize triggers resized(); keep it after controls exist
    setSize(1280, 820);
    setResizable(false, false);

    startTimer(100);  // Update status every 100ms
    
    // Initial status
    statusLabel->setText(processor.isModelReady() ? "Ready" : processor.getModelStatusText(),
                         juce::dontSendNotification);
}

PluginEditor::~PluginEditor()
{
    stopTimer();
    setLookAndFeel(nullptr);
}

//==============================================================================
void PluginEditor::createControls()
{
    controlsPanel = std::make_unique<ControlsPanel>();
    controlsViewport = std::make_unique<juce::Viewport>();
    controlsViewport->setViewedComponent(controlsPanel.get(), false);
    controlsViewport->setScrollBarsShown(true, false);
    addAndMakeVisible(*controlsViewport);

    // Top toolbar blocks (drawn behind controls/buttons)
    actionsBlock = std::make_unique<ToolbarBlock>("ACTIONS");
    actionsBlock->setInterceptsMouseClicks(false, false);
    addAndMakeVisible(*actionsBlock);

    bpmBlock = std::make_unique<ToolbarBlock>("BPM");
    bpmBlock->setInterceptsMouseClicks(false, false);
    addAndMakeVisible(*bpmBlock);

    bpmTopLabel = std::make_unique<juce::Label>("bpmTopLabel", "BPM");
    bpmTopLabel->setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 235, 235).withAlpha(0.80f));
    bpmTopLabel->setFont(juce::Font(juce::FontOptions(10.5f)).boldened());
    bpmTopLabel->setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(*bpmTopLabel);

    bpmTopEditor = std::make_unique<juce::TextEditor>("bpmTopEditor");
    bpmTopEditor->setInputRestrictions(0, "0123456789.");
    bpmTopEditor->setJustification(juce::Justification::centredLeft);
    bpmTopEditor->setColour(juce::TextEditor::backgroundColourId, juce::Colours::black.withAlpha(0.18f));
    bpmTopEditor->setColour(juce::TextEditor::outlineColourId, juce::Colour::fromRGB(85, 85, 85).withAlpha(0.65f));
    bpmTopEditor->setColour(juce::TextEditor::focusedOutlineColourId, juce::Colour::fromRGB(138, 255, 116).withAlpha(0.70f));
    bpmTopEditor->setColour(juce::TextEditor::textColourId, juce::Colours::white.withAlpha(0.92f));
    bpmTopEditor->setFont(juce::Font(juce::FontOptions(12.5f)));
    bpmTopEditor->setTextToShowWhenEmpty("120", juce::Colours::white.withAlpha(0.22f));
    bpmTopEditor->onReturnKey = [this]() { commitBpmFromText(); };
    bpmTopEditor->onFocusLost = [this]() { commitBpmFromText(); };
    addAndMakeVisible(*bpmTopEditor);

    auto makeActionLabel = [&](const juce::String& t) -> std::unique_ptr<juce::Label>
    {
        auto l = std::make_unique<juce::Label>();
        l->setText(t, juce::dontSendNotification);
        l->setInterceptsMouseClicks(false, false);
        l->setJustificationType(juce::Justification::centred);
        l->setColour(juce::Label::textColourId, juce::Colour::fromRGB(235, 235, 235).withAlpha(0.55f));
        l->setFont(juce::Font(juce::FontOptions(9.5f)).boldened());
        addAndMakeVisible(*l);
        return l;
    };
    goLabel = makeActionLabel("GO");
    regenLabel = makeActionLabel("REGEN");
    exportLabel = makeActionLabel("EXPORT");
    dragLabel = makeActionLabel("DRAG");

    // Page buttons (like reference plugin tabs)
    auto makePageBtn = [&](const juce::String& txt) -> std::unique_ptr<juce::TextButton>
    {
        auto b = std::make_unique<juce::TextButton>(txt);
        b->setClickingTogglesState(true);
        b->setColour(juce::TextButton::textColourOffId, juce::Colour::fromRGB(235, 235, 235).withAlpha(0.85f));
        controlsPanel->addAndMakeVisible(*b);
        return b;
    };

    pageGenButton = makePageBtn("GEN");
    pageFeelButton = makePageBtn("FEEL");
    pagePostButton = makePageBtn("POST");

    pageGenButton->setComponentID("tab");
    pageFeelButton->setComponentID("tab");
    pagePostButton->setComponentID("tab");

    pageGenButton->setToggleState(true, juce::dontSendNotification);
    pageGenButton->onClick = [this]()
    {
        currentPage = Page::Gen;
        pageGenButton->setToggleState(true, juce::dontSendNotification);
        pageFeelButton->setToggleState(false, juce::dontSendNotification);
        pagePostButton->setToggleState(false, juce::dontSendNotification);
        applyPageVisibility();
    };
    pageFeelButton->onClick = [this]()
    {
        currentPage = Page::Feel;
        pageGenButton->setToggleState(false, juce::dontSendNotification);
        pageFeelButton->setToggleState(true, juce::dontSendNotification);
        pagePostButton->setToggleState(false, juce::dontSendNotification);
        applyPageVisibility();
    };
    pagePostButton->onClick = [this]()
    {
        currentPage = Page::Post;
        pageGenButton->setToggleState(false, juce::dontSendNotification);
        pageFeelButton->setToggleState(false, juce::dontSendNotification);
        pagePostButton->setToggleState(true, juce::dontSendNotification);
        applyPageVisibility();
    };

    // Dividers (we place/enable them per-page in resized)
    addDivider();
    addDivider();
    addDivider();

    // Bottom strip group blocks (we position + retitle per page)
    addGroupBlock("GENERAL");
    addGroupBlock("SAMPLING");
    addGroupBlock("HARMONY");

    // === BASIC SETTINGS PANEL ===
    // Key ComboBox
    keyBox = std::make_unique<juce::ComboBox>();
    auto keyNames = juce::StringArray{
        "C Major", "G Major", "D Major", "A Major", "E Major", "B Major",
        "F# Major", "C# Major",
        "F Major", "Bb Major", "Eb Major", "Ab Major",
        "A Minor", "E Minor", "B Minor", "F# Minor", "C# Minor",
        "G# Minor", "D# Minor", "A# Minor",
        "D Minor", "G Minor", "C Minor", "F Minor"
    };
    for (int i = 0; i < keyNames.size(); ++i)
        keyBox->addItem(keyNames[i], i + 1);
    keyBox->setSelectedItemIndex(0);
    controlsPanel->addAndMakeVisible(*keyBox);
    addCaption(*keyBox, "KEY");
    comboAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        apvts, "key", *keyBox));

    // Max Length
    maxLenSlider = std::make_unique<AnimatedKnob>();
    maxLenSlider->setRange(64.0, 512.0, 1.0);
    maxLenSlider->setTextValueSuffix(" tk");
    controlsPanel->addAndMakeVisible(*maxLenSlider);
    addCaption(*maxLenSlider, "MAX LENGTH");
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "maxLen", *maxLenSlider));

    // Target Duration
    targetDurationSlider = std::make_unique<AnimatedKnob>();
    targetDurationSlider->setRange(1.0, 5.0, 0.1);
    targetDurationSlider->setTextValueSuffix(" s");
    controlsPanel->addAndMakeVisible(*targetDurationSlider);
    addCaption(*targetDurationSlider, "TARGET DURATION");
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "targetSeconds", *targetDurationSlider));

    seedSlider = std::make_unique<AnimatedKnob>();
    seedSlider->setRange(0.0, 999999.0, 1.0);
    seedSlider->setTextValueSuffix("");
    controlsPanel->addAndMakeVisible(*seedSlider);
    addCaption(*seedSlider, "SEED");
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "seed", *seedSlider));

    // === ROTARY KNOBS (Main Parameters) ===
    temperatureKnob = std::make_unique<AnimatedKnob>();
    temperatureKnob->setRange(0.1, 2.0, 0.01);
    temperatureKnob->setTextValueSuffix("");
    controlsPanel->addAndMakeVisible(*temperatureKnob);
    addCaption(*temperatureKnob, "TEMP");
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "temperature", *temperatureKnob));

    melodyLeapKnob = std::make_unique<AnimatedKnob>();
    melodyLeapKnob->setRange(3.0, 24.0, 1.0);
    melodyLeapKnob->setTextValueSuffix(" st");
    controlsPanel->addAndMakeVisible(*melodyLeapKnob);
    addCaption(*melodyLeapKnob, "LEAP");
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "maxMelodyLeap", *melodyLeapKnob));

    repetitionPenaltyKnob = std::make_unique<AnimatedKnob>();
    repetitionPenaltyKnob->setRange(1.0, 2.0, 0.01);
    repetitionPenaltyKnob->setTextValueSuffix("");
    controlsPanel->addAndMakeVisible(*repetitionPenaltyKnob);
    addCaption(*repetitionPenaltyKnob, "REPEAT");
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "repetitionPenalty", *repetitionPenaltyKnob));

    velocityFeelKnob = std::make_unique<AnimatedKnob>();
    velocityFeelKnob->setRange(-1.0, 1.0, 0.01);
    velocityFeelKnob->setTextValueSuffix("");
    velocityFeelKnob->setSkewFactor(1.0);
    controlsPanel->addAndMakeVisible(*velocityFeelKnob);
    addCaption(*velocityFeelKnob, "VEL FEEL");
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "velocityFeel", *velocityFeelKnob));

    grooveFeelKnob = std::make_unique<AnimatedKnob>();
    grooveFeelKnob->setRange(-1.0, 1.0, 0.01);
    grooveFeelKnob->setTextValueSuffix("");
    controlsPanel->addAndMakeVisible(*grooveFeelKnob);
    addCaption(*grooveFeelKnob, "GROOVE");
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "grooveFeel", *grooveFeelKnob));

    // === SAMPLING CONTROL SLIDERS ===
    topKSlider = std::make_unique<AnimatedKnob>();
    topKSlider->setRange(1.0, 50.0, 1.0);
    topKSlider->setTextValueSuffix("");
    controlsPanel->addAndMakeVisible(*topKSlider);
    addCaption(*topKSlider, "TOP-K");
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "topK", *topKSlider));

    topPSlider = std::make_unique<AnimatedKnob>();
    topPSlider->setRange(0.5, 1.0, 0.01);
    topPSlider->setTextValueSuffix("");
    controlsPanel->addAndMakeVisible(*topPSlider);
    addCaption(*topPSlider, "TOP-P");
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "topP", *topPSlider));

    ngramSlider = std::make_unique<AnimatedKnob>();
    ngramSlider->setRange(2.0, 8.0, 1.0);
    ngramSlider->setTextValueSuffix("");
    controlsPanel->addAndMakeVisible(*ngramSlider);
    addCaption(*ngramSlider, "NO-REPEAT NGRAM");
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "noRepeatNgramSize", *ngramSlider));

    // === MUSICAL CONSTRAINTS ===
    harmonyModeBox = std::make_unique<juce::ComboBox>();
    harmonyModeBox->addItem("None", 1);
    harmonyModeBox->addItem("Weak", 2);
    harmonyModeBox->addItem("Strong", 3);
    harmonyModeBox->setSelectedItemIndex(1);  // Default: Weak
    controlsPanel->addAndMakeVisible(*harmonyModeBox);
    addCaption(*harmonyModeBox, "HARMONY MODE");
    comboAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        apvts, "harmonyMode", *harmonyModeBox));

    harmonyBiasSlider = std::make_unique<AnimatedKnob>();
    harmonyBiasSlider->setRange(0.0, 1.0, 0.01);
    harmonyBiasSlider->setTextValueSuffix("");
    controlsPanel->addAndMakeVisible(*harmonyBiasSlider);
    addCaption(*harmonyBiasSlider, "HARMONY BIAS");
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "harmonyBias", *harmonyBiasSlider));

    primerModeBox = std::make_unique<juce::ComboBox>();
    primerModeBox->addItem("None", 1);
    primerModeBox->addItem("Dataset", 2);
    primerModeBox->setSelectedItemIndex(1);  // Default: Dataset
    controlsPanel->addAndMakeVisible(*primerModeBox);
    addCaption(*primerModeBox, "PRIMER");
    comboAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        apvts, "primerMode", *primerModeBox));

    primerLenSlider = std::make_unique<AnimatedKnob>();
    primerLenSlider->setRange(8.0, 32.0, 1.0);
    primerLenSlider->setTextValueSuffix("");
    controlsPanel->addAndMakeVisible(*primerLenSlider);
    addCaption(*primerLenSlider, "PRIMER LEN");
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "primerLen", *primerLenSlider));

    maxPolyphonySlider = std::make_unique<AnimatedKnob>();
    maxPolyphonySlider->setRange(1.0, 16.0, 1.0);
    maxPolyphonySlider->setTextValueSuffix("");
    controlsPanel->addAndMakeVisible(*maxPolyphonySlider);
    addCaption(*maxPolyphonySlider, "MAX POLYPHONY");
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "maxPolyphony", *maxPolyphonySlider));

    minBodyTokensSlider = std::make_unique<AnimatedKnob>();
    minBodyTokensSlider->setRange(0.0, 256.0, 1.0);
    minBodyTokensSlider->setTextValueSuffix("");
    controlsPanel->addAndMakeVisible(*minBodyTokensSlider);
    addCaption(*minBodyTokensSlider, "MIN BODY TOKENS");
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "minBodyTokens", *minBodyTokensSlider));

    // === PERFORMANCE (post) ===
    bpmSlider = std::make_unique<AnimatedKnob>();
    bpmSlider->setRange(40.0, 200.0, 1.0);
    bpmSlider->setTextValueSuffix(" bpm");
    controlsPanel->addAndMakeVisible(*bpmSlider);
    addCaption(*bpmSlider, "BPM");
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "bpm", *bpmSlider));

    quantizeGridBox = std::make_unique<juce::ComboBox>();
    quantizeGridBox->addItem("Off", 1);
    quantizeGridBox->addItem("1/4", 2);
    quantizeGridBox->addItem("1/8", 3);
    quantizeGridBox->addItem("1/16", 4);
    quantizeGridBox->addItem("1/32", 5);
    quantizeGridBox->setSelectedItemIndex(0);
    controlsPanel->addAndMakeVisible(*quantizeGridBox);
    addCaption(*quantizeGridBox, "QUANTIZE GRID");
    comboAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        apvts, "quantizeGrid", *quantizeGridBox));

    quantizeAmountSlider = std::make_unique<AnimatedKnob>();
    quantizeAmountSlider->setRange(0.0, 1.0, 0.01);
    controlsPanel->addAndMakeVisible(*quantizeAmountSlider);
    addCaption(*quantizeAmountSlider, "QUANTIZE AMOUNT");
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "quantizeAmount", *quantizeAmountSlider));

    swingAmountSlider = std::make_unique<AnimatedKnob>();
    swingAmountSlider->setRange(0.0, 1.0, 0.01);
    controlsPanel->addAndMakeVisible(*swingAmountSlider);
    addCaption(*swingAmountSlider, "SWING");
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "swingAmount", *swingAmountSlider));

    humanizeTimeSlider = std::make_unique<AnimatedKnob>();
    humanizeTimeSlider->setRange(0.0, 50.0, 1.0);
    humanizeTimeSlider->setTextValueSuffix(" ms");
    controlsPanel->addAndMakeVisible(*humanizeTimeSlider);
    addCaption(*humanizeTimeSlider, "HUMANIZE TIME");
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "humanizeTimeMs", *humanizeTimeSlider));

    humanizeVelocitySlider = std::make_unique<AnimatedKnob>();
    humanizeVelocitySlider->setRange(0.0, 30.0, 1.0);
    controlsPanel->addAndMakeVisible(*humanizeVelocitySlider);
    addCaption(*humanizeVelocitySlider, "HUMANIZE VEL");
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "humanizeVelocity", *humanizeVelocitySlider));

    velocityMinSlider = std::make_unique<AnimatedKnob>();
    velocityMinSlider->setRange(1.0, 127.0, 1.0);
    controlsPanel->addAndMakeVisible(*velocityMinSlider);
    addCaption(*velocityMinSlider, "VELOCITY MIN");
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "velocityMin", *velocityMinSlider));

    velocityMaxSlider = std::make_unique<AnimatedKnob>();
    velocityMaxSlider->setRange(1.0, 127.0, 1.0);
    controlsPanel->addAndMakeVisible(*velocityMaxSlider);
    addCaption(*velocityMaxSlider, "VELOCITY MAX");
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "velocityMax", *velocityMaxSlider));

    // === BUTTONS ===
    generateButton = std::make_unique<juce::TextButton>("GENERATE");
    generateButton->setColour(juce::TextButton::buttonColourId, juce::Colour(0x00FF80));  // Neon green
    generateButton->setColour(juce::TextButton::textColourOffId, juce::Colours::black);
    generateButton->onClick = [this]() {
        if (!processor.isGenerating()) {
            if (!processor.isModelReady())
            {
                statusLabel->setText(processor.getModelStatusText(), juce::dontSendNotification);
                return;
            }
            PluginProcessor::GenerationParams params;
            params.key = keyBox->getText().toStdString();
            params.seed = (int)*apvts.getRawParameterValue("seed");
            params.temperature = *apvts.getRawParameterValue("temperature");
            params.topK = static_cast<int>(*apvts.getRawParameterValue("topK"));
            params.topP = *apvts.getRawParameterValue("topP");
            params.repetitionPenalty = *apvts.getRawParameterValue("repetitionPenalty");
            params.noRepeatNgramSize = static_cast<int>(*apvts.getRawParameterValue("noRepeatNgramSize"));
            params.maxMelodyLeap = static_cast<int>(*apvts.getRawParameterValue("maxMelodyLeap"));
            params.harmonyMode = harmonyModeBox->getText().toStdString();
            params.harmonyBias = *apvts.getRawParameterValue("harmonyBias");
            params.maxLen = static_cast<int>(*apvts.getRawParameterValue("maxLen"));
            params.targetSeconds = *apvts.getRawParameterValue("targetSeconds");
            params.primerMode = primerModeBox->getText().toStdString();
            params.primerLen = static_cast<int>(*apvts.getRawParameterValue("primerLen"));
            params.velocityFeel = *apvts.getRawParameterValue("velocityFeel");
            params.grooveFeel = *apvts.getRawParameterValue("grooveFeel");
            params.maxPolyphony = static_cast<int>(*apvts.getRawParameterValue("maxPolyphony"));
            params.minBodyTokens = static_cast<int>(*apvts.getRawParameterValue("minBodyTokens"));
            params.bpm = *apvts.getRawParameterValue("bpm");
            params.quantizeGrid = static_cast<int>(*apvts.getRawParameterValue("quantizeGrid"));
            params.quantizeAmount = *apvts.getRawParameterValue("quantizeAmount");
            params.swingAmount = *apvts.getRawParameterValue("swingAmount");
            params.humanizeTimeMs = *apvts.getRawParameterValue("humanizeTimeMs");
            params.humanizeVelocity = static_cast<int>(*apvts.getRawParameterValue("humanizeVelocity"));
            params.velocityMin = static_cast<int>(*apvts.getRawParameterValue("velocityMin"));
            params.velocityMax = static_cast<int>(*apvts.getRawParameterValue("velocityMax"));
            
            processor.startGeneration(params);
            statusLabel->setText("Generating...", juce::dontSendNotification);
        }
    };
    generateButton->setButtonText("GO");
    generateButton->setTooltip("Generate");
    generateButton->setComponentID("toolbarPrimary");
    addAndMakeVisible(*generateButton);

    randomizeButton = std::make_unique<juce::TextButton>("RANDOMIZE");
    randomizeButton->setColour(juce::TextButton::buttonColourId, juce::Colour(0x00D4FF));  // Neon cyan
    randomizeButton->setColour(juce::TextButton::textColourOffId, juce::Colours::black);
    randomizeButton->onClick = [this]() {
        // Randomize parameters
        temperatureKnob->setValue(0.5 + juce::Random::getSystemRandom().nextFloat() * 1.5);
        topKSlider->setValue(5.0 + juce::Random::getSystemRandom().nextFloat() * 20.0);
        melodyLeapKnob->setValue(6.0 + juce::Random::getSystemRandom().nextFloat() * 12.0);
    };
    randomizeButton->setButtonText(juce::String::fromUTF8("✦"));
    randomizeButton->setTooltip("Randomize");
    randomizeButton->setComponentID("toolbar");
    addAndMakeVisible(*randomizeButton);

    regenButton = std::make_unique<juce::TextButton>("RE-GENERATE");
    regenButton->setColour(juce::TextButton::buttonColourId, juce::Colour(0x00D4FF));
    regenButton->setColour(juce::TextButton::textColourOffId, juce::Colours::black);
    regenButton->onClick = [this]() { processor.regenerateLast(); };
    regenButton->setButtonText(juce::String::fromUTF8("↻"));
    regenButton->setTooltip("Regenerate");
    regenButton->setComponentID("toolbar");
    addAndMakeVisible(*regenButton);

    exportButton = std::make_unique<juce::TextButton>("EXPORT .MID");
    exportButton->setColour(juce::TextButton::buttonColourId, juce::Colour(0x00ff80));
    exportButton->setColour(juce::TextButton::textColourOffId, juce::Colours::black);
    exportButton->onClick = [this]()
    {
        if (lastMidiMessages.empty())
            return;

        auto chooser = std::make_shared<juce::FileChooser>(
            "Export MIDI",
            juce::File::getSpecialLocation(juce::File::userDocumentsDirectory),
            "*.mid");

        chooser->launchAsync(juce::FileBrowserComponent::saveMode,
                             [chooser, this](const juce::FileChooser& fc)
                             {
                                 auto file = fc.getResult();
                                 if (file == juce::File())
                                     return;
                                 file = file.withFileExtension(".mid");
                                 auto messagesCopy = lastMidiMessages;
                                 DBG(MidiFileExporter::saveMidiFile(file, messagesCopy, 120.0)
                                         ? "MIDI exported OK"
                                         : "MIDI export failed");
                             });
    };
    exportButton->setButtonText(juce::String::fromUTF8("⤓"));
    exportButton->setTooltip("Export MIDI");
    exportButton->setComponentID("toolbar");
    addAndMakeVisible(*exportButton);

    dragDropButton = std::make_unique<juce::TextButton>("DRAG&DROP");
    dragDropButton->setColour(juce::TextButton::buttonColourId, juce::Colour(0x666666));
    dragDropButton->setColour(juce::TextButton::textColourOffId, juce::Colours::white);
    dragDropButton->onClick = [this]()
    {
        if (lastMidiMessages.empty())
            return;
        auto tmp = juce::File::getSpecialLocation(juce::File::tempDirectory)
                       .getChildFile("juce_midi_gen_drag.mid");
        if (!MidiFileExporter::saveMidiFile(tmp, lastMidiMessages, 120.0))
            return;

        juce::StringArray files;
        files.add(tmp.getFullPathName());
        performExternalDragDropOfFiles(files, true, nullptr);
    };
    dragDropButton->setButtonText(juce::String::fromUTF8("⇪"));
    dragDropButton->setTooltip("Drag & drop MIDI");
    dragDropButton->setComponentID("toolbar");
    addAndMakeVisible(*dragDropButton);

    // === STATUS DISPLAY ===
    statusLabel = std::make_unique<juce::Label>("status", "Ready");
    statusLabel->setJustificationType(juce::Justification::centred);
    statusLabel->setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(*statusLabel);

    generationProgressLabel = std::make_unique<juce::Label>("progress", "");
    generationProgressLabel->setJustificationType(juce::Justification::centred);
    generationProgressLabel->setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(*generationProgressLabel);

    midiInfoLabel = std::make_unique<juce::Label>("midiInfo", "No clip yet");
    midiInfoLabel->setColour(juce::Label::textColourId, juce::Colours::white);
    midiInfoLabel->setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(*midiInfoLabel);

    midiVisualizer = std::make_unique<MidiVisualizer>();
    addAndMakeVisible(*midiVisualizer);

    applyPageVisibility();
}

//==============================================================================
void PluginEditor::paint(juce::Graphics& g)
{
    // Main background (reference-like)
    juce::ColourGradient grad(juce::Colour::fromRGB(22, 22, 22), 0, 0,
                              juce::Colour::fromRGB(10, 10, 10), 0, (float)getHeight(), false);
    g.setGradientFill(grad);
    g.fillAll();

    // Top header bar (flat, like reference)
    auto strip = getLocalBounds().removeFromTop(56).toFloat();
    juce::ColourGradient head(juce::Colour::fromRGB(28, 28, 28), strip.getX(), strip.getY(),
                              juce::Colour::fromRGB(18, 18, 18), strip.getX(), strip.getBottom(), false);
    g.setGradientFill(head);
    g.fillRect(strip);
    g.setColour(juce::Colour::fromRGB(70, 70, 70).withAlpha(0.7f));
    g.drawLine(0.0f, strip.getBottom(), (float)getWidth(), strip.getBottom(), 1.0f);

    // Title
    g.setColour(juce::Colour::fromRGB(235, 235, 235).withAlpha(0.92f));
    g.setFont(juce::Font(juce::FontOptions(18.0f)).boldened());
    g.drawText("MIDI Generation", 22, 14, getWidth() - 44, 24, juce::Justification::centredLeft);

    g.setColour(juce::Colour::fromRGB(138, 255, 116).withAlpha(0.75f));
    g.setFont(juce::Font(juce::FontOptions(12.5f)));
    g.drawText("Full mode", 22, 34, getWidth() - 44, 18, juce::Justification::centredLeft);

    // Outer frame stroke (blocks/obvodka)
    auto r = getLocalBounds().toFloat().reduced(10.0f);
    g.setColour(juce::Colour::fromRGB(75, 75, 75).withAlpha(0.55f));
    g.drawRoundedRectangle(r, 12.0f, 1.0f);
}

void PluginEditor::resized()
{
    auto area = getLocalBounds();
    area.removeFromTop(56); // header strip
    area = area.reduced(20);
    // Give more space to piano roll; keep controls strip fixed
    auto bottom = area.removeFromBottom(210);
    area.removeFromBottom(12);
    auto top = area;

    // Top: piano roll + toolbar (like reference)
    auto toolbar = top.removeFromTop(44);

    // Fixed zones: [midiInfo][actions][gap][bpm]
    const int actionsW = 260;
    const int bpmW = 290;
    const int gapW = 12;
    auto rightPack = toolbar.removeFromRight(actionsW + gapW + bpmW);
    auto btnArea = rightPack.removeFromLeft(actionsW);
    rightPack.removeFromLeft(gapW);
    auto bpmArea = rightPack;

    midiInfoLabel->setBounds(toolbar);
    const int btnH = 26;
    const int gap = 8;

    // Actions block behind buttons
    if (actionsBlock)
        actionsBlock->setBounds(btnArea.expanded(6, 4));
    const int bw = (btnArea.getWidth() - gap * 3) / 4;
    generateButton->setBounds(btnArea.getX(), btnArea.getY() + 2, bw, btnH);
    regenButton->setBounds(btnArea.getX() + (bw + gap), btnArea.getY() + 2, bw, btnH);
    exportButton->setBounds(btnArea.getX() + 2 * (bw + gap), btnArea.getY() + 2, bw, btnH);
    dragDropButton->setBounds(btnArea.getX() + 3 * (bw + gap), btnArea.getY() + 2, bw, btnH);

    // Tiny labels under action buttons
    const int labelH = 12;
    const int labelY = btnArea.getBottom() - labelH + 2;
    if (goLabel) goLabel->setBounds(generateButton->getX(), labelY, generateButton->getWidth(), labelH);
    if (regenLabel) regenLabel->setBounds(regenButton->getX(), labelY, regenButton->getWidth(), labelH);
    if (exportLabel) exportLabel->setBounds(exportButton->getX(), labelY, exportButton->getWidth(), labelH);
    if (dragLabel) dragLabel->setBounds(dragDropButton->getX(), labelY, dragDropButton->getWidth(), labelH);

    // BPM block + slider
    if (bpmBlock)
        bpmBlock->setBounds(bpmArea.expanded(6, 4));
    if (bpmTopLabel)
        bpmTopLabel->setBounds(bpmArea.getX() + 10, bpmArea.getY() + 6, 44, 16);
    if (bpmTopEditor)
        bpmTopEditor->setBounds(bpmArea.getX() + 52, bpmArea.getY() + 4, bpmArea.getWidth() - 62, 22);

    auto statusRow = top.removeFromTop(26);
    randomizeButton->setBounds(statusRow.removeFromRight(70).reduced(0, 1));
    statusLabel->setBounds(statusRow.removeFromLeft(statusRow.getWidth() - 160));
    generationProgressLabel->setBounds(statusRow);

    midiVisualizer->setBounds(top);

    // Bottom: controls strip (reference-style)
    if (controlsViewport)
    {
        controlsViewport->setVisible(true);
        controlsViewport->setBounds(bottom);
        controlsViewport->setScrollBarsShown(false, false);
    }

    // Important: everything in the bottom strip lives inside controlsPanel (inside Viewport),
    // so layout must be in controlsPanel-local coordinates.
    if (controlsPanel && controlsViewport)
        controlsPanel->setSize(controlsViewport->getWidth(), controlsViewport->getHeight());

    auto stripOuter = (controlsPanel ? controlsPanel->getLocalBounds() : bottom).reduced(6);
    auto tabsArea = stripOuter.removeFromBottom(34).reduced(2, 4);
    const int tabH = tabsArea.getHeight();
    const int tabW = (tabsArea.getWidth() - 10) / 3;
    pageGenButton->setBounds(tabsArea.getX(), tabsArea.getY(), tabW, tabH);
    pageFeelButton->setBounds(tabsArea.getX() + tabW + 5, tabsArea.getY(), tabW, tabH);
    pagePostButton->setBounds(tabsArea.getX() + 2 * (tabW + 5), tabsArea.getY(), tabW, tabH);

    // Three-block strip layout (Instacomposer-like)
    for (auto& d : dividers)
        if (d) d->setVisible(false);

    const int capH = 12;
    const int pad = 10;
    const int rowGap = 8;
    const int colGap = 12;
    const int knobH = 58;      // includes textbox (AnimatedKnob)
    const int fieldH = 24;     // small linear/field controls

    auto blockArea = stripOuter;
    const int blockW = (blockArea.getWidth() - colGap * 2) / 3;
    auto b0 = blockArea.removeFromLeft(blockW);
    blockArea.removeFromLeft(colGap);
    auto b1 = blockArea.removeFromLeft(blockW);
    blockArea.removeFromLeft(colGap);
    auto b2 = blockArea;

    auto* gb0 = groupBlocks.size() > 0 ? groupBlocks[0].get() : nullptr;
    auto* gb1 = groupBlocks.size() > 1 ? groupBlocks[1].get() : nullptr;
    auto* gb2 = groupBlocks.size() > 2 ? groupBlocks[2].get() : nullptr;

    if (gb0) { gb0->setVisible(true); gb0->setBounds(b0); }
    if (gb1) { gb1->setVisible(true); gb1->setBounds(b1); }
    if (gb2) { gb2->setVisible(true); gb2->setBounds(b2); }

    auto captionFor = [&](juce::Component& comp) -> juce::Label*
    {
        if (auto it = captionByComponent.find(&comp); it != captionByComponent.end())
            return it->second;
        return nullptr;
    };

    auto layoutCell = [&](juce::Component& comp, juce::Rectangle<int> cell, int compH)
    {
        if (!comp.isVisible())
            return;
        if (auto* cap = captionFor(comp))
        {
            cap->setVisible(true);
            cap->setBounds(cell.getX(), cell.getY(), cell.getWidth(), capH);
        }
        comp.setBounds(cell.getX(), cell.getY() + capH, cell.getWidth(), compH);
    };

    auto layoutStack = [&](juce::Rectangle<int> r, std::initializer_list<std::pair<juce::Component*, int>> items)
    {
        int y = r.getY();
        for (auto& it : items)
        {
            auto* c = it.first;
            const int h = it.second;
            if (!c || !c->isVisible())
                continue;
            auto cell = juce::Rectangle<int>(r.getX(), y, r.getWidth(), capH + h);
            layoutCell(*c, cell, h);
            y += capH + h + rowGap;
        }
    };

    auto layoutRow2 = [&](juce::Rectangle<int> r, juce::Component* a, juce::Component* b, int h)
    {
        const int colW2 = (r.getWidth() - colGap) / 2;
        auto left = juce::Rectangle<int>(r.getX(), r.getY(), colW2, capH + h);
        auto right = juce::Rectangle<int>(r.getX() + colW2 + colGap, r.getY(), colW2, capH + h);
        if (a && a->isVisible()) layoutCell(*a, left, h);
        if (b && b->isVisible()) layoutCell(*b, right, h);
    };

    auto inner0 = b0.reduced(pad);
    auto inner1 = b1.reduced(pad);
    auto inner2 = b2.reduced(pad);
    inner0.removeFromTop(18);
    inner1.removeFromTop(18);
    inner2.removeFromTop(18);

    if (currentPage == Page::Gen)
    {
        if (gb0) gb0->setTitle("GENERAL");
        if (gb1) gb1->setTitle("SAMPLING");
        if (gb2) gb2->setTitle("HARMONY");

        // GENERAL: Key + three knobs in a row
        layoutStack(inner0, { { keyBox.get(), 24 } });
        auto rowY = inner0.getY() + (capH + 24 + rowGap);
        auto row = juce::Rectangle<int>(inner0.getX(), rowY, inner0.getWidth(), capH + knobH);
        const int colW3 = (row.getWidth() - colGap * 2) / 3;
        auto c1 = juce::Rectangle<int>(row.getX(), row.getY(), colW3, row.getHeight());
        auto c2 = juce::Rectangle<int>(row.getX() + colW3 + colGap, row.getY(), colW3, row.getHeight());
        auto c3 = juce::Rectangle<int>(row.getX() + 2 * (colW3 + colGap), row.getY(), colW3, row.getHeight());
        layoutCell(*seedSlider, c1, knobH);
        layoutCell(*maxLenSlider, c2, knobH);
        layoutCell(*targetDurationSlider, c3, knobH);

        // SAMPLING
        layoutRow2(inner1, temperatureKnob.get(), repetitionPenaltyKnob.get(), knobH);
        auto y2 = inner1.getY() + (capH + knobH + rowGap);
        // Small modern knobs instead of “boxes”
        layoutRow2(juce::Rectangle<int>(inner1.getX(), y2, inner1.getWidth(), capH + knobH),
                   topKSlider.get(), topPSlider.get(), knobH);
        layoutStack(juce::Rectangle<int>(inner1.getX(), y2 + (capH + knobH + rowGap), inner1.getWidth(), inner1.getHeight()),
                    { { ngramSlider.get(), knobH } });

        // HARMONY
        layoutStack(inner2, { { harmonyModeBox.get(), 24 }, { harmonyBiasSlider.get(), knobH },
                              { primerModeBox.get(), 24 }, { primerLenSlider.get(), knobH } });
        if (gb2) gb2->setVisible(true);
    }
    else if (currentPage == Page::Feel)
    {
        if (gb0) gb0->setTitle("DYNAMICS");
        if (gb1) gb1->setTitle("DENSITY");
        if (gb2) { gb2->setTitle(""); gb2->setVisible(false); }

        layoutRow2(inner0, velocityFeelKnob.get(), grooveFeelKnob.get(), knobH);
        layoutStack(inner1, { { maxPolyphonySlider.get(), fieldH }, { minBodyTokensSlider.get(), fieldH } });
    }
    else // Post
    {
        if (gb0) gb0->setTitle("TIMING");
        if (gb1) gb1->setTitle("HUMANIZE");
        if (gb2) gb2->setTitle("VELOCITY");

        layoutStack(inner0, { { bpmSlider.get(), fieldH }, { quantizeGridBox.get(), 24 },
                              { quantizeAmountSlider.get(), fieldH }, { swingAmountSlider.get(), fieldH } });
        layoutStack(inner1, { { humanizeTimeSlider.get(), fieldH }, { humanizeVelocitySlider.get(), fieldH } });
        layoutStack(inner2, { { velocityMinSlider.get(), fieldH }, { velocityMaxSlider.get(), fieldH } });
        if (gb2) gb2->setVisible(true);
    }

    // Keep panel size equal to viewport for non-scroll strip
    if (controlsPanel && controlsViewport)
        controlsPanel->setSize(controlsViewport->getWidth(), controlsViewport->getHeight());

    // Ensure tabs stay on top of blocks
    if (pageGenButton) pageGenButton->toFront(false);
    if (pageFeelButton) pageFeelButton->toFront(false);
    if (pagePostButton) pagePostButton->toFront(false);
}

//==============================================================================
void PluginEditor::timerCallback()
{
    updateStatusDisplay();
}

void PluginEditor::commitBpmFromText()
{
    if (bpmTopEditor == nullptr || bpmParam == nullptr)
        return;

    const auto txt = bpmTopEditor->getText().trim();
    if (txt.isEmpty())
        return;

    const double v = txt.getDoubleValue();
    const float bpm = juce::jlimit(40.0f, 200.0f, (float)v);

    if (auto* rap = dynamic_cast<juce::RangedAudioParameter*>(bpmParam))
    {
        const float norm = rap->convertTo0to1(bpm);
        rap->beginChangeGesture();
        rap->setValueNotifyingHost(norm);
        rap->endChangeGesture();
    }
}

void PluginEditor::updateStatusDisplay()
{
    if (!processor.isModelReady())
    {
        generateButton->setEnabled(false);
        statusLabel->setText(processor.getModelStatusText(), juce::dontSendNotification);
        return;
    }

    generateButton->setEnabled(true);

    if (processor.isGenerating()) {
        statusLabel->setText("Generating...", juce::dontSendNotification);
    } else {
        statusLabel->setText("Ready", juce::dontSendNotification);
    }

    const auto v = processor.getLastMidiVersion();
    if (v != lastMidiVersion)
    {
        lastMidiVersion = v;
        lastMidiMessages = processor.getLastMidiMessagesCopy();
        if (midiVisualizer)
            midiVisualizer->setMidiMessages(lastMidiMessages);

        int noteOn = 0;
        double maxT = 0.0;
        for (const auto& m : lastMidiMessages)
        {
            if (m.isNoteOn())
                ++noteOn;
            maxT = std::max(maxT, m.getTimeStamp());
        }
        if (midiInfoLabel)
            midiInfoLabel->setText("Notes: " + juce::String(noteOn) + " | Duration: " + juce::String(maxT, 2) + "s",
                                   juce::dontSendNotification);
    }

    if (midiVisualizer)
        midiVisualizer->setBpm((float)*apvts.getRawParameterValue("bpm"));

    // Keep BPM editor in sync (unless user is editing)
    if (bpmTopEditor && !bpmTopEditor->hasKeyboardFocus(true))
    {
        const float bpm = (float)*apvts.getRawParameterValue("bpm");
        const auto s = juce::String(bpm, 0);
        if (bpmTopEditor->getText() != s)
            bpmTopEditor->setText(s, false);
    }
}
