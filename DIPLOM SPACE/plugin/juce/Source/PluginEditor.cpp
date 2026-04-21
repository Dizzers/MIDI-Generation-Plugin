#include "PluginEditor.h"
#include "PluginProcessor.h"

//==============================================================================
PluginEditor::PluginEditor(PluginProcessor& p)
    : AudioProcessorEditor(&p), processor(p), apvts(p.getValueTreeState())
{
    setSize(700, 900);
    setResizable(true, true);
    setResizeLimits(600, 800, 1000, 1200);
    
    createControls();
    startTimer(100);  // Update status every 100ms
    
    // Initial status
    statusLabel->setText("Ready", juce::dontSendNotification);
}

PluginEditor::~PluginEditor()
{
    stopTimer();
}

//==============================================================================
void PluginEditor::createControls()
{
    // === BASIC SETTINGS PANEL ===
    // Role ComboBox
    roleBox = std::make_unique<juce::ComboBox>();
    roleBox->addItem("MELODY", 1);
    roleBox->addItem("BASS", 2);
    roleBox->addItem("CHORDS", 3);
    roleBox->setSelectedItemIndex(0);
    addAndMakeVisible(*roleBox);
    comboAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        apvts, "role", *roleBox));

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
    addAndMakeVisible(*keyBox);
    comboAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        apvts, "key", *keyBox));

    // Max Length Slider
    maxLenSlider = std::make_unique<juce::Slider>(juce::Slider::LinearHorizontal, juce::Slider::TextBoxRight);
    maxLenSlider->setRange(64.0, 512.0, 1.0);
    maxLenSlider->setTextValueSuffix(" tokens");
    addAndMakeVisible(*maxLenSlider);
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "maxLen", *maxLenSlider));

    // Target Duration Slider
    targetDurationSlider = std::make_unique<juce::Slider>(juce::Slider::LinearHorizontal, juce::Slider::TextBoxRight);
    targetDurationSlider->setRange(1.0, 5.0, 0.1);
    targetDurationSlider->setTextValueSuffix(" s");
    addAndMakeVisible(*targetDurationSlider);
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "targetSeconds", *targetDurationSlider));

    // === ROTARY KNOBS (Main Parameters) ===
    temperatureKnob = std::make_unique<juce::Slider>(juce::Slider::RotaryVerticalDrag, juce::Slider::TextBoxBelow);
    temperatureKnob->setRange(0.1, 2.0, 0.01);
    temperatureKnob->setTextValueSuffix("");
    addAndMakeVisible(*temperatureKnob);
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "temperature", *temperatureKnob));

    melodyLeapKnob = std::make_unique<juce::Slider>(juce::Slider::RotaryVerticalDrag, juce::Slider::TextBoxBelow);
    melodyLeapKnob->setRange(3.0, 24.0, 1.0);
    melodyLeapKnob->setTextValueSuffix(" st");
    addAndMakeVisible(*melodyLeapKnob);
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "maxMelodyLeap", *melodyLeapKnob));

    repetitionPenaltyKnob = std::make_unique<juce::Slider>(juce::Slider::RotaryVerticalDrag, juce::Slider::TextBoxBelow);
    repetitionPenaltyKnob->setRange(1.0, 2.0, 0.01);
    repetitionPenaltyKnob->setTextValueSuffix("");
    addAndMakeVisible(*repetitionPenaltyKnob);
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "repetitionPenalty", *repetitionPenaltyKnob));

    // === SAMPLING CONTROL SLIDERS ===
    topKSlider = std::make_unique<juce::Slider>(juce::Slider::LinearHorizontal, juce::Slider::TextBoxRight);
    topKSlider->setRange(1.0, 50.0, 1.0);
    topKSlider->setTextValueSuffix("");
    addAndMakeVisible(*topKSlider);
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "topK", *topKSlider));

    topPSlider = std::make_unique<juce::Slider>(juce::Slider::LinearHorizontal, juce::Slider::TextBoxRight);
    topPSlider->setRange(0.5, 1.0, 0.01);
    topPSlider->setTextValueSuffix("");
    addAndMakeVisible(*topPSlider);
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "topP", *topPSlider));

    ngramSlider = std::make_unique<juce::Slider>(juce::Slider::LinearHorizontal, juce::Slider::TextBoxRight);
    ngramSlider->setRange(2.0, 8.0, 1.0);
    ngramSlider->setTextValueSuffix("");
    addAndMakeVisible(*ngramSlider);
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "noRepeatNgramSize", *ngramSlider));

    // === MUSICAL CONSTRAINTS ===
    harmonyModeBox = std::make_unique<juce::ComboBox>();
    harmonyModeBox->addItem("None", 1);
    harmonyModeBox->addItem("Weak", 2);
    harmonyModeBox->addItem("Strong", 3);
    harmonyModeBox->setSelectedItemIndex(1);  // Default: Weak
    addAndMakeVisible(*harmonyModeBox);
    comboAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        apvts, "harmonyMode", *harmonyModeBox));

    harmonyBiasSlider = std::make_unique<juce::Slider>(juce::Slider::LinearHorizontal, juce::Slider::TextBoxRight);
    harmonyBiasSlider->setRange(0.0, 1.0, 0.01);
    harmonyBiasSlider->setTextValueSuffix("");
    addAndMakeVisible(*harmonyBiasSlider);
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "harmonyBias", *harmonyBiasSlider));

    primerModeBox = std::make_unique<juce::ComboBox>();
    primerModeBox->addItem("None", 1);
    primerModeBox->addItem("Dataset", 2);
    primerModeBox->setSelectedItemIndex(1);  // Default: Dataset
    addAndMakeVisible(*primerModeBox);
    comboAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(
        apvts, "primerMode", *primerModeBox));

    primerLenSlider = std::make_unique<juce::Slider>(juce::Slider::LinearHorizontal, juce::Slider::TextBoxRight);
    primerLenSlider->setRange(8.0, 32.0, 1.0);
    primerLenSlider->setTextValueSuffix("");
    addAndMakeVisible(*primerLenSlider);
    sliderAttachments.push_back(std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        apvts, "primerLen", *primerLenSlider));

    // === BUTTONS ===
    generateButton = std::make_unique<juce::TextButton>("GENERATE");
    generateButton->setColour(juce::TextButton::buttonColourId, juce::Colour(0x00FF80));  // Neon green
    generateButton->setColour(juce::TextButton::textColourOffId, juce::Colours::black);
    generateButton->onClick = [this]() {
        if (!processor.isGenerating()) {
            PluginProcessor::GenerationParams params;
            params.role = roleBox->getText().toStdString();
            params.key = keyBox->getText().toStdString();
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
            
            processor.startGeneration(params);
            statusLabel->setText("Generating...", juce::dontSendNotification);
        }
    };
    addAndMakeVisible(*generateButton);

    randomizeButton = std::make_unique<juce::TextButton>("RANDOMIZE");
    randomizeButton->setColour(juce::TextButton::buttonColourId, juce::Colour(0x00D4FF));  // Neon cyan
    randomizeButton->setColour(juce::TextButton::textColourOffId, juce::Colours::black);
    randomizeButton->onClick = [this]() {
        // Randomize parameters (keeping role fixed)
        temperatureKnob->setValue(0.5 + juce::Random::getSystemRandom().nextFloat() * 1.5);
        topKSlider->setValue(5.0 + juce::Random::getSystemRandom().nextFloat() * 20.0);
        melodyLeapKnob->setValue(6.0 + juce::Random::getSystemRandom().nextFloat() * 12.0);
    };
    addAndMakeVisible(*randomizeButton);

    outputWindowButton = std::make_unique<juce::TextButton>("📤 OUTPUT");
    outputWindowButton->setColour(juce::TextButton::buttonColourId, juce::Colour(0x666666));
    outputWindowButton->setColour(juce::TextButton::textColourOffId, juce::Colours::white);
    outputWindowButton->onClick = [this]() {
        processor.showOutputWindow();
    };
    addAndMakeVisible(*outputWindowButton);

    // === STATUS DISPLAY ===
    statusLabel = std::make_unique<juce::Label>("status", "Ready");
    statusLabel->setJustificationType(juce::Justification::centred);
    statusLabel->setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(*statusLabel);

    generationProgressLabel = std::make_unique<juce::Label>("progress", "");
    generationProgressLabel->setJustificationType(juce::Justification::centred);
    generationProgressLabel->setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(*generationProgressLabel);
}

//==============================================================================
void PluginEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0x1a1a1a));  // Dark background

    // Draw section headers
    g.setColour(juce::Colour(0x9d4edd));  // Purple accent
    g.setFont(juce::Font(14.0f).withExtraKerningFactor(0.1f).boldened());
    
    g.drawText("⚙️  BASIC SETTINGS", 20, 10, getWidth() - 40, 20, juce::Justification::topLeft);
    g.drawText("🎚️  SAMPLING", 20, 180, getWidth() - 40, 20, juce::Justification::topLeft);
    g.drawText("🎼  CONSTRAINTS", 20, 420, getWidth() - 40, 20, juce::Justification::topLeft);
}

void PluginEditor::resized()
{
    auto area = getLocalBounds().reduced(20);
    auto w = area.getWidth();

    int yPos = area.getY();

    // Title section
    yPos += 40;

    // BASIC SETTINGS
    roleBox->setBounds(20, yPos, 150, 30);
    keyBox->setBounds(180, yPos, 150, 30);
    yPos += 40;

    maxLenSlider->setBounds(20, yPos, w - 40, 50);
    yPos += 60;

    targetDurationSlider->setBounds(20, yPos, w - 40, 50);
    yPos += 70;

    // ROTARY KNOBS (Temperature, Leap, Penalty)
    int knobSize = 80;
    int knobSpacing = (w - 40 - 3 * knobSize) / 4;
    temperatureKnob->setBounds(20 + knobSpacing, yPos, knobSize, knobSize + 30);
    melodyLeapKnob->setBounds(20 + 2 * knobSpacing + knobSize, yPos, knobSize, knobSize + 30);
    repetitionPenaltyKnob->setBounds(20 + 3 * knobSpacing + 2 * knobSize, yPos, knobSize, knobSize + 30);
    yPos += knobSize + 40;

    // SAMPLING CONTROL
    topKSlider->setBounds(20, yPos, w - 40, 50);
    yPos += 60;

    topPSlider->setBounds(20, yPos, w - 40, 50);
    yPos += 60;

    ngramSlider->setBounds(20, yPos, w - 40, 50);
    yPos += 70;

    // MUSICAL CONSTRAINTS
    harmonyModeBox->setBounds(20, yPos, 150, 30);
    harmonyBiasSlider->setBounds(180, yPos, w - 200, 30);
    yPos += 50;

    primerModeBox->setBounds(20, yPos, 150, 30);
    primerLenSlider->setBounds(180, yPos, w - 200, 30);
    yPos += 60;

    // BUTTONS
    int btnWidth = (w - 40) / 3 - 5;
    generateButton->setBounds(20, yPos, btnWidth, 50);
    randomizeButton->setBounds(20 + btnWidth + 10, yPos, btnWidth, 50);
    outputWindowButton->setBounds(20 + 2 * (btnWidth + 10), yPos, btnWidth, 50);
    yPos += 70;

    // STATUS
    statusLabel->setBounds(20, yPos, w - 40, 30);
    yPos += 40;

    generationProgressLabel->setBounds(20, yPos, w - 40, 30);
}

//==============================================================================
void PluginEditor::timerCallback()
{
    updateStatusDisplay();
}

void PluginEditor::updateStatusDisplay()
{
    if (processor.isGenerating()) {
        statusLabel->setText("Generating... ⏳", juce::dontSendNotification);
    } else {
        statusLabel->setText("Ready ✓", juce::dontSendNotification);
    }
}
