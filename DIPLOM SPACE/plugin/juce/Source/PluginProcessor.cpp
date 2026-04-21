#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "GeneratorThread.h"
#include "ModelInference.h"
#include "OutputWindow.h"

//==============================================================================
PluginProcessor::PluginProcessor()
    : AudioProcessor(BusesProperties()
        .withInput("Input", juce::AudioChannelSet::stereo(), true)
        .withOutput("Output", juce::AudioChannelSet::stereo(), true)
        .withInput("MIDI In", juce::AudioChannelSet::discreteChannels(16), true)
        .withOutput("MIDI Out", juce::AudioChannelSet::discreteChannels(16), true))
    , apvts(*this, nullptr, "PARAMETERS", createParameterLayout())
{
    createParameters();
    
    // Initialize components
    try {
        modelInference = std::make_unique<ModelInference>();
        generatorThread = std::make_unique<GeneratorThread>(*this, *modelInference);
        generatorThread->startThread();
        outputWindow = std::make_unique<OutputWindow>(*this);
    } catch (const std::exception& e) {
        DBG("Error initializing plugin components: " << e.what());
    }
}

PluginProcessor::~PluginProcessor()
{
    if (generatorThread) {
        generatorThread->stopThread(5000);
    }
}

//==============================================================================
juce::AudioProcessorValueTreeState::ParameterLayout PluginProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
    
    // === BASIC SETTINGS ===
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        "role", "Role",
        juce::StringArray{"MELODY", "BASS", "CHORDS"}, 0));
    
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        "key", "Key",
        juce::StringArray{
            "C_MAJOR", "G_MAJOR", "D_MAJOR", "A_MAJOR", "E_MAJOR", "B_MAJOR",
            "F_SHARP_MAJOR", "C_SHARP_MAJOR",
            "F_MAJOR", "B_FLAT_MAJOR", "E_FLAT_MAJOR", "A_FLAT_MAJOR",
            "A_MINOR", "E_MINOR", "B_MINOR", "F_SHARP_MINOR", "C_SHARP_MINOR",
            "G_SHARP_MINOR", "D_SHARP_MINOR", "A_SHARP_MINOR",
            "D_MINOR", "G_MINOR", "C_MINOR", "F_MINOR"
        }, 0));
    
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "maxLen", "Max Length", 64.0f, 512.0f, 256.0f));
    
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "targetSeconds", "Target Duration", 1.0f, 5.0f, 2.5f));
    
    // === SAMPLING CONTROL ===
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "temperature", "Temperature", 0.1f, 2.0f, 0.95f));
    
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "topK", "Top-K", 1.0f, 50.0f, 12.0f));
    
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "topP", "Top-P", 0.5f, 1.0f, 0.9f));
    
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "repetitionPenalty", "Repetition Penalty", 1.0f, 2.0f, 1.15f));
    
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "noRepeatNgramSize", "N-gram Size", 2.0f, 8.0f, 4.0f));
    
    // === MUSICAL CONSTRAINTS ===
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "maxMelodyLeap", "Melody Leap", 3.0f, 24.0f, 12.0f));
    
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        "harmonyMode", "Harmony Mode",
        juce::StringArray{"None", "Weak", "Strong"}, 1));
    
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "harmonyBias", "Harmony Bias", 0.0f, 1.0f, 0.35f));
    
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        "primerMode", "Primer Mode",
        juce::StringArray{"None", "Dataset"}, 1));
    
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        "primerLen", "Primer Length", 8.0f, 32.0f, 24.0f));
    
    return { params.begin(), params.end() };
}

void PluginProcessor::createParameters()
{
    // Parameters are created via createParameterLayout()
    // This method is kept for future expansion if needed
}

void PluginProcessor::setupValueTreeState()
{
    // Value tree state is initialized in constructor
}

//==============================================================================
void PluginProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate;
    currentBlockSize = samplesPerBlock;
    DBG("Preparing to play: " << sampleRate << "Hz, block size: " << samplesPerBlock);
}

void PluginProcessor::releaseResources()
{
    DBG("Releasing resources");
}

void PluginProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{
    auto totalNumInputChannels = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();

    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear(i, 0, buffer.getNumSamples());

    // Process MIDI output
    {
        juce::ScopedLock lock(midiQueueLock);
        if (!midiOutputQueue.empty()) {
            for (auto& msg : midiOutputQueue) {
                midi.addEvent(msg, 0);
            }
            midiOutputQueue.clear();
        }
    }
}

//==============================================================================
juce::AudioProcessorEditor* PluginProcessor::createEditor()
{
    return new PluginEditor(*this);
}

//==============================================================================
void PluginProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
    DBG("Plugin state saved");
}

void PluginProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));
    if (xmlState != nullptr)
        if (xmlState->hasTagName(apvts.state.getType()))
            apvts.replaceState(juce::ValueTree::fromXml(*xmlState));
    DBG("Plugin state restored");
}

//==============================================================================
void PluginProcessor::startGeneration(const GenerationParams& params)
{
    if (generatorThread && modelInference) {
        generatorThread->startGeneration(params);
        DBG("Generation started: " << params.role << " in " << params.key
            << " (temp=" << params.temperature << ")");
    }
}

void PluginProcessor::cancelGeneration()
{
    if (generatorThread) {
        generatorThread->cancelGeneration();
        DBG("Generation cancelled");
    }
}

bool PluginProcessor::isGenerating() const
{
    return generatorThread && generatorThread->isGenerating();
}

void PluginProcessor::queueMidiOutput(const std::vector<juce::MidiMessage>& midiMessages)
{
    juce::ScopedLock lock(midiQueueLock);
    midiOutputQueue = midiMessages;
}

void PluginProcessor::showOutputWindow()
{
    if (outputWindow) {
        outputWindow->toFront(true);
    }
}

//==============================================================================
// This creates new instances of the plugin
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new PluginProcessor();
}
