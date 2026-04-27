#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <vector>
#include <memory>

class GeneratorThread;
class ModelInference;
class OutputWindow;

/**
 * Main AudioProcessor for MIDI Generation Plugin
 * Handles:
 * - 14 synthesis parameters
 * - MIDI output to DAW
 * - Async token generation
 * - Parameter persistence
 */
class PluginProcessor : public juce::AudioProcessor
{
public:
    PluginProcessor();
    ~PluginProcessor() override;

    // JUCE AudioProcessor Implementation
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return true; }
    bool isMidiEffect() const override { return true; }
    double getTailLengthSeconds() const override { return 0.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return "Default"; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    // === GENERATION CONTROL ===
    struct GenerationParams
    {
        juce::String role;          // MELODY, BASS, CHORDS
        juce::String key;           // A_MINOR, C_MAJOR, etc.
        int seed = 42;              // deterministic generation
        float temperature = 0.95f;
        int topK = 12;
        float topP = 0.9f;
        float repetitionPenalty = 1.15f;
        int noRepeatNgramSize = 4;
        int maxMelodyLeap = 12;
        juce::String harmonyMode;   // None, Weak, Strong
        float harmonyBias = 0.35f;
        int maxLen = 256;
        float targetSeconds = 2.5f;
        juce::String primerMode;    // None, Dataset
        int primerLen = 24;
    };

    void startGeneration(const GenerationParams& params);
    void regenerateLast();
    void cancelGeneration();
    bool isGenerating() const;
    
    // === MIDI OUTPUT ===
    void queueMidiOutput(const std::vector<juce::MidiMessage>& midiMessages);
    
    // === OUTPUT WINDOW ===
    void showOutputWindow();
    OutputWindow* getOutputWindow() { return outputWindow.get(); }

    // === PARAMETER ACCESS ===
    juce::AudioProcessorValueTreeState& getValueTreeState() { return apvts; }

private:
    // === PARAMETERS ===
    void createParameters();
    void setupValueTreeState();

    juce::AudioProcessorValueTreeState apvts;
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    // === THREADING ===
    std::unique_ptr<GeneratorThread> generatorThread;
    std::unique_ptr<ModelInference> modelInference;
    std::unique_ptr<OutputWindow> outputWindow;

    // Remember last generation request for regen
    juce::CriticalSection lastParamsLock;
    GenerationParams lastParams;
    bool hasLastParams = false;

    // === GENERATED CLIP SCHEDULER ===
    struct ScheduledMidiEvent
    {
        juce::MidiMessage message;
        int64_t samplePos = 0; // absolute sample position since clip start
    };

    juce::CriticalSection midiQueueLock;
    std::vector<ScheduledMidiEvent> scheduledClip;
    size_t scheduledReadIndex = 0;
    int64_t clipPlayheadSamples = 0;

    double currentSampleRate = 44100.0;
    int currentBlockSize = 512;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PluginProcessor)
};
