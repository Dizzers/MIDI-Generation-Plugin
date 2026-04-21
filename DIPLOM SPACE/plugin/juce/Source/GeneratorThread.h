#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "PluginProcessor.h"

class ModelInference;

/**
 * Background thread for async MIDI generation
 * Prevents blocking the audio or UI threads
 */
class GeneratorThread : public juce::Thread
{
public:
    GeneratorThread(PluginProcessor& proc, ModelInference& model);
    ~GeneratorThread() override;

    void run() override;
    void startGeneration(const PluginProcessor::GenerationParams& params);
    void cancelGeneration();
    bool isGenerating() const;

private:
    PluginProcessor& processor;
    ModelInference& modelInference;
    
    juce::CriticalSection paramLock;
    PluginProcessor::GenerationParams currentParams;
    bool shouldGenerate = false;
    bool isRunning = false;
    
    juce::WaitableEvent wakeupEvent;
};
