#include "GeneratorThread.h"
#include "ModelInference.h"
#include "MidiGenerator.h"

GeneratorThread::GeneratorThread(PluginProcessor& proc, ModelInference& model)
    : Thread("GeneratorThread"), processor(proc), modelInference(model)
{
}

GeneratorThread::~GeneratorThread()
{
    stopThread(5000);
}

void GeneratorThread::run()
{
    while (!threadShouldExit()) {
        // Wait for generation request
        wakeupEvent.wait(100);
        
        if (threadShouldExit())
            break;

        {
            juce::ScopedLock lock(paramLock);
            if (!shouldGenerate)
                continue;
            isRunning = true;
        }

        DBG("Generation thread: starting token generation");

        // Generate tokens
        auto result = modelInference.generateTokens(
            currentParams.role.toStdString(),
            currentParams.key.toStdString(),
            currentParams.temperature,
            currentParams.topK,
            currentParams.topP,
            currentParams.repetitionPenalty,
            currentParams.noRepeatNgramSize,
            currentParams.maxMelodyLeap,
            currentParams.harmonyBias,
            currentParams.maxLen,
            currentParams.targetSeconds);

        if (result.success) {
            // Convert tokens to MIDI
            auto midiMessages = MidiGenerator::convertTokensToMidi(result.tokenIds);
            processor.queueMidiOutput(midiMessages);
            DBG("Generated " << midiMessages.size() << " MIDI messages");
        } else {
            DBG("Token generation failed: " << result.errorMessage);
        }

        {
            juce::ScopedLock lock(paramLock);
            shouldGenerate = false;
            isRunning = false;
        }
    }
}

void GeneratorThread::startGeneration(const PluginProcessor::GenerationParams& params)
{
    {
        juce::ScopedLock lock(paramLock);
        currentParams = params;
        shouldGenerate = true;
    }
    wakeupEvent.signal();
    DBG("Generation requested");
}

void GeneratorThread::cancelGeneration()
{
    juce::ScopedLock lock(paramLock);
    shouldGenerate = false;
}

bool GeneratorThread::isGenerating() const
{
    juce::ScopedLock lock(paramLock);
    return isRunning;
}
