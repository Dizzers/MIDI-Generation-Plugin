#include "GeneratorThread.h"
#include "ModelInference.h"
#include "MidiGenerator.h"
#include "MidiPostProcessor.h"

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
            currentParams.key.toStdString(),
            currentParams.seed,
            currentParams.temperature,
            currentParams.topK,
            currentParams.topP,
            currentParams.repetitionPenalty,
            currentParams.noRepeatNgramSize,
            currentParams.maxMelodyLeap,
            currentParams.harmonyBias,
            currentParams.maxLen,
            currentParams.targetSeconds,
            currentParams.velocityFeel,
            currentParams.grooveFeel,
            currentParams.maxPolyphony,
            currentParams.minBodyTokens);

        if (result.success) {
            // Convert tokens to MIDI
            auto midiMessages = MidiGenerator::convertTokensToMidi(result.tokenIds);

            MidiPostProcessor::Params post;
            post.seed = currentParams.seed;
            post.bpm = currentParams.bpm;
            post.quantizeGrid = static_cast<MidiPostProcessor::QuantizeGrid>(currentParams.quantizeGrid);
            post.quantizeAmount = currentParams.quantizeAmount;
            post.swingAmount = currentParams.swingAmount;
            post.humanizeTimeMs = currentParams.humanizeTimeMs;
            post.humanizeVelocity = currentParams.humanizeVelocity;
            post.velocityMin = currentParams.velocityMin;
            post.velocityMax = currentParams.velocityMax;

            midiMessages = MidiPostProcessor::process(midiMessages, post);
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
