#include "GeneratorThread.h"
#include "ModelInference.h"
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
    while (!threadShouldExit())
    {
        wakeupEvent.wait(100);
        if (threadShouldExit()) break;

        {
            juce::ScopedLock lock(paramLock);
            if (!shouldGenerate) continue;
            isRunning = true;
        }

        DBG("Generation thread: invoking SkyTNT runtime");

        ModelInference::GenerationParams mp;
        mp.key            = currentParams.key;
        mp.seed           = currentParams.seed;
        mp.temperature    = currentParams.temperature;
        mp.topK           = currentParams.topK;
        mp.topP           = currentParams.topP;
        mp.maxLen         = currentParams.maxLen;
        mp.bpm            = currentParams.bpm;
        mp.disablePatchChange = false;
        mp.disableControlChange = false;

        auto res = modelInference.generateMidi(mp);

        if (res.success)
        {
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

            auto post_msgs = MidiPostProcessor::process(res.messages, post);
            processor.queueMidiOutput(post_msgs);
            DBG("Generated " << post_msgs.size() << " MIDI messages");
        }
        else
        {
            DBG("SkyTNT generation failed: " << res.errorMessage);
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
}

void GeneratorThread::cancelGeneration()
{
    {
        juce::ScopedLock lock(paramLock);
        shouldGenerate = false;
    }
    modelInference.cancel();
}

bool GeneratorThread::isGenerating() const
{
    juce::ScopedLock lock(paramLock);
    return isRunning;
}
