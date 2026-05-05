#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_core/juce_core.h>

namespace skytnt { class Runtime; }

/**
 *  Facade over the SkyTNT ONNX runtime. Public surface is intentionally
 *  small so the rest of the plugin doesn't need to change much:
 *
 *    - generateMidi() returns a fully-formed list of MidiMessages with
 *      seconds-based timestamps (tempo applied via `bpm`).
 *    - isLoaded() reflects whether both ONNX models + tokenizer config were
 *      successfully loaded.
 *
 *  No tokens are exposed to callers any more; the SkyTNT vocab grid is an
 *  internal detail of this class.
 */
class ModelInference
{
public:
    ModelInference();
    ~ModelInference();

    struct GenerationResult
    {
        std::vector<juce::MidiMessage> messages;
        bool success = false;
        std::string errorMessage;
    };

    struct GenerationParams
    {
        juce::String key = "C_MAJOR"; // free-form, mapped to (sf, mi)
        int seed = 42;
        float temperature = 1.0f;
        int topK = 20;
        float topP = 0.94f;
        int maxLen = 256;          // total midi steps
        float bpm = 120.0f;
        bool disableControlChange = false;
        bool disablePatchChange = false;
    };

    GenerationResult generateMidi(const GenerationParams& params);

    void cancel() { abortFlag.store(true); }

    bool isLoaded() const;
    juce::String getStatusText() const;

private:
    void tryLoadFromDefaultLocations();

    std::unique_ptr<skytnt::Runtime> runtime;
    juce::String statusText { "Initialising..." };
    std::atomic<bool> abortFlag { false };

    static juce::File findArtifact(const char* relativePath);
};
