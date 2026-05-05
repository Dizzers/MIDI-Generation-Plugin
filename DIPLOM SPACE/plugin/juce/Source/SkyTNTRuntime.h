#pragma once

#include "SkyTNTTokenizer.h"

#include <juce_core/juce_core.h>

#include <atomic>
#include <memory>
#include <random>
#include <string>
#include <vector>

#if defined(MIDI_GEN_USE_ONNX) && MIDI_GEN_USE_ONNX
    #include <onnxruntime_cxx_api.h>
#endif

namespace skytnt
{

/** Abstract over runtime so we can compile without ONNX Runtime (header-only
 *  fallback that always reports "model not loaded").
 */
class Runtime
{
public:
    Runtime();
    ~Runtime();

    /** Configure paths to the two ONNX files + tokenizer config and try to
     *  load them. Returns false on failure with a human-readable message. */
    bool loadModels(const juce::File& modelBasePath,
                    const juce::File& modelTokenPath,
                    const juce::File& tokenizerJsonPath,
                    juce::String& errorOut);

    bool isLoaded() const noexcept { return loaded; }
    const juce::String& getStatus() const noexcept { return statusText; }
    const Tokenizer& getTokenizer() const noexcept { return tokenizer; }

    struct GenerateOptions
    {
        int maxLen = 256;            // total midi steps (including prompt)
        float temperature = 1.0f;
        float topP = 0.94f;
        int topK = 20;
        uint32_t seed = 42;
        bool disablePatchChange = false;
        bool disableControlChange = false;
        std::vector<int> disableChannels;
        // Optional setup to seed the prompt. If empty -> just BOS.
        // E.g. {"set_tempo": [0,0,0,bpm], "key_signature":[0,0,0,sf+7,mi]}
        std::vector<std::pair<juce::String, std::vector<int>>> setupEvents;
        // If true, generation aborts on user request (set abortFlag from another thread).
        std::atomic<bool>* abortFlag = nullptr;
    };

    /** Run the SkyTNT two-stage generation loop. Output is a 2D token grid
     *  shaped [mid_seq, max_token_seq], same as the Python pipeline. */
    bool generate(const GenerateOptions& opt,
                  std::vector<std::vector<int>>& outGrid,
                  juce::String& errorOut);

private:
    bool loaded = false;
    juce::String statusText = "ONNX runtime not initialised";
    Tokenizer tokenizer;

#if defined(MIDI_GEN_USE_ONNX) && MIDI_GEN_USE_ONNX
    struct ModelMeta
    {
        std::vector<std::string> inputNames;
        std::vector<std::string> outputNames;
        std::vector<int> pastKeyInputIdx;     // ordered "past_key_values.<i>.key" indices
        std::vector<int> pastValueInputIdx;
        std::vector<int> presentKeyOutputIdx; // ordered
        std::vector<int> presentValueOutputIdx;
        int xInputIdx = -1;
        int hiddenInputIdx = -1;        // token-model only
        int hiddenOutputIdx = -1;       // base-model only
        int yOutputIdx = -1;            // token-model only
        int numHeads = 0;
        int headDim = 0;
        int numLayers = 0;
        int embDim = 0;                 // discovered from base hidden output
    };

    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> baseSession;
    std::unique_ptr<Ort::Session> tokenSession;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    ModelMeta baseMeta;
    ModelMeta tokenMeta;

    bool inspectSession(Ort::Session& sess, ModelMeta& meta, juce::String& err);
#endif
};

} // namespace skytnt
