#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <juce_core/juce_core.h>

#if MIDI_GEN_USE_TORCH
    #include <torch/script.h>
#endif

class PluginProcessor;

/**
 * PyTorch model inference wrapper
 * Loads pre-trained checkpoint and generates token sequences
 */
class ModelInference
{
public:
    ModelInference();
    ~ModelInference();

    struct GenerationResult
    {
        std::vector<int> tokenIds;
        bool success = false;
        std::string errorMessage;
    };

    /**
     * Generate token sequence for MIDI
     * @param role MELODY, BASS, or CHORDS
     * @param key A_MINOR, C_MAJOR, etc.
     * @param temperature Sampling temperature (>1 = more creative)
     * @param topK Keep top-K candidates
     * @param topP Nucleus sampling threshold
     * @param repetitionPenalty Penalize repeating tokens
     * @param maxLen Maximum sequence length
     * @param targetSeconds Target duration estimate
     * @param maxMelodyLeap Max semitones between notes (melody only)
     * @param harmonyBias Boost notes in key
     */
    GenerationResult generateTokens(
        const std::string& role,
        const std::string& key,
        int seed,
        float temperature,
        int topK,
        float topP,
        float repetitionPenalty,
        int noRepeatNgramSize,
        int maxMelodyLeap,
        float harmonyBias,
        int maxLen,
        float targetSeconds);

    bool isLoaded() const { return modelLoaded; }

private:
    bool modelLoaded = false;
    void loadCheckpoint();
    void loadVocabulary();

    bool loadVocabJsonFile(const juce::File& vocabPath, std::string& errorOut);
    juce::File findModelFile() const;
    juce::File findVocabFile() const;

    // Vocab / conditioning
    std::unordered_map<std::string, int> token2id;
    std::unordered_map<int, std::string> id2token;
    std::unordered_map<std::string, int> genreTokenToIndex;
    std::vector<int> bannedIds;
    int bosId = -1;
    int eosId = -1;
    int unkId = -1;

#if MIDI_GEN_USE_TORCH
    torch::jit::Module module;
#endif
};
