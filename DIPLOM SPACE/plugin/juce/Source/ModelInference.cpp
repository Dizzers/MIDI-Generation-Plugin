#include "ModelInference.h"
#include <juce_core/juce_core.h>

#if MIDI_GEN_USE_TORCH
    #include <torch/script.h>
    #include <torch/torch.h>
#endif

namespace
{
    juce::File findBinFileNearby(const juce::String& filename)
    {
        const auto tryPath = [&](const juce::File& base) -> juce::File
        {
            auto f = base.getChildFile("bin").getChildFile(filename);
            return f.existsAsFile() ? f : juce::File();
        };

        {
            auto f = tryPath(juce::File::getCurrentWorkingDirectory());
            if (f.existsAsFile())
                return f;
        }

        {
            auto exe = juce::File::getSpecialLocation(juce::File::currentExecutableFile);
            auto dir = exe.getParentDirectory();
            for (int i = 0; i < 8; ++i)
            {
                auto f = tryPath(dir);
                if (f.existsAsFile())
                    return f;
                dir = dir.getParentDirectory();
            }
        }

        return {};
    }

#if MIDI_GEN_USE_TORCH
    torch::Tensor topKTopPFilter(torch::Tensor logits, int topK, double topP)
    {
        // logits: (V)
        auto filtered = logits.clone();
        const int64_t vocab = filtered.size(0);

        if (topK > 0)
        {
            const int64_t k = std::min<int64_t>(topK, vocab);
            auto topk = std::get<0>(filtered.topk(k));
            auto threshold = topk.index({k - 1});
            filtered = torch::where(filtered < threshold, torch::full_like(filtered, -INFINITY), filtered);
        }

        if (topP < 1.0)
        {
            auto sortPair = filtered.sort(/*dim=*/0, /*descending=*/true);
            auto sortedLogits = std::get<0>(sortPair);
            auto sortedIdx = std::get<1>(sortPair);
            auto probs = torch::softmax(sortedLogits, 0);
            auto cumulative = torch::cumsum(probs, 0);
            auto removeMask = cumulative > topP;
            if (removeMask.numel() > 1)
            {
                auto shifted = removeMask.clone();
                shifted.index_put_({torch::indexing::Slice(1, torch::indexing::None)}, removeMask.index({torch::indexing::Slice(0, -1)}));
                shifted.index_put_({0}, false);
                removeMask = shifted;
            }
            sortedLogits.index_put_({removeMask}, -INFINITY);
            filtered = torch::full_like(filtered, -INFINITY);
            filtered.scatter_(0, sortedIdx, sortedLogits);
        }

        return filtered;
    }
#endif
}

ModelInference::ModelInference()
{
    loadCheckpoint();
    loadVocabulary();
    DBG("ModelInference initialized");
}

ModelInference::~ModelInference()
{
    DBG("ModelInference destroyed");
}

void ModelInference::loadCheckpoint()
{
#if MIDI_GEN_USE_TORCH
    auto modelPath = findModelFile();
    if (!modelPath.existsAsFile())
    {
        DBG("ModelInference: model file not found");
        modelLoaded = false;
        return;
    }

    try
    {
        module = torch::jit::load(modelPath.getFullPathName().toStdString());
        module.eval();
        modelLoaded = true;
        DBG("ModelInference: loaded model " << modelPath.getFullPathName());
    }
    catch (const c10::Error& e)
    {
        DBG("ModelInference: torch::jit::load failed: " << e.what());
        modelLoaded = false;
    }
#else
    DBG("ModelInference: built without LibTorch (USE_TORCH=OFF)");
    modelLoaded = false;
#endif
}

void ModelInference::loadVocabulary()
{
    std::string error;
    auto vocabPath = findVocabFile();
    if (!vocabPath.existsAsFile())
    {
        DBG("ModelInference: vocab.json not found");
        return;
    }
    if (!loadVocabJsonFile(vocabPath, error))
    {
        DBG("ModelInference: failed to load vocab.json: " << error);
        return;
    }
    DBG("ModelInference: vocabulary loaded, size=" << (int)token2id.size());
}

ModelInference::GenerationResult ModelInference::generateTokens(
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
    float targetSeconds)
{
    GenerationResult result;
    
    if (!modelLoaded)
    {
        result.errorMessage = "Model not loaded (or built without LibTorch)";
        result.success = false;
        return result;
    }

    if (token2id.empty() || id2token.empty() || bosId < 0)
    {
        result.errorMessage = "Vocabulary not loaded";
        result.success = false;
        return result;
    }

#if !MIDI_GEN_USE_TORCH
    (void)role; (void)key; (void)seed; (void)temperature; (void)topK; (void)topP;
    (void)repetitionPenalty; (void)noRepeatNgramSize; (void)maxMelodyLeap;
    (void)harmonyBias; (void)maxLen; (void)targetSeconds;
    result.errorMessage = "Built without LibTorch";
    result.success = false;
    return result;
#else
    // Map plugin UI "role" to a genre token for now (until we add explicit genre parameter).
    std::string genreToken = "<GENRE_TRAP>";
    if (role == "BASS")
        genreToken = "<GENRE_CLASSICAL>";

    // Map UI key like "C_MAJOR" -> vocab key token like "<KEY_C_MAJ>"
    auto keyToken = std::string("<KEY_UNKNOWN>");
    {
        auto upper = juce::String(key).toUpperCase();
        if (upper.endsWith("_MAJOR"))
            upper = upper.replace("_MAJOR", "_MAJ");
        if (upper.endsWith("_MINOR"))
            upper = upper.replace("_MINOR", "_MIN");
        keyToken = "<KEY_" + upper.toStdString() + ">";
        if (token2id.find(keyToken) == token2id.end())
            keyToken = "<KEY_UNKNOWN>";
    }

    auto itGenre = token2id.find(genreToken);
    if (itGenre == token2id.end())
        itGenre = token2id.find("<GENRE_TRAP>");

    int genreIndex = 0;
    {
        auto itIdx = genreTokenToIndex.find(itGenre->first);
        genreIndex = (itIdx != genreTokenToIndex.end()) ? itIdx->second : 0;
    }

    std::vector<int64_t> generated;
    generated.reserve((size_t)maxLen + 8);
    generated.push_back((int64_t)bosId);
    generated.push_back((int64_t)itGenre->second);

    auto itKey = token2id.find(keyToken);
    if (itKey != token2id.end())
        generated.push_back((int64_t)itKey->second);

    // Generation loop (single-sample)
    const int contextMax = 512; // matches model max_len in many configs; we clamp anyway
    double elapsedSeconds = 0.0;
    const double timeShiftResolution = 0.05;

    // Deterministic sampling per-request
    torch::manual_seed((uint64_t)juce::jmax(0, seed));

    for (int step = 0; step < maxLen; ++step)
    {
        // Context window
        const int64_t start = (int64_t)std::max<int>(0, (int)generated.size() - contextMax);
        const int64_t T = (int64_t)generated.size() - start;

        auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
        auto x = torch::empty({1, T}, options);
        auto* data = x.data_ptr<int64_t>();
        for (int64_t i = 0; i < T; ++i)
            data[i] = generated[(size_t)start + (size_t)i];

        auto g = torch::tensor({genreIndex}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

        torch::NoGradGuard guard;
        auto out = module.forward({x, g}).toTensor(); // (1, T, V)
        auto logits = out.index({0, T - 1});          // (V)

        // Temperature
        const double temp = std::max(1e-5, (double)temperature);
        logits = logits / temp;

        // Repetition penalty (simple set-based window)
        if (repetitionPenalty > 1.0f)
        {
            const int window = 128;
            std::unordered_set<int64_t> recent;
            for (int i = (int)generated.size() - 1; i >= 0 && (int)recent.size() < window && i >= 0; --i)
                recent.insert(generated[(size_t)i]);

            for (auto tok : recent)
            {
                auto v = logits.index({tok});
                logits.index_put_({tok}, torch::where(v >= 0, v / repetitionPenalty, v * repetitionPenalty));
            }
        }

        // Ban special tokens and genre/key tokens, etc.
        for (int id : bannedIds)
            logits.index_put_({id}, -INFINITY);

        // No-repeat ngram (basic)
        if (noRepeatNgramSize > 1 && (int)generated.size() >= (noRepeatNgramSize - 1))
        {
            const int n = noRepeatNgramSize;
            std::vector<int64_t> prefix(generated.end() - (n - 1), generated.end());
            for (size_t i = 0; i + (size_t)n <= generated.size(); ++i)
            {
                bool match = true;
                for (int j = 0; j < n - 1; ++j)
                {
                    if (generated[i + (size_t)j] != prefix[(size_t)j])
                    {
                        match = false;
                        break;
                    }
                }
                if (match)
                {
                    int64_t banned = generated[i + (size_t)(n - 1)];
                    logits.index_put_({banned}, -INFINITY);
                }
            }
        }

        // Top-k / top-p filter
        logits = topKTopPFilter(logits, topK, (double)topP);

        auto probs = torch::softmax(logits, 0);
        if (torch::isnan(probs).any().item<bool>() || probs.sum().item<double>() <= 0.0)
        {
            auto next = logits.argmax().item<int64_t>();
            generated.push_back(next);
        }
        else
        {
            auto next = torch::multinomial(probs, 1).item<int64_t>();
            generated.push_back(next);
        }

        const int last = (int)generated.back();
        if (eosId >= 0 && last == eosId)
            break;

        // Track approximate duration to stop near targetSeconds (based on TIME_SHIFT tokens)
        auto itTok = id2token.find(last);
        if (itTok != id2token.end())
        {
            const auto& name = itTok->second;
            if (name.rfind("TIME_SHIFT_", 0) == 0)
            {
                const auto pos = name.find_last_of('_');
                if (pos != std::string::npos)
                {
                    try
                    {
                        int steps = std::stoi(name.substr(pos + 1), nullptr, 16);
                        elapsedSeconds += (double)steps * timeShiftResolution;
                    }
                    catch (...) {}
                }
            }
        }

        if (targetSeconds > 0.0f && elapsedSeconds >= (double)targetSeconds && (int)generated.size() > 64)
        {
            if (eosId >= 0)
                generated.push_back((int64_t)eosId);
            break;
        }
    }

    result.tokenIds.reserve(generated.size());
    for (auto id : generated)
        result.tokenIds.push_back((int)id);
    result.success = true;
    return result;
#endif
}

juce::File ModelInference::findModelFile() const
{
    auto f = findBinFileNearby("model_best.ts.pt");
    if (f.existsAsFile())
        return f;
    return findBinFileNearby("model_best.pt");
}

juce::File ModelInference::findVocabFile() const
{
    return findBinFileNearby("vocab.json");
}

bool ModelInference::loadVocabJsonFile(const juce::File& vocabPath, std::string& errorOut)
{
    errorOut.clear();
    token2id.clear();
    id2token.clear();
    genreTokenToIndex.clear();
    bannedIds.clear();

    auto jsonText = vocabPath.loadFileAsString();
    auto parsed = juce::JSON::parse(jsonText);
    if (parsed.isVoid() || !parsed.isObject())
    {
        errorOut = "invalid json";
        return false;
    }

    auto* obj = parsed.getDynamicObject();
    if (obj == nullptr)
    {
        errorOut = "not an object";
        return false;
    }

    auto vToken2Id = obj->getProperty("token2id");
    auto vId2Token = obj->getProperty("id2token");
    if (!vToken2Id.isObject() || !vId2Token.isObject())
    {
        errorOut = "missing token2id/id2token";
        return false;
    }

    auto* token2idObj = vToken2Id.getDynamicObject();
    auto* id2tokenObj = vId2Token.getDynamicObject();
    if (!token2idObj || !id2tokenObj)
    {
        errorOut = "dynamic object missing";
        return false;
    }

    // token2id
    for (const auto& entry : token2idObj->getProperties())
        token2id.emplace(entry.name.toString().toStdString(), (int)entry.value.toString().getIntValue());

    // id2token
    for (const auto& entry : id2tokenObj->getProperties())
        id2token.emplace(entry.name.toString().getIntValue(), entry.value.toString().toStdString());

    auto it = token2id.find("<BOS>");
    bosId = (it != token2id.end()) ? it->second : -1;
    it = token2id.find("<EOS>");
    eosId = (it != token2id.end()) ? it->second : -1;
    it = token2id.find("<UNK>");
    unkId = (it != token2id.end()) ? it->second : -1;

    // Genre index mapping must match python (sorted genre tokens).
    std::vector<std::string> genres;
    genres.reserve(8);
    for (const auto& kv : token2id)
        if (kv.first.rfind("<GENRE_", 0) == 0)
            genres.push_back(kv.first);
    std::sort(genres.begin(), genres.end());
    for (size_t idx = 0; idx < genres.size(); ++idx)
        genreTokenToIndex.emplace(genres[idx], (int)idx);

    // Precompute banned ids similar to python generate.py
    for (const auto& kv : token2id)
    {
        const auto& tok = kv.first;
        const int id = kv.second;
        if (tok == "<PAD>" || tok == "<BOS>" || tok == "<UNK>" || tok.rfind("<GENRE_", 0) == 0 || tok.rfind("<KEY_", 0) == 0)
            bannedIds.push_back(id);
    }

    if (bosId < 0)
    {
        errorOut = "missing <BOS>";
        return false;
    }

    return true;
}
