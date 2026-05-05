#include "SkyTNTRuntime.h"

#include <algorithm>
#include <cstring>

#if defined(MIDI_GEN_USE_ONNX) && MIDI_GEN_USE_ONNX
    #include <onnxruntime_cxx_api.h>
#endif

namespace skytnt
{

Runtime::Runtime() = default;
Runtime::~Runtime() = default;

#if !(defined(MIDI_GEN_USE_ONNX) && MIDI_GEN_USE_ONNX)

bool Runtime::loadModels(const juce::File&, const juce::File&, const juce::File&, juce::String& err)
{
    err = "Plugin built without ONNX Runtime (MIDI_GEN_USE_ONNX=0)";
    statusText = err;
    loaded = false;
    return false;
}

bool Runtime::generate(const GenerateOptions&, std::vector<std::vector<int>>&, juce::String& err)
{
    err = "Plugin built without ONNX Runtime (MIDI_GEN_USE_ONNX=0)";
    return false;
}

#else

namespace
{
#if defined(_WIN32)
inline std::wstring toOrtPath(const juce::File& f)
{
    juce::String s = f.getFullPathName();
    return std::wstring(s.toWideCharPointer());
}
#else
inline std::string toOrtPath(const juce::File& f)
{
    return f.getFullPathName().toStdString();
}
#endif
} // namespace

bool Runtime::inspectSession(Ort::Session& sess, ModelMeta& meta, juce::String& err)
{
    Ort::AllocatorWithDefaultOptions alloc;
    size_t numIn = sess.GetInputCount();
    size_t numOut = sess.GetOutputCount();
    meta.inputNames.clear();
    meta.outputNames.clear();
    meta.inputNames.resize(numIn);
    meta.outputNames.resize(numOut);

    struct PastIdx { int layer; int idx; };
    std::vector<PastIdx> pastKeys, pastVals;
    std::vector<PastIdx> presentKeys, presentVals;

    for (size_t i = 0; i < numIn; ++i)
    {
        auto p = sess.GetInputNameAllocated(i, alloc);
        meta.inputNames[i] = p.get();
        const std::string& nm = meta.inputNames[i];

        if (nm == "x") meta.xInputIdx = (int)i;
        else if (nm == "hidden") meta.hiddenInputIdx = (int)i;
        else if (nm.rfind("past_key_values.", 0) == 0)
        {
            // past_key_values.<L>.key / past_key_values.<L>.value
            size_t firstDot = nm.find('.');
            size_t secondDot = nm.find('.', firstDot + 1);
            if (firstDot == std::string::npos || secondDot == std::string::npos) continue;
            int layer = std::atoi(nm.substr(firstDot + 1, secondDot - firstDot - 1).c_str());
            std::string suffix = nm.substr(secondDot + 1);
            if (suffix == "key")   pastKeys.push_back({layer, (int)i});
            else if (suffix == "value") pastVals.push_back({layer, (int)i});
        }
    }
    for (size_t i = 0; i < numOut; ++i)
    {
        auto p = sess.GetOutputNameAllocated(i, alloc);
        meta.outputNames[i] = p.get();
        const std::string& nm = meta.outputNames[i];
        if (nm == "hidden") meta.hiddenOutputIdx = (int)i;
        else if (nm == "y") meta.yOutputIdx = (int)i;
        else if (nm.rfind("present.", 0) == 0)
        {
            size_t firstDot = nm.find('.');
            size_t secondDot = nm.find('.', firstDot + 1);
            if (firstDot == std::string::npos || secondDot == std::string::npos) continue;
            int layer = std::atoi(nm.substr(firstDot + 1, secondDot - firstDot - 1).c_str());
            std::string suffix = nm.substr(secondDot + 1);
            if (suffix == "key")   presentKeys.push_back({layer, (int)i});
            else if (suffix == "value") presentVals.push_back({layer, (int)i});
        }
    }
    auto byLayer = [](const PastIdx& a, const PastIdx& b){ return a.layer < b.layer; };
    std::sort(pastKeys.begin(), pastKeys.end(), byLayer);
    std::sort(pastVals.begin(), pastVals.end(), byLayer);
    std::sort(presentKeys.begin(), presentKeys.end(), byLayer);
    std::sort(presentVals.begin(), presentVals.end(), byLayer);

    if (pastKeys.size() != pastVals.size() || pastKeys.size() != presentKeys.size())
    {
        err = "model graph is missing past/present KV inputs";
        return false;
    }
    meta.numLayers = (int)pastKeys.size();
    meta.pastKeyInputIdx.clear();
    meta.pastValueInputIdx.clear();
    meta.presentKeyOutputIdx.clear();
    meta.presentValueOutputIdx.clear();
    for (auto& p : pastKeys)    meta.pastKeyInputIdx.push_back(p.idx);
    for (auto& p : pastVals)    meta.pastValueInputIdx.push_back(p.idx);
    for (auto& p : presentKeys) meta.presentKeyOutputIdx.push_back(p.idx);
    for (auto& p : presentVals) meta.presentValueOutputIdx.push_back(p.idx);

    if (!pastKeys.empty())
    {
        auto info = sess.GetInputTypeInfo(pastKeys.front().idx);
        auto t = info.GetTensorTypeAndShapeInfo();
        auto shape = t.GetShape();
        // shape: [batch, num_heads, past_seq, head_dim]
        if (shape.size() == 4)
        {
            meta.numHeads = (shape[1] > 0) ? (int)shape[1] : 0;
            meta.headDim  = (shape[3] > 0) ? (int)shape[3] : 0;
        }
    }
    if (meta.hiddenOutputIdx >= 0)
    {
        auto info = sess.GetOutputTypeInfo(meta.hiddenOutputIdx);
        auto t = info.GetTensorTypeAndShapeInfo();
        auto shape = t.GetShape();
        if (shape.size() == 3 && shape[2] > 0)
            meta.embDim = (int)shape[2];
    }
    return true;
}

bool Runtime::loadModels(const juce::File& basePath, const juce::File& tokenPath,
                         const juce::File& tokenizerJson, juce::String& errorOut)
{
    loaded = false;
    statusText = "Loading SkyTNT models...";

    if (!basePath.existsAsFile())
    { errorOut = "model_base.onnx not found: " + basePath.getFullPathName(); statusText = errorOut; return false; }
    if (!tokenPath.existsAsFile())
    { errorOut = "model_token.onnx not found: " + tokenPath.getFullPathName(); statusText = errorOut; return false; }

    juce::String tokErr;
    if (tokenizerJson.existsAsFile())
    {
        if (! tokenizer.loadFromJsonFile(tokenizerJson, tokErr))
        {
            errorOut = "tokenizer config load failed: " + tokErr;
            statusText = errorOut;
            return false;
        }
    }
    else
    {
        // Fallback: assume tv2 + optimise=true (matches default config).
        if (! tokenizer.buildFromVersion("v2", true))
        {
            errorOut = "Failed to bootstrap tokenizer (V2)";
            statusText = errorOut;
            return false;
        }
    }

    try
    {
        if (!env)
            env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "skytnt");

        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        opts.SetIntraOpNumThreads(juce::jmax(1, (int) std::thread::hardware_concurrency() / 2));

        baseSession  = std::make_unique<Ort::Session>(*env, toOrtPath(basePath).c_str(), opts);
        tokenSession = std::make_unique<Ort::Session>(*env, toOrtPath(tokenPath).c_str(), opts);

        juce::String e1, e2;
        if (!inspectSession(*baseSession, baseMeta, e1)) { errorOut = "base: " + e1; statusText = errorOut; return false; }
        if (!inspectSession(*tokenSession, tokenMeta, e2)) { errorOut = "token: " + e2; statusText = errorOut; return false; }
    }
    catch (const Ort::Exception& e)
    {
        errorOut = juce::String("ONNX exception: ") + e.what();
        statusText = errorOut;
        return false;
    }

    loaded = true;
    statusText = "SkyTNT models loaded ("
               + juce::String(baseMeta.numLayers) + "/" + juce::String(tokenMeta.numLayers)
               + " layers, vocab=" + juce::String(tokenizer.getVocabSize()) + ")";
    return true;
}

namespace
{

// CPU softmax along last axis, in-place.
void softmaxLastAxis(float* data, int rows, int cols)
{
    for (int r = 0; r < rows; ++r)
    {
        float* row = data + r * cols;
        float m = row[0];
        for (int i = 1; i < cols; ++i) m = std::max(m, row[i]);
        float s = 0.0f;
        for (int i = 0; i < cols; ++i) { row[i] = std::exp(row[i] - m); s += row[i]; }
        if (s <= 0.0f) s = 1.0f;
        for (int i = 0; i < cols; ++i) row[i] /= s;
    }
}

// Sample one token id from `probs` (length=cols) using top-p / top-k.
int sampleTopPK(const float* probs, int cols, float topP, int topK, std::mt19937& rng)
{
    std::vector<int> idx(cols);
    for (int i = 0; i < cols; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return probs[a] > probs[b]; });

    std::vector<float> sorted(cols);
    for (int i = 0; i < cols; ++i) sorted[i] = probs[idx[i]];

    float cum = 0.0f;
    int cutoff = cols;
    for (int i = 0; i < cols; ++i)
    {
        // mask if cum_sum - sorted > p (matches numpy/torch impl)
        if (cum - sorted[i] > topP) { cutoff = i; break; }
        cum += sorted[i];
    }
    cutoff = std::min(cutoff, std::max(1, topK));

    double total = 0.0;
    for (int i = 0; i < cutoff; ++i) total += sorted[i];
    if (total <= 0.0) return idx[0];

    std::uniform_real_distribution<double> dist(0.0, total);
    double r = dist(rng);
    double acc = 0.0;
    for (int i = 0; i < cutoff; ++i)
    {
        acc += sorted[i];
        if (r <= acc) return idx[i];
    }
    return idx[cutoff - 1];
}

// Helper: build an Ort::Value owning a float buffer. We always over-allocate by
// one element so the data pointer stays valid even for fully-empty tensors
// (ORT's CreateTensor requires non-null data even when element_count == 0).
Ort::Value makeFloatTensor(const Ort::MemoryInfo& mi,
                           std::vector<float>& backing,
                           const std::vector<int64_t>& shape)
{
    int64_t total = 1;
    for (auto d : shape) total *= d;
    backing.assign((size_t)std::max<int64_t>(total, 1), 0.0f);
    return Ort::Value::CreateTensor<float>(mi, backing.data(), (size_t)total,
                                           shape.data(), shape.size());
}

Ort::Value makeInt64Tensor(const Ort::MemoryInfo& mi,
                           std::vector<int64_t>& backing,
                           const std::vector<int64_t>& shape)
{
    int64_t total = 1;
    for (auto d : shape) total *= d;
    backing.assign((size_t)std::max<int64_t>(total, 1), 0);
    return Ort::Value::CreateTensor<int64_t>(mi, backing.data(), (size_t)total,
                                             shape.data(), shape.size());
}

} // namespace

bool Runtime::generate(const GenerateOptions& opt,
                       std::vector<std::vector<int>>& outGrid,
                       juce::String& errorOut)
{
    if (!loaded)
    {
        errorOut = "Models not loaded";
        return false;
    }
    if (!tokenizer.isLoaded())
    {
        errorOut = "Tokenizer not loaded";
        return false;
    }
    const int maxTokenSeq = tokenizer.getMaxTokenSeq();
    const int vocab = tokenizer.getVocabSize();
    const int padId = tokenizer.getPadId();
    const int bosId = tokenizer.getBosId();
    const int eosId = tokenizer.getEosId();

    // === Build initial token grid ===
    std::vector<std::vector<int>> grid;
    {
        std::vector<int> bosRow(maxTokenSeq, padId);
        bosRow[0] = bosId;
        grid.push_back(std::move(bosRow));
        for (const auto& kv : opt.setupEvents)
        {
            auto row = tokenizer.event2tokens(kv.first, kv.second);
            if (!row.empty()) grid.push_back(std::move(row));
        }
    }

    std::mt19937 rng((uint32_t)opt.seed);

    // Working buffers for KV cache. We keep ownership of the byte arrays so
    // that Ort::Value views remain valid across step boundaries.
    auto makeEmptyKV = [&](const ModelMeta& m, std::vector<std::vector<float>>& storage,
                           std::vector<Ort::Value>& values, int64_t batch = 1)
    {
        storage.clear();
        values.clear();
        storage.reserve(2 * (size_t)m.numLayers);
        values.reserve(2 * (size_t)m.numLayers);
        std::vector<int64_t> shape{ batch, (int64_t)m.numHeads, 0, (int64_t)m.headDim };
        for (int li = 0; li < m.numLayers; ++li)
        {
            for (int kv = 0; kv < 2; ++kv)
            {
                storage.emplace_back();
                Ort::Value v = makeFloatTensor(memInfo, storage.back(), shape);
                values.push_back(std::move(v));
            }
        }
    };

    std::vector<std::vector<float>> baseKvStorage;
    std::vector<Ort::Value> baseKvValues;
    makeEmptyKV(baseMeta, baseKvStorage, baseKvValues);

    int curLen = (int)grid.size();
    int pastLen = 0;

    while (curLen < opt.maxLen)
    {
        if (opt.abortFlag && opt.abortFlag->load()) { errorOut = "aborted"; return false; }

        // ---- model_base forward ----
        int newSteps = curLen - pastLen;
        std::vector<int64_t> xData((size_t)newSteps * (size_t)maxTokenSeq);
        for (int s = 0; s < newSteps; ++s)
        {
            const auto& row = grid[pastLen + s];
            for (int t = 0; t < maxTokenSeq; ++t) xData[s * maxTokenSeq + t] = row[t];
        }
        std::vector<int64_t> xShape{ 1, newSteps, maxTokenSeq };
        Ort::Value xVal = Ort::Value::CreateTensor<int64_t>(memInfo, xData.data(), xData.size(),
                                                             xShape.data(), xShape.size());

        std::vector<const char*> inNames(baseMeta.inputNames.size());
        for (size_t i = 0; i < baseMeta.inputNames.size(); ++i) inNames[i] = baseMeta.inputNames[i].c_str();
        std::vector<const char*> outNames(baseMeta.outputNames.size());
        for (size_t i = 0; i < baseMeta.outputNames.size(); ++i) outNames[i] = baseMeta.outputNames[i].c_str();

        std::vector<Ort::Value> baseInputs;
        baseInputs.reserve(baseMeta.inputNames.size());
        for (size_t i = 0; i < baseMeta.inputNames.size(); ++i)
        {
            if ((int)i == baseMeta.xInputIdx) baseInputs.push_back(std::move(xVal));
            else
            {
                // Find the layer/kind for this past_key_values.<L>.<key|value> input
                int li = -1;
                for (size_t k = 0; k < baseMeta.pastKeyInputIdx.size(); ++k)
                    if (baseMeta.pastKeyInputIdx[k] == (int)i) { li = (int)(2 * k); break; }
                if (li < 0)
                    for (size_t k = 0; k < baseMeta.pastValueInputIdx.size(); ++k)
                        if (baseMeta.pastValueInputIdx[k] == (int)i) { li = (int)(2 * k + 1); break; }
                if (li < 0) { errorOut = "missing KV input"; return false; }
                baseInputs.push_back(std::move(baseKvValues[li]));
            }
        }

        std::vector<Ort::Value> baseOutputs;
        try
        {
            baseOutputs = baseSession->Run(Ort::RunOptions{nullptr},
                                           inNames.data(), baseInputs.data(), baseInputs.size(),
                                           outNames.data(), outNames.size());
        }
        catch (const Ort::Exception& e)
        {
            errorOut = juce::String("base.Run: ") + e.what();
            return false;
        }

        // Re-collect new KV (present.*) into baseKvValues, and grab the last hidden.
        baseKvValues.clear();
        baseKvValues.reserve((size_t)(2 * baseMeta.numLayers));
        for (int li = 0; li < baseMeta.numLayers; ++li)
        {
            int outKey = baseMeta.presentKeyOutputIdx[li];
            int outVal = baseMeta.presentValueOutputIdx[li];
            baseKvValues.push_back(std::move(baseOutputs[outKey]));
            baseKvValues.push_back(std::move(baseOutputs[outVal]));
        }
        Ort::Value& hiddenAll = baseOutputs[baseMeta.hiddenOutputIdx];
        auto hiddenShape = hiddenAll.GetTensorTypeAndShapeInfo().GetShape();
        if (hiddenShape.size() != 3) { errorOut = "unexpected hidden shape"; return false; }
        const int embDim = (int)hiddenShape[2];
        if (baseMeta.embDim == 0) baseMeta.embDim = embDim;
        const float* hiddenData = hiddenAll.GetTensorData<float>();
        std::vector<float> lastHidden((size_t)embDim);
        // last step (along axis 1)
        const int hiddenSeq = (int)hiddenShape[1];
        std::memcpy(lastHidden.data(),
                    hiddenData + (size_t)(hiddenSeq - 1) * (size_t)embDim,
                    sizeof(float) * (size_t)embDim);

        // ---- model_token inner loop ----
        std::vector<std::vector<float>> tokKvStorage;
        std::vector<Ort::Value> tokKvValues;
        makeEmptyKV(tokenMeta, tokKvStorage, tokKvValues);

        std::vector<int> nextRow;
        nextRow.reserve(maxTokenSeq);
        bool ended = false;
        juce::String eventName;

        // current "hidden" for the token model. shape: (1, hidden_seq, emb_dim)
        std::vector<float> hiddenBuf = lastHidden;
        int hiddenSeqLen = 1;

        std::vector<const char*> tInNames(tokenMeta.inputNames.size());
        for (size_t i = 0; i < tokenMeta.inputNames.size(); ++i) tInNames[i] = tokenMeta.inputNames[i].c_str();
        std::vector<const char*> tOutNames(tokenMeta.outputNames.size());
        for (size_t i = 0; i < tokenMeta.outputNames.size(); ++i) tOutNames[i] = tokenMeta.outputNames[i].c_str();

        for (int i = 0; i < maxTokenSeq; ++i)
        {
            // Build mask for this position.
            std::vector<int> mask(vocab, 0);
            if (ended)
            {
                mask[padId] = 1;
            }
            else if (i == 0)
            {
                for (int eid : tokenizer.getEventIdsList())
                {
                    juce::String nm = tokenizer.eventNameFromId(eid);
                    if (opt.disablePatchChange  && nm == "patch_change")  continue;
                    if (opt.disableControlChange && nm == "control_change") continue;
                    mask[eid] = 1;
                }
                mask[eosId] = 1;
            }
            else
            {
                const auto& pn = tokenizer.paramNamesFor(eventName);
                if ((int)i > (int)pn.size()) mask[padId] = 1;
                else
                {
                    juce::String paramName = pn[i - 1];
                    const auto& ids = tokenizer.parameterIdsFor(paramName);
                    if (paramName == "channel" && !opt.disableChannels.empty())
                    {
                        for (int id : ids)
                        {
                            int channel = id - ids.front();
                            bool blocked = false;
                            for (int dc : opt.disableChannels) if (dc == channel) { blocked = true; break; }
                            if (!blocked) mask[id] = 1;
                        }
                    }
                    else
                    {
                        for (int id : ids) mask[id] = 1;
                    }
                }
            }

            // Inputs:  hidden=(1, hiddenSeqLen, embDim), x=(1, xLen)
            std::vector<int64_t> hiddenShapeIn{ 1, hiddenSeqLen, (int64_t)embDim };
            Ort::Value hiddenVal = Ort::Value::CreateTensor<float>(
                memInfo, hiddenBuf.data(), hiddenBuf.size(),
                hiddenShapeIn.data(), hiddenShapeIn.size());

            int xLen = (i == 0) ? 0 : 1;
            int64_t xLast = (i == 0) ? 0 : (int64_t)nextRow.back();
            std::vector<int64_t> xShapeT{ 1, xLen };
            // Always allocate a valid backing pointer; ORT can still see element_count=0.
            std::vector<int64_t> xBuf(1, xLast);
            Ort::Value xValT = Ort::Value::CreateTensor<int64_t>(
                memInfo, xBuf.data(), (size_t)xLen,
                xShapeT.data(), xShapeT.size());

            std::vector<Ort::Value> tIn;
            tIn.reserve(tokenMeta.inputNames.size());
            for (size_t k = 0; k < tokenMeta.inputNames.size(); ++k)
            {
                if ((int)k == tokenMeta.hiddenInputIdx) tIn.push_back(std::move(hiddenVal));
                else if ((int)k == tokenMeta.xInputIdx) tIn.push_back(std::move(xValT));
                else
                {
                    int li = -1;
                    for (size_t kk = 0; kk < tokenMeta.pastKeyInputIdx.size(); ++kk)
                        if (tokenMeta.pastKeyInputIdx[kk] == (int)k) { li = (int)(2 * kk); break; }
                    if (li < 0)
                        for (size_t kk = 0; kk < tokenMeta.pastValueInputIdx.size(); ++kk)
                            if (tokenMeta.pastValueInputIdx[kk] == (int)k) { li = (int)(2 * kk + 1); break; }
                    if (li < 0) { errorOut = "missing token KV input"; return false; }
                    tIn.push_back(std::move(tokKvValues[li]));
                }
            }

            std::vector<Ort::Value> tOut;
            try
            {
                tOut = tokenSession->Run(Ort::RunOptions{nullptr},
                                          tInNames.data(), tIn.data(), tIn.size(),
                                          tOutNames.data(), tOutNames.size());
            }
            catch (const Ort::Exception& e)
            {
                errorOut = juce::String("token.Run: ") + e.what();
                return false;
            }

            tokKvValues.clear();
            tokKvValues.reserve((size_t)(2 * tokenMeta.numLayers));
            for (int li = 0; li < tokenMeta.numLayers; ++li)
            {
                tokKvValues.push_back(std::move(tOut[tokenMeta.presentKeyOutputIdx[li]]));
                tokKvValues.push_back(std::move(tOut[tokenMeta.presentValueOutputIdx[li]]));
            }

            Ort::Value& yAll = tOut[tokenMeta.yOutputIdx];
            auto yShape = yAll.GetTensorTypeAndShapeInfo().GetShape();
            // yAll: (1, token_seq1, vocab). Take the last position.
            const int ySeq = (int)yShape[1];
            const float* yData = yAll.GetTensorData<float>();
            std::vector<float> logits((size_t)vocab);
            std::memcpy(logits.data(),
                        yData + (size_t)(ySeq - 1) * (size_t)vocab,
                        sizeof(float) * (size_t)vocab);

            float invT = 1.0f / std::max(1e-3f, opt.temperature);
            for (auto& v : logits) v *= invT;
            // Apply mask BEFORE softmax to fully zero out forbidden tokens.
            for (int v = 0; v < vocab; ++v) if (!mask[v]) logits[v] = -1e9f;
            softmaxLastAxis(logits.data(), 1, vocab);

            int sampled = sampleTopPK(logits.data(), vocab, opt.topP, opt.topK, rng);
            nextRow.push_back(sampled);

            if (i == 0)
            {
                if (sampled == eosId) ended = true;
                else
                {
                    eventName = tokenizer.eventNameFromId(sampled);
                    if (eventName.isEmpty()) ended = true;
                }
                // After step 0 the model expects an empty hidden tensor; we still
                // need a non-null backing pointer for ORT.
                hiddenBuf.assign(1, 0.0f);
                hiddenSeqLen = 0;
            }
            else
            {
                if (!ended)
                {
                    int paramCount = (int)tokenizer.paramNamesFor(eventName).size();
                    if (i == paramCount) break;
                }
            }
        }

        while ((int)nextRow.size() < maxTokenSeq) nextRow.push_back(padId);
        grid.push_back(std::move(nextRow));

        pastLen = curLen;
        ++curLen;

        if (ended) break;
    }

    outGrid = std::move(grid);
    return true;
}

#endif // MIDI_GEN_USE_ONNX

} // namespace skytnt
