#include "ModelInference.h"
#include "SkyTNTRuntime.h"

#include <utility>

namespace
{
struct KeySpec { const char* name; int sf; int mi; };

const KeySpec kKeyTable[] = {
    {"C_MAJOR", 0, 0},   {"G_MAJOR", 1, 0},   {"D_MAJOR", 2, 0},  {"A_MAJOR", 3, 0},
    {"E_MAJOR", 4, 0},   {"B_MAJOR", 5, 0},   {"F_SHARP_MAJOR", 6, 0},
    {"C_SHARP_MAJOR", 7, 0},
    {"F_MAJOR", -1, 0},  {"B_FLAT_MAJOR", -2, 0}, {"E_FLAT_MAJOR", -3, 0},
    {"A_FLAT_MAJOR", -4, 0},
    {"A_MINOR", 0, 1},   {"E_MINOR", 1, 1},   {"B_MINOR", 2, 1},
    {"F_SHARP_MINOR", 3, 1}, {"C_SHARP_MINOR", 4, 1},
    {"G_SHARP_MINOR", 5, 1}, {"D_SHARP_MINOR", 6, 1}, {"A_SHARP_MINOR", 7, 1},
    {"D_MINOR", -1, 1},  {"G_MINOR", -2, 1},  {"C_MINOR", -3, 1},  {"F_MINOR", -4, 1}
};

bool keyToSfMi(const juce::String& key, int& sf, int& mi)
{
    for (const auto& k : kKeyTable)
        if (key == k.name) { sf = k.sf; mi = k.mi; return true; }
    return false;
}
} // namespace

ModelInference::ModelInference()
    : runtime(std::make_unique<skytnt::Runtime>())
{
    tryLoadFromDefaultLocations();
}

ModelInference::~ModelInference() = default;

juce::File ModelInference::findArtifact(const char* relativePath)
{
    // Search up to project root for an artifacts/ folder. The plugin can also
    // override locations via env vars (handy on dev boxes).
    auto envOverride = juce::SystemStats::getEnvironmentVariable("MIDIGEN_ARTIFACT_DIR", {});
    if (envOverride.isNotEmpty())
    {
        juce::File f = juce::File(envOverride).getChildFile(relativePath);
        if (f.existsAsFile()) return f;
    }
    const std::vector<juce::File> roots {
        juce::File::getSpecialLocation(juce::File::currentExecutableFile).getParentDirectory(),
        juce::File::getSpecialLocation(juce::File::currentApplicationFile).getParentDirectory(),
        juce::File::getCurrentWorkingDirectory()
    };
    for (auto root : roots)
    {
        for (int i = 0; i < 8 && root.exists(); ++i)
        {
            juce::File candidate = root.getChildFile("artifacts").getChildFile(relativePath);
            if (candidate.existsAsFile()) return candidate;
            candidate = root.getChildFile("plugin/juce/bin").getChildFile(relativePath);
            if (candidate.existsAsFile()) return candidate;
            candidate = root.getChildFile("bin").getChildFile(relativePath);
            if (candidate.existsAsFile()) return candidate;
            root = root.getParentDirectory();
        }
    }
    return {};
}

void ModelInference::tryLoadFromDefaultLocations()
{
    juce::File baseFile  = findArtifact("onnx/model_base.onnx");
    juce::File tokenFile = findArtifact("onnx/model_token.onnx");
    juce::File tokFile   = findArtifact("tokenizer/tokenizer_config.json");

    if (!baseFile.existsAsFile() || !tokenFile.existsAsFile())
    {
        statusText = "ONNX models not found. Drop model_base.onnx + model_token.onnx into artifacts/onnx/";
        return;
    }

    juce::String err;
    if (!runtime->loadModels(baseFile, tokenFile, tokFile, err))
        statusText = "Load failed: " + err;
    else
        statusText = runtime->getStatus();
}

bool ModelInference::isLoaded() const
{
    return runtime && runtime->isLoaded();
}

juce::String ModelInference::getStatusText() const
{
    if (runtime && runtime->isLoaded()) return runtime->getStatus();
    return statusText;
}

ModelInference::GenerationResult ModelInference::generateMidi(const GenerationParams& p)
{
    GenerationResult r;
    if (!runtime || !runtime->isLoaded())
    {
        r.errorMessage = "Model not loaded: " + getStatusText().toStdString();
        return r;
    }

    abortFlag.store(false);

    skytnt::Runtime::GenerateOptions opt;
    opt.maxLen = juce::jmax(8, p.maxLen);
    opt.temperature = juce::jlimit(0.05f, 4.0f, p.temperature);
    opt.topP = juce::jlimit(0.05f, 1.0f, p.topP);
    opt.topK = juce::jlimit(1, 1024, p.topK);
    opt.seed = (uint32_t) p.seed;
    opt.disablePatchChange = p.disablePatchChange;
    opt.disableControlChange = p.disableControlChange;
    opt.abortFlag = &abortFlag;

    // Setup events: tempo and (V2 only) key signature.
    if (p.bpm > 0.0f)
    {
        int bpm = juce::jlimit(1, 383, (int) std::round(p.bpm));
        // event order: time1, time2, track, bpm
        opt.setupEvents.emplace_back("set_tempo", std::vector<int>{ 0, 0, 0, bpm });
    }
    int sf = 0, mi = 0;
    if (keyToSfMi(p.key, sf, mi))
    {
        if (runtime->getTokenizer().getVersion() == "v2")
        {
            opt.setupEvents.emplace_back(
                "key_signature",
                std::vector<int>{ 0, 0, 0, sf + 7, mi });
        }
    }
    // Default acoustic-grand patch on channel 0 so the stream produces audible piano.
    opt.setupEvents.emplace_back("patch_change", std::vector<int>{ 0, 0, 0, 0, 0 });

    std::vector<std::vector<int>> grid;
    juce::String err;
    if (!runtime->generate(opt, grid, err))
    {
        r.errorMessage = err.toStdString();
        return r;
    }

    auto sequence = runtime->getTokenizer().detokenize(grid, p.bpm);
    r.messages.reserve((size_t) sequence.getNumEvents());
    for (int i = 0; i < sequence.getNumEvents(); ++i)
        r.messages.push_back(sequence.getEventPointer(i)->message);
    r.success = true;
    return r;
}
