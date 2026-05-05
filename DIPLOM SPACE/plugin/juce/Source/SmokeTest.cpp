// Tiny CLI smoke test for the SkyTNT runtime. Builds independently of the
// plugin host so we can verify ONNX inference + detokenization without DAW.
//
//   skytnt_smoke_test --base model_base.onnx --token model_token.onnx \
//                     --tokenizer tokenizer_config.json \
//                     --out out.mid --max-len 128 --bpm 120
//
#include "SkyTNTRuntime.h"

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace
{
struct Args
{
    juce::File base, token, tokenizer, output { "skytnt_sample.mid" };
    int maxLen = 128;
    float bpm = 120.0f;
    float temperature = 1.0f;
    float topP = 0.94f;
    int topK = 20;
    int seed = 42;
    juce::String key = "C_MAJOR";
};

void printUsage()
{
    std::fprintf(stderr,
        "Usage: skytnt_smoke_test --base BASE.onnx --token TOKEN.onnx \\\n"
        "       --tokenizer tokenizer_config.json [--out file.mid] [--max-len N]\n"
        "       [--bpm N] [--temperature F] [--top-p F] [--top-k N] [--seed N]\n"
        "       [--key C_MAJOR]\n");
}

bool parseArgs(int argc, char** argv, Args& a)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string s = argv[i];
        auto next = [&]() -> std::string
        {
            if (i + 1 >= argc) { printUsage(); std::exit(2); }
            return argv[++i];
        };
        if (s == "--base") a.base = juce::File(juce::String(next()));
        else if (s == "--token") a.token = juce::File(juce::String(next()));
        else if (s == "--tokenizer") a.tokenizer = juce::File(juce::String(next()));
        else if (s == "--out") a.output = juce::File(juce::String(next()));
        else if (s == "--max-len") a.maxLen = std::atoi(next().c_str());
        else if (s == "--bpm") a.bpm = (float) std::atof(next().c_str());
        else if (s == "--temperature") a.temperature = (float) std::atof(next().c_str());
        else if (s == "--top-p") a.topP = (float) std::atof(next().c_str());
        else if (s == "--top-k") a.topK = std::atoi(next().c_str());
        else if (s == "--seed") a.seed = std::atoi(next().c_str());
        else if (s == "--key") a.key = juce::String(next());
        else { printUsage(); return false; }
    }
    return a.base.existsAsFile() && a.token.existsAsFile();
}

void writeMidi(const juce::File& path, const juce::MidiMessageSequence& seq, double bpm)
{
    juce::MidiFile mf;
    mf.setTicksPerQuarterNote(480);
    juce::MidiMessageSequence ticked;
    double secondsPerTick = 60.0 / (juce::jmax(1.0, (double)bpm) * 480.0);
    for (int i = 0; i < seq.getNumEvents(); ++i)
    {
        auto* h = seq.getEventPointer(i);
        auto m = h->message;
        m.setTimeStamp(h->message.getTimeStamp() / secondsPerTick); // -> ticks
        ticked.addEvent(m);
    }
    ticked.updateMatchedPairs();
    mf.addTrack(ticked);

    if (path.existsAsFile()) path.deleteFile();
    juce::FileOutputStream os(path);
    if (! os.openedOk()) { std::fprintf(stderr, "failed to open %s\n", path.getFullPathName().toRawUTF8()); return; }
    mf.writeTo(os);
}
} // namespace

int main(int argc, char** argv)
{
    Args args;
    if (!parseArgs(argc, argv, args))
    {
        printUsage();
        return 1;
    }

    skytnt::Runtime rt;
    juce::String err;
    if (! rt.loadModels(args.base, args.token, args.tokenizer, err))
    {
        std::fprintf(stderr, "loadModels failed: %s\n", err.toRawUTF8());
        return 1;
    }
    std::printf("[smoke] %s\n", rt.getStatus().toRawUTF8());

    skytnt::Runtime::GenerateOptions opt;
    opt.maxLen = args.maxLen;
    opt.temperature = args.temperature;
    opt.topP = args.topP;
    opt.topK = args.topK;
    opt.seed = (uint32_t) args.seed;

    if (args.bpm > 0)
        opt.setupEvents.emplace_back(
            "set_tempo", std::vector<int>{ 0, 0, 0, juce::jlimit(1, 383, (int)args.bpm) });
    opt.setupEvents.emplace_back("patch_change", std::vector<int>{ 0, 0, 0, 0, 0 });

    std::vector<std::vector<int>> grid;
    if (! rt.generate(opt, grid, err))
    {
        std::fprintf(stderr, "generate failed: %s\n", err.toRawUTF8());
        return 1;
    }
    std::printf("[smoke] generated %zu midi steps (max_token_seq=%d, vocab=%d)\n",
                grid.size(), rt.getTokenizer().getMaxTokenSeq(), rt.getTokenizer().getVocabSize());
    auto seq = rt.getTokenizer().detokenize(grid, args.bpm);
    std::printf("[smoke] detokenized to %d MIDI events\n", seq.getNumEvents());

    writeMidi(args.output, seq, args.bpm);
    std::printf("[smoke] wrote %s\n", args.output.getFullPathName().toRawUTF8());
    return 0;
}
