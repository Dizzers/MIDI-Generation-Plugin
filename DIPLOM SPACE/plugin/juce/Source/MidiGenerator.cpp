#include "MidiGenerator.h"

std::unordered_map<int, juce::String> MidiGenerator::idToToken;
bool MidiGenerator::vocabLoaded = false;

static constexpr double kTimeShiftResolutionSeconds = 0.05; // must match dataset/tokenize_midi.py

std::vector<juce::MidiMessage> MidiGenerator::convertTokensToMidi(const std::vector<int>& tokenIds)
{
    std::vector<juce::MidiMessage> midiEvents;

    if (!ensureVocabLoaded())
    {
        DBG("MidiGenerator: vocab.json not loaded, returning empty MIDI");
        return midiEvents;
    }

    double currentTimeSeconds = 0.0;
    int currentVelocity = 100;
    const int midiChannel = 1;

    for (const int id : tokenIds)
    {
        juce::String token;
        if (!tokenIdToString(id, token))
            continue;

        // Skip special/meta tokens
        if (token.startsWithChar('<'))
        {
            if (token == "<EOS>")
                break;
            continue;
        }

        if (token.startsWith("TIME_SHIFT_"))
        {
            int steps = 0;
            if (parseHexSuffix(token, steps))
                currentTimeSeconds += (double)steps * kTimeShiftResolutionSeconds;
            continue;
        }

        if (token.startsWith("VELOCITY_"))
        {
            int velBin = 0;
            if (parseHexSuffix(token, velBin))
            {
                // tokenizer uses 8 bins; map [0..7] -> [1..127] roughly
                const int bins = 8;
                velBin = juce::jlimit(0, bins - 1, velBin);
                currentVelocity = juce::jlimit(1, 127, (int)std::lround((velBin / (double)(bins - 1)) * 127.0));
            }
            continue;
        }

        if (token.startsWith("NOTE_ON_"))
        {
            int pitch = 0;
            if (!parseHexSuffix(token, pitch))
                continue;
            pitch = juce::jlimit(0, 127, pitch);
            auto msg = juce::MidiMessage::noteOn(midiChannel, pitch, (juce::uint8)currentVelocity);
            msg.setTimeStamp(currentTimeSeconds);
            midiEvents.push_back(msg);
            continue;
        }

        if (token.startsWith("NOTE_OFF_"))
        {
            int pitch = 0;
            if (!parseHexSuffix(token, pitch))
                continue;
            pitch = juce::jlimit(0, 127, pitch);
            auto msg = juce::MidiMessage::noteOff(midiChannel, pitch);
            msg.setTimeStamp(currentTimeSeconds);
            midiEvents.push_back(msg);
            continue;
        }
    }

    return midiEvents;
}

juce::File MidiGenerator::findVocabJson()
{
    // Typical locations we try, in priority order.
    // 1) CWD/bin/vocab.json (dev runs)
    // 2) Executable parent chains (bundle/build artefacts)
    // 3) User documents fallback (manual copy)

    const auto tryPath = [](const juce::File& base) -> juce::File
    {
        auto f = base.getChildFile("bin").getChildFile("vocab.json");
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

    {
        auto docs = juce::File::getSpecialLocation(juce::File::userDocumentsDirectory);
        auto f = docs.getChildFile("MIDI-Generation-Plugin").getChildFile("vocab.json");
        if (f.existsAsFile())
            return f;
    }

    return {};
}

bool MidiGenerator::ensureVocabLoaded()
{
    if (vocabLoaded)
        return true;

    const auto vocabPath = findVocabJson();
    if (!vocabPath.existsAsFile())
    {
        DBG("MidiGenerator: vocab.json not found");
        return false;
    }

    juce::String jsonText = vocabPath.loadFileAsString();
    auto parsed = juce::JSON::parse(jsonText);
    if (parsed.isVoid() || !parsed.isObject())
    {
        DBG("MidiGenerator: failed to parse vocab.json");
        return false;
    }

    auto* obj = parsed.getDynamicObject();
    if (obj == nullptr || !obj->hasProperty("id2token"))
    {
        DBG("MidiGenerator: vocab.json missing id2token");
        return false;
    }

    auto id2tokVar = obj->getProperty("id2token");
    if (!id2tokVar.isObject())
    {
        DBG("MidiGenerator: id2token is not an object");
        return false;
    }

    auto* id2tokObj = id2tokVar.getDynamicObject();
    if (id2tokObj == nullptr)
        return false;

    idToToken.clear();
    const auto& props = id2tokObj->getProperties();
    for (const auto& entry : props)
    {
        // JSON keys are strings; convert to int.
        const int id = entry.name.toString().getIntValue();
        const juce::String tok = entry.value.toString();
        idToToken.emplace(id, tok);
    }

    vocabLoaded = !idToToken.empty();
    DBG("MidiGenerator: loaded vocab entries=" << (int)idToToken.size());
    return vocabLoaded;
}

bool MidiGenerator::tokenIdToString(int id, juce::String& tokenName)
{
    if (!ensureVocabLoaded())
        return false;

    auto it = idToToken.find(id);
    if (it == idToToken.end())
        return false;

    tokenName = it->second;
    return true;
}

bool MidiGenerator::parseHexSuffix(const juce::String& token, int& valueOut)
{
    const int underscore = token.lastIndexOfChar('_');
    if (underscore < 0)
        return false;

    auto hex = token.substring(underscore + 1).trim();
    if (hex.startsWithIgnoreCase("0x"))
        hex = hex.substring(2);

    if (hex.isEmpty())
        return false;

    valueOut = (int)hex.getHexValue32();
    return true;
}
