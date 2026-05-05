#include "SkyTNTTokenizer.h"

#include <algorithm>
#include <cmath>

namespace skytnt
{

const std::vector<juce::String> Tokenizer::emptyParamList;
const std::vector<int> Tokenizer::emptyIdList;

namespace
{

// V1 / V2 tables hardcoded so we can rebuild without the JSON dump.
struct TokenizerSpec
{
    std::vector<std::pair<juce::String, std::vector<juce::String>>> events; // ordered
    std::vector<std::pair<juce::String, int>> eventParameters;              // ordered
};

TokenizerSpec buildSpecV1()
{
    TokenizerSpec s;
    s.events = {
        {"note",          {"time1", "time2", "track", "duration", "channel", "pitch", "velocity"}},
        {"patch_change",  {"time1", "time2", "track", "channel", "patch"}},
        {"control_change",{"time1", "time2", "track", "channel", "controller", "value"}},
        {"set_tempo",     {"time1", "time2", "track", "bpm"}},
    };
    s.eventParameters = {
        {"time1", 128}, {"time2", 16}, {"duration", 2048}, {"track", 128},
        {"channel", 16}, {"pitch", 128}, {"velocity", 128}, {"patch", 128},
        {"controller", 128}, {"value", 128}, {"bpm", 256},
    };
    return s;
}

TokenizerSpec buildSpecV2()
{
    TokenizerSpec s;
    s.events = {
        {"note",          {"time1", "time2", "track", "channel", "pitch", "velocity", "duration"}},
        {"patch_change",  {"time1", "time2", "track", "channel", "patch"}},
        {"control_change",{"time1", "time2", "track", "channel", "controller", "value"}},
        {"set_tempo",     {"time1", "time2", "track", "bpm"}},
        {"time_signature",{"time1", "time2", "track", "nn", "dd"}},
        {"key_signature", {"time1", "time2", "track", "sf", "mi"}},
    };
    s.eventParameters = {
        {"time1", 128}, {"time2", 16}, {"duration", 2048}, {"track", 128},
        {"channel", 16}, {"pitch", 128}, {"velocity", 128}, {"patch", 128},
        {"controller", 128}, {"value", 128}, {"bpm", 384},
        {"nn", 16}, {"dd", 4}, {"sf", 15}, {"mi", 2},
    };
    return s;
}

} // namespace

bool Tokenizer::buildFromVersion(const juce::String& v, bool opt)
{
    version = v;
    optimiseMidi = opt;

    TokenizerSpec spec = (v == "v1") ? buildSpecV1() : buildSpecV2();

    // Replace existing tables.
    events.clear();
    paramOrder.clear();
    paramSizes.clear();
    paramIds.clear();
    eventNames.clear();
    eventIdsList.clear();
    eventNameToId.clear();
    idToEventName.clear();

    for (auto& ev : spec.events)
    {
        events[ev.first.toStdString()] = ev.second;
        eventNames.push_back(ev.first);
    }
    for (auto& pp : spec.eventParameters)
    {
        paramOrder.push_back(pp.first);
        paramSizes[pp.first.toStdString()] = pp.second;
    }

    rebuildFromEventsTable();
    loaded = true;
    return true;
}

bool Tokenizer::loadFromJsonFile(const juce::File& path, juce::String& errorOut)
{
    if (!path.existsAsFile())
    {
        errorOut = "tokenizer_config.json not found at " + path.getFullPathName();
        return false;
    }
    juce::var root = juce::JSON::parse(path);
    if (root.isVoid())
    {
        errorOut = "Failed to parse JSON: " + path.getFullPathName();
        return false;
    }
    return buildFromTopLevel(root, errorOut);
}

bool Tokenizer::buildFromTopLevel(const juce::var& root, juce::String& errorOut)
{
    if (!root.isObject())
    {
        errorOut = "tokenizer_config root is not an object";
        return false;
    }
    juce::String v = root.getProperty("version", "v2").toString();
    bool opt = (bool)root.getProperty("optimise_midi", false);
    if (! buildFromVersion(v, opt))
    {
        errorOut = "Unknown tokenizer version: " + v;
        return false;
    }

    // (Optional) cross-check sizes against dumped values.
    if (root.hasProperty("vocab_size"))
    {
        int dumpedVocab = (int)root.getProperty("vocab_size", vocabSize);
        if (dumpedVocab != vocabSize)
        {
            errorOut = "vocab_size mismatch (json=" + juce::String(dumpedVocab)
                     + ", local=" + juce::String(vocabSize) + ")";
            return false;
        }
    }
    return true;
}

void Tokenizer::rebuildFromEventsTable()
{
    int next = 0;
    padId = next++;
    bosId = next++;
    eosId = next++;

    eventIdsList.clear();
    eventIdsList.reserve(eventNames.size());
    eventNameToId.clear();
    idToEventName.clear();
    for (const auto& n : eventNames)
    {
        eventIdsList.push_back(next);
        eventNameToId[n.toStdString()] = next;
        idToEventName[next] = n.toStdString();
        ++next;
    }

    paramIds.clear();
    for (const auto& pn : paramOrder)
    {
        int sz = paramSizes[pn.toStdString()];
        std::vector<int> ids(sz);
        for (int i = 0; i < sz; ++i)
            ids[i] = next + i;
        paramIds[pn.toStdString()] = ids;
        next += sz;
    }
    vocabSize = next;

    int maxLen = 0;
    for (const auto& kv : events)
        maxLen = std::max(maxLen, (int)kv.second.size());
    maxTokenSeq = maxLen + 1;
}

int Tokenizer::eventIdFor(const juce::String& name) const
{
    auto it = eventNameToId.find(name.toStdString());
    return (it == eventNameToId.end()) ? -1 : it->second;
}

const std::vector<juce::String>& Tokenizer::paramNamesFor(const juce::String& name) const
{
    auto it = events.find(name.toStdString());
    return (it == events.end()) ? emptyParamList : it->second;
}

const std::vector<int>& Tokenizer::parameterIdsFor(const juce::String& name) const
{
    auto it = paramIds.find(name.toStdString());
    return (it == paramIds.end()) ? emptyIdList : it->second;
}

int Tokenizer::parameterSize(const juce::String& name) const
{
    auto it = paramSizes.find(name.toStdString());
    return (it == paramSizes.end()) ? 0 : it->second;
}

juce::String Tokenizer::eventNameFromId(int tokenId) const
{
    auto it = idToEventName.find(tokenId);
    return (it == idToEventName.end()) ? juce::String() : juce::String(it->second);
}

std::vector<int> Tokenizer::event2tokens(const juce::String& name,
                                         const std::vector<int>& params) const
{
    const auto& pn = paramNamesFor(name);
    if (pn.empty() || params.size() != pn.size()) return {};
    std::vector<int> tokens;
    tokens.reserve(maxTokenSeq);
    int evId = eventIdFor(name);
    if (evId < 0) return {};
    tokens.push_back(evId);
    for (size_t i = 0; i < pn.size(); ++i)
    {
        int sz = parameterSize(pn[i]);
        if (params[i] < 0 || params[i] >= sz) return {};
        const auto& ids = parameterIdsFor(pn[i]);
        tokens.push_back(ids[params[i]]);
    }
    while ((int)tokens.size() < maxTokenSeq) tokens.push_back(padId);
    return tokens;
}

std::vector<int> Tokenizer::event2tokensFromList(const std::vector<int>& packed,
                                                 const juce::String& name) const
{
    return event2tokens(name, packed);
}

bool Tokenizer::tokens2event(const std::vector<int>& tokens, ScoreEvent& out) const
{
    if (tokens.empty()) return false;
    auto it = idToEventName.find(tokens[0]);
    if (it == idToEventName.end()) return false;
    juce::String name = juce::String(it->second);
    const auto& pn = paramNamesFor(name);
    if ((int)tokens.size() <= (int)pn.size()) return false;

    std::vector<int> p(pn.size(), 0);
    for (size_t i = 0; i < pn.size(); ++i)
    {
        const auto& ids = parameterIdsFor(pn[i]);
        if (ids.empty()) return false;
        int raw = tokens[i + 1] - ids[0];
        int sz = parameterSize(pn[i]);
        if (raw < 0 || raw >= sz) return false;
        p[i] = raw;
    }

    out.name = name;
    out.timeTicks = 0; // filled in detokenize()
    if (name == "note")
    {
        if (version == "v1")
        {
            // ["time1","time2","track","duration","channel","pitch","velocity"]
            out.track    = p[2];
            out.duration = p[3];
            out.channel  = p[4];
            out.pitch    = p[5];
            out.velocity = p[6];
        }
        else
        {
            // V2: ["time1","time2","track","channel","pitch","velocity","duration"]
            out.track    = p[2];
            out.channel  = p[3];
            out.pitch    = p[4];
            out.velocity = p[5];
            out.duration = p[6];
        }
    }
    else if (name == "patch_change")
    {
        out.track   = p[2];
        out.channel = p[3];
        out.patch   = p[4];
    }
    else if (name == "control_change")
    {
        out.track      = p[2];
        out.channel    = p[3];
        out.controller = p[4];
        out.value      = p[5];
    }
    else if (name == "set_tempo")
    {
        out.track = p[2];
        int bpm   = p[3];
        if (bpm == 0) bpm = 1;
        out.tempoUs = (int)(60000000.0 / (double)bpm);
    }
    else if (name == "time_signature")
    {
        out.track = p[2];
        out.nn = p[3] + 1;
        out.dd = p[4] + 1;
    }
    else if (name == "key_signature")
    {
        out.track = p[2];
        out.sf = p[3] - 7; // shift back
        out.mi = p[4];
    }
    else
    {
        return false;
    }
    return true;
}

juce::MidiMessageSequence Tokenizer::detokenize(
    const std::vector<std::vector<int>>& midiSeq, double bpmHint) const
{
    constexpr int ticksPerBeat = 480;

    struct Item
    {
        ScoreEvent ev;
        long long absTicks = 0;
    };
    std::unordered_map<int, std::vector<Item>> tracks;

    long long t1Acc = 0;
    for (const auto& tokens : midiSeq)
    {
        ScoreEvent ev;
        if (!tokens2event(tokens, ev)) continue;

        // Reconstruct absolute time from t1 (delta) + t2 (16ths within beat).
        const juce::String& name = ev.name;
        auto it = events.find(name.toStdString());
        if (it == events.end()) continue;
        const auto& pn = it->second;
        const auto& tokensVec = tokens;

        int rawT1 = tokensVec[1] - parameterIdsFor("time1")[0];
        int rawT2 = tokensVec[2] - parameterIdsFor("time2")[0];
        if (rawT1 < 0) rawT1 = 0;
        if (rawT2 < 0) rawT2 = 0;
        t1Acc += rawT1;
        long long t16 = t1Acc * 16 + rawT2;
        long long absTicks = (long long) std::llround((double)t16 * ticksPerBeat / 16.0);
        ev.timeTicks = absTicks;
        if (name == "note")
            ev.duration = (int)std::llround((double)ev.duration * ticksPerBeat / 16.0);

        Item it2{ ev, absTicks };
        tracks[ev.track].push_back(it2);
    }

    // Build a global, time-sorted sequence and produce MIDI messages with seconds.
    std::vector<Item> all;
    all.reserve(64);
    for (auto& kv : tracks)
        for (auto& it : kv.second) all.push_back(it);
    std::sort(all.begin(), all.end(), [](const Item& a, const Item& b)
    {
        if (a.absTicks != b.absTicks) return a.absTicks < b.absTicks;
        // notes after meta to mirror Python event_name_order
        auto ord = [](const juce::String& s) -> int
        {
            if (s == "time_signature") return 0;
            if (s == "key_signature")  return 1;
            if (s == "set_tempo")      return 2;
            if (s == "patch_change")   return 3;
            if (s == "control_change") return 4;
            if (s == "note")           return 5;
            return 6;
        };
        return ord(a.ev.name) < ord(b.ev.name);
    });

    juce::MidiMessageSequence outSeq;

    // Walk events keeping a running tempoUs to convert ticks -> seconds.
    double tempoUs = 60000000.0 / juce::jmax(1.0, (double)bpmHint);
    long long lastTicks = 0;
    double curSeconds = 0.0;
    auto ticksToSeconds = [&] (long long absTicks) -> double
    {
        long long delta = absTicks - lastTicks;
        if (delta < 0) delta = 0;
        double sec = curSeconds + ((double)delta * tempoUs * 1e-6 / (double)ticksPerBeat);
        return sec;
    };
    auto advanceTo = [&] (long long absTicks)
    {
        long long delta = absTicks - lastTicks;
        if (delta < 0) delta = 0;
        curSeconds += ((double)delta * tempoUs * 1e-6 / (double)ticksPerBeat);
        lastTicks = absTicks;
    };

    // We also need to schedule note-offs. Track currently playing notes so we
    // can emit a noteOff when we hit start+duration.
    struct PendingOff
    {
        long long offTicks = 0;
        int channel = 0;
        int pitch = 0;
    };
    std::vector<PendingOff> pendingOffs;

    auto flushOffsUpTo = [&] (long long upTo)
    {
        // Emit note-offs whose offTicks <= upTo, in time order.
        std::sort(pendingOffs.begin(), pendingOffs.end(),
                  [](const PendingOff& a, const PendingOff& b){ return a.offTicks < b.offTicks; });
        while (!pendingOffs.empty() && pendingOffs.front().offTicks <= upTo)
        {
            const PendingOff p = pendingOffs.front();
            pendingOffs.erase(pendingOffs.begin());
            advanceTo(p.offTicks);
            auto m = juce::MidiMessage::noteOff(juce::jlimit(1, 16, p.channel + 1),
                                                juce::jlimit(0, 127, p.pitch));
            m.setTimeStamp(curSeconds);
            outSeq.addEvent(m);
        }
    };

    for (const auto& it : all)
    {
        flushOffsUpTo(it.absTicks);
        advanceTo(it.absTicks);

        const auto& ev = it.ev;
        if (ev.name == "set_tempo")
        {
            tempoUs = (double)ev.tempoUs;
            // No JUCE MIDI tempo event needed for processor output; the
            // post-processor / DAW normally drives transport timing.
        }
        else if (ev.name == "patch_change")
        {
            auto m = juce::MidiMessage::programChange(juce::jlimit(1, 16, ev.channel + 1),
                                                      juce::jlimit(0, 127, ev.patch));
            m.setTimeStamp(curSeconds);
            outSeq.addEvent(m);
        }
        else if (ev.name == "control_change")
        {
            auto m = juce::MidiMessage::controllerEvent(juce::jlimit(1, 16, ev.channel + 1),
                                                        juce::jlimit(0, 127, ev.controller),
                                                        juce::jlimit(0, 127, ev.value));
            m.setTimeStamp(curSeconds);
            outSeq.addEvent(m);
        }
        else if (ev.name == "note")
        {
            int chJuce = juce::jlimit(1, 16, ev.channel + 1);
            int pitch  = juce::jlimit(0, 127, ev.pitch);
            int vel    = juce::jlimit(1, 127, ev.velocity);
            auto on    = juce::MidiMessage::noteOn(chJuce, pitch, (juce::uint8) vel);
            on.setTimeStamp(curSeconds);
            outSeq.addEvent(on);
            PendingOff po;
            po.offTicks = it.absTicks + juce::jmax(1, ev.duration);
            po.channel  = ev.channel;
            po.pitch    = pitch;
            pendingOffs.push_back(po);
        }
    }

    // Drain remaining note-offs.
    flushOffsUpTo(std::numeric_limits<long long>::max());
    outSeq.updateMatchedPairs();
    return outSeq;
}

} // namespace skytnt
