#pragma once

#include <juce_core/juce_core.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace skytnt
{

/** A single decoded MIDI event in SkyTNT score-internal form. */
struct ScoreEvent
{
    juce::String name;        // "note", "set_tempo", "patch_change", ...
    int track = 0;
    long long timeTicks = 0;  // absolute MIDI ticks (post detokenize)

    int duration = 0;         // for "note"
    int channel = 0;          // for note/cc/patch
    int pitch = 0;            // for note
    int velocity = 0;         // for note
    int patch = 0;            // for patch_change
    int controller = 0;       // for control_change
    int value = 0;            // for control_change
    int tempoUs = 0;          // for set_tempo (microseconds per quarter note)
    int nn = 4, dd = 2;       // for time_signature (already +1 from raw param)
    int sf = 0, mi = 0;       // for key_signature (sf is post-+(-7) shift)
};

/**
 *  Light-weight C++ port of `MIDITokenizerV1/V2` from SkyTNT/midi-model.
 *
 *  Constructed deterministically from a `tokenizer_config.json` produced by
 *  `skytnt_adapter.export_skytnt_onnx`. Public surface mirrors the parts the
 *  generation loop and detokenizer need:
 *
 *    - vocab metadata (sizes, pad/bos/eos, max_token_seq)
 *    - event_ids / parameter_ids / events / event_parameters tables
 *    - event2tokens
 *    - tokens2event
 *    - detokenize  (token grid -> juce::MidiMessageSequence with seconds)
 *
 *  No tokenize() yet (MIDI -> tokens). For the JUCE plugin MVP we generate
 *  from a learned BOS/setup prompt rather than from arbitrary user MIDI.
 */
class Tokenizer
{
public:
    Tokenizer() = default;

    /** Load tokenizer metadata from the JSON file dumped by Python. */
    bool loadFromJsonFile(const juce::File& path, juce::String& errorOut);

    /** Build deterministically from version (V1/V2). Used as a fallback. */
    bool buildFromVersion(const juce::String& version, bool optimiseMidi = true);

    bool isLoaded() const noexcept { return loaded; }
    const juce::String& getVersion() const noexcept { return version; }

    int getPadId() const noexcept { return padId; }
    int getBosId() const noexcept { return bosId; }
    int getEosId() const noexcept { return eosId; }
    int getVocabSize() const noexcept { return vocabSize; }
    int getMaxTokenSeq() const noexcept { return maxTokenSeq; }

    const std::vector<juce::String>& getEventNames() const noexcept { return eventNames; }
    const std::vector<int>& getEventIdsList() const noexcept { return eventIdsList; }
    int eventIdFor(const juce::String& name) const;
    const std::vector<juce::String>& paramNamesFor(const juce::String& eventName) const;
    const std::vector<int>& parameterIdsFor(const juce::String& paramName) const;
    int parameterSize(const juce::String& paramName) const;
    juce::String eventNameFromId(int tokenId) const;

    /** Encode a single SkyTNT event description into one token row. Returns empty
     *  vector on out-of-range parameters. */
    std::vector<int> event2tokens(const juce::String& name,
                                  const std::vector<int>& params) const;

    /** Decode one token row (length max_token_seq) into a ScoreEvent. Returns
     *  false on malformed/all-PAD rows. */
    bool tokens2event(const std::vector<int>& tokens, ScoreEvent& out) const;

    /** Convert a 2D token grid to a JUCE MidiMessageSequence with absolute
     *  seconds. `bpmHint` is used until the model emits a `set_tempo`. */
    juce::MidiMessageSequence detokenize(
        const std::vector<std::vector<int>>& midiSeq,
        double bpmHint = 120.0) const;

    /** Convenience: the full event2tokens for an event row described as a vector
     *  starting with the event name string. */
    std::vector<int> event2tokensFromList(const std::vector<int>& packed,
                                          const juce::String& name) const;

private:
    bool buildFromTopLevel(const juce::var& root, juce::String& errorOut);
    void rebuildFromEventsTable();

    bool loaded = false;
    juce::String version = "v2";
    bool optimiseMidi = true;

    int padId = 0;
    int bosId = 1;
    int eosId = 2;
    int vocabSize = 0;
    int maxTokenSeq = 0;

    std::vector<juce::String> eventNames;                       // insertion order
    std::vector<int> eventIdsList;                              // matches eventNames

    // events[name] -> ordered list of parameter names
    std::unordered_map<std::string, std::vector<juce::String>> events;

    // event_parameters: insertion-ordered (vector) + lookup
    std::vector<juce::String> paramOrder;
    std::unordered_map<std::string, int> paramSizes;
    std::unordered_map<std::string, std::vector<int>> paramIds;

    std::unordered_map<std::string, int> eventNameToId;
    std::unordered_map<int, std::string> idToEventName;

    static const std::vector<juce::String> emptyParamList;
    static const std::vector<int> emptyIdList;
};

} // namespace skytnt
