#include "MidiFileExporter.h"

bool MidiFileExporter::saveMidiFile(
    const juce::File& targetFile,
    const std::vector<juce::MidiMessage>& midiMessages,
    double tempo)
{
    if (midiMessages.empty())
        return false;

    const int ppq = 960;
    const double bpm = (tempo > 1e-3) ? tempo : 120.0;
    const double ticksPerSecond = (bpm / 60.0) * (double)ppq;

    juce::MidiMessageSequence seq;
    seq.ensureStorageAllocated((int)midiMessages.size());

    for (auto msg : midiMessages)
    {
        // Our messages use timeStamp in seconds; convert to ticks for the MIDI file.
        const double seconds = msg.getTimeStamp();
        msg.setTimeStamp(seconds * ticksPerSecond);
        seq.addEvent(msg);
    }
    seq.updateMatchedPairs();

    juce::MidiFile mf;
    mf.setTicksPerQuarterNote(ppq);
    mf.addTrack(seq);

    targetFile.getParentDirectory().createDirectory();
    std::unique_ptr<juce::FileOutputStream> out(targetFile.createOutputStream());
    if (out == nullptr || !out->openedOk())
        return false;

    const bool ok = mf.writeTo(*out);
    out->flush();
    return ok;
}
