#pragma once
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <vector>

/**
 * Piano roll visualizer for MIDI events
 * Displays notes as rectangles on a time/pitch grid
 */
class MidiVisualizer : public juce::Component
{
public:
    MidiVisualizer();
    ~MidiVisualizer() override;

    void paint(juce::Graphics& g) override;
    void resized() override;
    void mouseWheelMove(const juce::MouseEvent& e, const juce::MouseWheelDetails& wheel) override;
    void mouseDown(const juce::MouseEvent& e) override;
    void mouseDrag(const juce::MouseEvent& e) override;
    void mouseDoubleClick(const juce::MouseEvent& e) override;
    void setMidiMessages(const std::vector<juce::MidiMessage>& messages);
    void setBpm(float bpm);

private:
    struct NoteRect
    {
        int pitch = 60;
        double start = 0.0;
        double end = 0.0;
        int velocity = 100;
    };

    void rebuildNotes();
    double getMaxTimeSeconds() const;
    int getMinPitch() const;
    int getMaxPitch() const;
    juce::Rectangle<int> getDrawArea() const;

    void clampScroll();

    std::vector<juce::MidiMessage> midiMessages;
    std::vector<NoteRect> notes;

    // View state
    float zoomX = 1.0f;        // 1 = fit; >1 = zoom in
    float scrollX = 0.0f;      // pixels
    float scrollY = 0.0f;      // pixels
    juce::Point<int> dragAnchor;
    juce::Point<float> dragScrollAnchor;

    float bpm = 120.0f;
};
