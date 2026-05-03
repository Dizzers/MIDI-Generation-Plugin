"""Single source of truth for the vocab/token contract between Python training
pipeline and the existing JUCE C++ plugin (do not change without updating
plugin/juce/Source/ModelInference.cpp and plugin/juce/Source/MidiGenerator.cpp).

Token classes (all hex literals lowercase, no exceptions):
    <PAD>, <BOS>, <EOS>, <UNK>          fixed special tokens, ids 0..3
    <GENRE_*>                           prefix tokens (alphabetical order
                                        defines the model's genre_id index)
    <KEY_{PC}_{MAJ|MIN}>, <KEY_UNKNOWN> 24 + 1 key tokens (Krumhansl)
    NOTE_ON_0xPP / NOTE_OFF_0xPP        pitch in [0..127], 4-char hex (0xPP)
    TIME_SHIFT_0xSSSS                   1..MAX_TIME_STEPS, 6-char hex (0xSSSS),
                                        each step = TIME_SHIFT_RESOLUTION sec
    VELOCITY_0xVV                       0..VELOCITY_BINS-1, 4-char hex (0xVV)

The model is a causal LM with signature
    forward(input_ids: Tensor[B,T] long, genre_id: Tensor[B] long) -> Tensor[B,T,V]

Generation prefix produced by the C++ plugin is exactly:
    [<BOS>, <GENRE_TRAP>, <KEY_*>]
followed by body events.
"""
from __future__ import annotations

import re

TIME_SHIFT_RESOLUTION = 0.05      # seconds per step (matches MidiGenerator.cpp)
MAX_TIME_STEPS = 32               # max steps in a single TIME_SHIFT token (1.6 s)
VELOCITY_BINS = 8                 # 0..7

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

GENRE_TOKENS = ["<GENRE_TRAP>"]

KEY_TOKENS = [
    *(f"<KEY_{pc}_MAJ>" for pc in PITCH_CLASSES),
    *(f"<KEY_{pc}_MIN>" for pc in PITCH_CLASSES),
    "<KEY_UNKNOWN>",
]

NOTE_ON_TOKENS = [f"NOTE_ON_{p:#04x}" for p in range(128)]
NOTE_OFF_TOKENS = [f"NOTE_OFF_{p:#04x}" for p in range(128)]
TIME_SHIFT_TOKENS = [f"TIME_SHIFT_{s:#06x}" for s in range(1, MAX_TIME_STEPS + 1)]
VELOCITY_TOKENS = [f"VELOCITY_{v:#04x}" for v in range(VELOCITY_BINS)]

ALL_TOKENS = (
    SPECIAL_TOKENS
    + GENRE_TOKENS
    + KEY_TOKENS
    + NOTE_ON_TOKENS
    + NOTE_OFF_TOKENS
    + TIME_SHIFT_TOKENS
    + VELOCITY_TOKENS
)

NOTE_ON_RE = re.compile(r"^NOTE_ON_0x[0-9a-f]+$")
NOTE_OFF_RE = re.compile(r"^NOTE_OFF_0x[0-9a-f]+$")
TIME_SHIFT_RE = re.compile(r"^TIME_SHIFT_0x[0-9a-f]+$")
VELOCITY_RE = re.compile(r"^VELOCITY_0x[0-9a-f]+$")
GENRE_RE = re.compile(r"^<GENRE_[A-Z0-9_]+>$")
KEY_RE = re.compile(r"^<KEY_[A-G#]+(?:_MAJ|_MIN)>$|^<KEY_UNKNOWN>$")


def velocity_bin(value: int) -> int:
    """Map MIDI velocity (1..127) into [0..VELOCITY_BINS-1]."""
    bucket = int((float(value) / 127.0) * (VELOCITY_BINS - 1))
    return max(0, min(VELOCITY_BINS - 1, bucket))


def quantize_time_steps(delta_seconds: float) -> int:
    """Round a positive time delta to a positive integer number of steps."""
    steps = int(round(float(delta_seconds) / TIME_SHIFT_RESOLUTION))
    return max(1, steps)


def split_time_shift(delta_seconds: float):
    """Yield TIME_SHIFT_<hex> tokens for a delta, splitting into multiple
    tokens if it exceeds MAX_TIME_STEPS."""
    steps = quantize_time_steps(delta_seconds)
    while steps > 0:
        chunk = min(steps, MAX_TIME_STEPS)
        yield f"TIME_SHIFT_{chunk:#06x}"
        steps -= chunk


def note_on_token(pitch: int) -> str:
    return f"NOTE_ON_{int(pitch) & 0x7f:#04x}"


def note_off_token(pitch: int) -> str:
    return f"NOTE_OFF_{int(pitch) & 0x7f:#04x}"


def velocity_token(value: int) -> str:
    return f"VELOCITY_{velocity_bin(value):#04x}"


def parse_hex_suffix(token: str):
    """Mirror of MidiGenerator::parseHexSuffix in C++.
    Returns int or None."""
    if "_" not in token:
        return None
    suffix = token.rsplit("_", 1)[-1]
    if suffix.lower().startswith("0x"):
        suffix = suffix[2:]
    if not suffix:
        return None
    try:
        return int(suffix, 16)
    except ValueError:
        return None
