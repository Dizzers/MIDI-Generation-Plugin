"""SkyTNT/midi-model adapter (finetune + ONNX export + sampling).

The upstream SkyTNT/midi-model source is vendored under
`skytnt_adapter/midi_model_upstream/`. Scripts here import its modules by
manipulating sys.path so we don't fork the upstream code.
"""
import os
import sys

UPSTREAM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "midi_model_upstream")
if UPSTREAM_DIR not in sys.path:
    sys.path.insert(0, UPSTREAM_DIR)
