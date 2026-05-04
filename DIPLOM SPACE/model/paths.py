"""Writable paths when the repo lives on a read-only mount (e.g. Kaggle
`/kaggle/input/...`). Checkpoints and exports must go under `/kaggle/working`.

Override with env vars if needed:
    MIDI_GEN_CHECKPOINT_DIR   — directory for .pth, history.json, plots/
    MIDI_GEN_PLUGIN_BIN_DIR   — directory for --copy-to-bin (model_best.ts.pt, vocab.json)
"""
from __future__ import annotations

import os
from pathlib import Path


def _project_root_str(project_root: Path) -> str:
    try:
        return str(project_root.resolve())
    except Exception:
        return str(project_root)


def resolve_checkpoint_dir(project_root: Path, cli_override: str | None = None) -> Path:
    if cli_override and str(cli_override).strip():
        return Path(cli_override.strip())
    env = os.environ.get("MIDI_GEN_CHECKPOINT_DIR", "").strip()
    if env:
        return Path(env)
    root = _project_root_str(project_root)
    if root.startswith("/kaggle/input"):
        return Path("/kaggle/working/midi_gen_checkpoints")
    return project_root / "checkpoints"


def resolve_plugin_bin_dir(project_root: Path) -> Path:
    env = os.environ.get("MIDI_GEN_PLUGIN_BIN_DIR", "").strip()
    if env:
        return Path(env)
    root = _project_root_str(project_root)
    if root.startswith("/kaggle/input"):
        return Path("/kaggle/working/midi_gen_plugin_bin")
    return project_root / "plugin" / "juce" / "bin"


def resolve_generated_dir(project_root: Path) -> Path:
    root = _project_root_str(project_root)
    if root.startswith("/kaggle/input"):
        return Path("/kaggle/working/midi_gen_generated")
    return project_root / "generated"
