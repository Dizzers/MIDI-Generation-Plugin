"""Python sanity-check generator. Mirrors the inference / sampling logic of
DIPLOM SPACE/plugin/juce/Source/ModelInference.cpp so we can A/B compare
against the JUCE plugin output for the same seed and parameters.

It loads the TorchScript module saved by model/export_torchscript.py (or
falls back to the eager checkpoint), runs autoregressive sampling with
temperature / top-k / top-p / repetition-penalty / no-repeat-ngram /
harmony-bias / groove-feel / velocity-feel / max-polyphony, and writes the
result as a .mid file.

Usage:
    python -m model.generate \
        --key C_MAJOR --seed 42 --temperature 0.95 --top_k 12 --top_p 0.9 \
        --target_seconds 4.0 --bpm 120 --out generated/sample.mid

    Strict A/B with the JUCE plugin (disable Python-only note guards):

    python3 -m model.generate --key C_MAJOR --seed 42 --min_note_ons 0 --note_on_logit_boost 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import mido
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.vocab_contract import TIME_SHIFT_RESOLUTION
from model.paths import resolve_checkpoint_dir, resolve_generated_dir
from model.transformer import TransformerLM

_DEFAULT_CKPT_DIR = resolve_checkpoint_dir(PROJECT_ROOT)
DEFAULT_VOCAB = PROJECT_ROOT / "dataset" / "processed" / "vocab.json"
DEFAULT_TS_MODEL = _DEFAULT_CKPT_DIR / "model_best.ts.pt"
DEFAULT_PTH_MODEL = _DEFAULT_CKPT_DIR / "model_best.pth"
_DEFAULT_GEN_DIR = resolve_generated_dir(PROJECT_ROOT)


def map_key_to_token(ui_key: str) -> str:
    """Map UI key like 'C_MAJOR' / 'A_MINOR' to vocab token like '<KEY_C_MAJ>'."""
    upper = ui_key.upper().strip()
    upper = upper.replace("_MAJOR", "_MAJ").replace("_MINOR", "_MIN")
    upper = upper.replace("F_SHARP", "F#").replace("C_SHARP", "C#")
    upper = upper.replace("D_SHARP", "D#").replace("G_SHARP", "G#").replace("A_SHARP", "A#")
    upper = upper.replace("F_FLAT", "E").replace("B_FLAT", "A#").replace("E_FLAT", "D#").replace("A_FLAT", "G#")
    return f"<KEY_{upper}>"


def load_vocab(path: Path):
    with open(path, encoding="utf-8") as handle:
        vocab = json.load(handle)
    token2id = vocab["token2id"]
    id2token = {int(k): v for k, v in vocab["id2token"].items()}
    return token2id, id2token


def build_token_class_indices(token2id: dict[str, int]):
    note_on_ids: list[int] = []
    note_off_ids: list[int] = []
    note_on_pitch_to_id: dict[int, int] = {}
    note_off_pitch_to_id: dict[int, int] = {}
    time_shift_id_steps: list[tuple[int, int]] = []
    velocity_id_bins: list[tuple[int, int]] = []
    banned: list[int] = []

    for tok, idx in token2id.items():
        if tok in {"<PAD>", "<BOS>", "<UNK>"}:
            banned.append(idx)
            continue
        if tok.startswith("<GENRE_") or tok.startswith("<KEY_"):
            banned.append(idx)
            continue
        if tok.startswith("NOTE_ON_"):
            try:
                pitch = int(tok.rsplit("_", 1)[-1], 16)
            except ValueError:
                continue
            note_on_ids.append(idx)
            note_on_pitch_to_id[pitch] = idx
        elif tok.startswith("NOTE_OFF_"):
            try:
                pitch = int(tok.rsplit("_", 1)[-1], 16)
            except ValueError:
                continue
            note_off_ids.append(idx)
            note_off_pitch_to_id[pitch] = idx
        elif tok.startswith("TIME_SHIFT_"):
            try:
                steps = int(tok.rsplit("_", 1)[-1], 16)
            except ValueError:
                continue
            time_shift_id_steps.append((idx, steps))
        elif tok.startswith("VELOCITY_"):
            try:
                value = int(tok.rsplit("_", 1)[-1], 16)
            except ValueError:
                continue
            velocity_id_bins.append((idx, value))

    return {
        "banned": banned,
        "note_on_ids": note_on_ids,
        "note_off_ids": note_off_ids,
        "note_on_pitch_to_id": note_on_pitch_to_id,
        "note_off_pitch_to_id": note_off_pitch_to_id,
        "time_shift_id_steps": time_shift_id_steps,
        "velocity_id_bins": velocity_id_bins,
    }


def in_key_pitch_classes(key_token: str) -> set[int]:
    if key_token == "<KEY_UNKNOWN>":
        return set(range(12))
    inner = key_token[len("<KEY_"):-1]
    if "_MAJ" in inner:
        pc_name = inner.replace("_MAJ", "")
        scale = (0, 2, 4, 5, 7, 9, 11)
    else:
        pc_name = inner.replace("_MIN", "")
        scale = (0, 2, 3, 5, 7, 8, 10)
    pc_map = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
              "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
    root = pc_map.get(pc_name, 0)
    return {(root + s) % 12 for s in scale}


def _body_note_on_count(generated: list[int], id2token: dict[int, str], prefix_len: int) -> int:
    n = 0
    for tid in generated[prefix_len:]:
        tok = id2token.get(int(tid), "")
        if isinstance(tok, str) and tok.startswith("NOTE_ON_"):
            n += 1
    return n


def _summarize_body_tokens(generated: list[int], id2token: dict[int, str], prefix_len: int) -> dict[str, int]:
    counts = {
        "note_on": 0, "note_off": 0, "time_shift": 0, "velocity": 0, "other": 0,
    }
    for tid in generated[prefix_len:]:
        tok = id2token.get(int(tid), "")
        if not isinstance(tok, str):
            counts["other"] += 1
            continue
        if tok.startswith("NOTE_ON_"):
            counts["note_on"] += 1
        elif tok.startswith("NOTE_OFF_"):
            counts["note_off"] += 1
        elif tok.startswith("TIME_SHIFT_"):
            counts["time_shift"] += 1
        elif tok.startswith("VELOCITY_"):
            counts["velocity"] += 1
        else:
            counts["other"] += 1
    return counts


def top_k_top_p_filter(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    out = logits.clone()
    V = out.size(0)
    if top_k > 0:
        k = min(top_k, V)
        threshold = torch.topk(out, k).values[-1]
        out = torch.where(out < threshold, torch.full_like(out, float("-inf")), out)
    if top_p < 1.0:
        sorted_logits, sorted_idx = out.sort(descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        if remove.numel() > 1:
            shifted = remove.clone()
            shifted[1:] = remove[:-1]
            shifted[0] = False
            remove = shifted
        sorted_logits[remove] = float("-inf")
        out = torch.full_like(out, float("-inf"))
        out.scatter_(0, sorted_idx, sorted_logits)
    return out


def load_model(token2id: dict[str, int], device: str):
    """Try TorchScript first; fall back to eager checkpoint."""
    if DEFAULT_TS_MODEL.exists():
        try:
            module = torch.jit.load(str(DEFAULT_TS_MODEL), map_location=device)
            module.eval()
            return module, "torchscript"
        except Exception as exc:
            print(f"  failed to load TorchScript: {exc}; falling back to .pth")

    if not DEFAULT_PTH_MODEL.exists():
        raise FileNotFoundError(
            "No model found. Train first via model/train.py and export via model/export_torchscript.py"
        )
    payload = torch.load(DEFAULT_PTH_MODEL, map_location=device)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    args = payload.get("args", {}) if isinstance(payload, dict) else {}
    pad_id = token2id["<PAD>"]
    num_genres = max(1, sum(1 for t in token2id if t.startswith("<GENRE_")))
    model = TransformerLM(
        vocab_size=len(token2id),
        num_genres=num_genres,
        d_model=args.get("d_model", 512),
        n_heads=args.get("n_heads", 8),
        n_layers=args.get("n_layers", 8),
        d_ff=args.get("d_ff", 2048),
        dropout=args.get("dropout", 0.0),
        max_len=args.get("max_len", 1024),
        pad_id=pad_id,
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, "eager"


def generate(args) -> int:
    if os.environ.get("MIDI_GEN_PRINT_PATH", "").strip():
        print(f"model.generate loaded from: {__file__}", file=sys.stderr)
    token2id, id2token = load_vocab(Path(args.vocab))
    cls = build_token_class_indices(token2id)

    bos_id = token2id["<BOS>"]
    eos_id = token2id["<EOS>"]
    genre_token = "<GENRE_TRAP>"
    if genre_token not in token2id:
        print(f"vocab missing {genre_token}; cannot continue")
        return 1
    genres_sorted = sorted(t for t in token2id if t.startswith("<GENRE_"))
    genre_index = genres_sorted.index(genre_token)

    key_token = map_key_to_token(args.key)
    if key_token not in token2id:
        print(f"warning: {key_token} not in vocab, falling back to <KEY_UNKNOWN>")
        key_token = "<KEY_UNKNOWN>"
    in_key = in_key_pitch_classes(key_token)

    device = args.device
    model, kind = load_model(token2id, device)
    print(f"loaded model ({kind}); seed={args.seed} key={key_token}")

    torch.manual_seed(int(args.seed))

    generated: list[int] = [bos_id, token2id[genre_token], token2id[key_token]]
    prefix_len = len(generated)
    active_pitches: set[int] = set()
    elapsed_seconds = 0.0
    context_max = 1024

    note_on_ids_t = torch.tensor(cls["note_on_ids"], dtype=torch.long, device=device)
    note_off_ids_t = torch.tensor(cls["note_off_ids"], dtype=torch.long, device=device)
    banned_t = torch.tensor(cls["banned"], dtype=torch.long, device=device)
    g_t = torch.tensor([genre_index], dtype=torch.long, device=device)

    for step in range(args.max_len):
        start = max(0, len(generated) - context_max)
        x = torch.tensor([generated[start:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits_full = model(x, g_t)
        logits = logits_full[0, -1].clone()

        temp = max(1e-5, float(args.temperature))
        logits = logits / temp

        if args.repetition_penalty > 1.0:
            window = 128
            recent = set(generated[-window:])
            for tok in recent:
                v = logits[tok]
                logits[tok] = v / args.repetition_penalty if v >= 0 else v * args.repetition_penalty

        logits[banned_t] = float("-inf")

        body_len = len(generated) - prefix_len
        n_note_on_body = _body_note_on_count(generated, id2token, prefix_len)
        if body_len < args.min_body_tokens:
            logits[eos_id] = float("-inf")
        if int(getattr(args, "min_note_ons", 0)) > 0 and n_note_on_body < int(args.min_note_ons):
            logits[eos_id] = float("-inf")

        boost = float(getattr(args, "note_on_logit_boost", 0.0))
        min_n_goal = int(getattr(args, "min_note_ons", 0))
        if boost > 0.0 and min_n_goal > 0 and n_note_on_body < min_n_goal and cls["note_on_ids"]:
            for tid in cls["note_on_ids"]:
                logits[tid] = logits[tid] + boost

        if len(active_pitches) >= args.max_polyphony and note_on_ids_t.numel() > 0:
            logits[note_on_ids_t] = float("-inf")
        for pitch in active_pitches:
            on_id = cls["note_on_pitch_to_id"].get(pitch)
            if on_id is not None:
                logits[on_id] = float("-inf")
        if note_off_ids_t.numel() > 0:
            logits[note_off_ids_t] = float("-inf")
            for pitch in active_pitches:
                off_id = cls["note_off_pitch_to_id"].get(pitch)
                if off_id is not None:
                    logits[off_id] = 0.0

        if abs(args.velocity_feel) > 1e-4 and cls["velocity_id_bins"]:
            amount = max(-1.0, min(1.0, args.velocity_feel))
            for tok_id, bin_v in cls["velocity_id_bins"]:
                t = bin_v / 7.0
                bias = amount * (t - 0.5) * 2.0
                logits[tok_id] = logits[tok_id] + bias

        if abs(args.groove_feel) > 1e-4 and cls["time_shift_id_steps"]:
            amount = max(-1.0, min(1.0, args.groove_feel))
            for tok_id, steps in cls["time_shift_id_steps"]:
                t = max(0.0, min(1.0, steps / 32.0))
                prefer_short = 1.0 - t
                bias = amount * (prefer_short - 0.5) * 1.6
                logits[tok_id] = logits[tok_id] + bias

        if args.harmony_bias > 0 and cls["note_on_ids"]:
            for tok_id, _ in [(tid, None) for tid in cls["note_on_ids"]]:
                pass  # see explicit pass below for clarity
            for pitch, tok_id in cls["note_on_pitch_to_id"].items():
                if pitch % 12 in in_key:
                    logits[tok_id] = logits[tok_id] + args.harmony_bias

        if args.no_repeat_ngram_size > 1 and len(generated) >= args.no_repeat_ngram_size - 1:
            n = args.no_repeat_ngram_size
            prefix = tuple(generated[-(n - 1):])
            for i in range(len(generated) - n + 1):
                if tuple(generated[i:i + n - 1]) == prefix:
                    banned_id = generated[i + n - 1]
                    logits[banned_id] = float("-inf")

        logits = top_k_top_p_filter(logits, args.top_k, args.top_p)
        probs = torch.softmax(logits, dim=-1)
        if torch.isnan(probs).any() or float(probs.sum().item()) <= 0.0:
            next_id = int(logits.argmax().item())
        else:
            next_id = int(torch.multinomial(probs, 1).item())
        generated.append(next_id)

        tok = id2token.get(next_id, "<UNK>")
        if tok == "<EOS>":
            break
        if tok.startswith("NOTE_ON_"):
            try:
                active_pitches.add(int(tok.rsplit("_", 1)[-1], 16))
            except ValueError:
                pass
        elif tok.startswith("NOTE_OFF_"):
            try:
                active_pitches.discard(int(tok.rsplit("_", 1)[-1], 16))
            except ValueError:
                pass
        elif tok.startswith("TIME_SHIFT_"):
            try:
                steps = int(tok.rsplit("_", 1)[-1], 16)
                elapsed_seconds += steps * TIME_SHIFT_RESOLUTION
            except ValueError:
                pass

        min_notes = int(getattr(args, "min_note_ons", 0))
        notes_ok = (min_notes <= 0) or (
            _body_note_on_count(generated, id2token, prefix_len) >= min_notes
        )
        if (
            args.target_seconds > 0
            and elapsed_seconds >= args.target_seconds
            and len(generated) > 64
            and notes_ok
        ):
            generated.append(eos_id)
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_midi(generated, id2token, out_path, args.bpm)
    stats = _summarize_body_tokens(generated, id2token, prefix_len)
    print(
        f"generated {len(generated)} tokens, ~{elapsed_seconds:.2f}s -> {out_path}\n"
        f"  body token mix: {stats} (if note_on is tiny, MIDI will be mostly silence / few notes)"
    )
    return 0


def write_midi(token_ids: list[int], id2token: dict[int, str], out_path: Path, bpm: float) -> None:
    """Convert token ids to a single-track .mid file. Mirrors MidiGenerator.cpp."""
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    ticks_per_beat = mid.ticks_per_beat
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))

    seconds_per_tick = 60.0 / (bpm * ticks_per_beat)
    current_time_seconds = 0.0
    last_event_seconds = 0.0
    current_velocity = 100

    for tid in token_ids:
        tok = id2token.get(int(tid))
        if tok is None or tok.startswith("<"):
            if tok == "<EOS>":
                break
            continue
        if tok.startswith("TIME_SHIFT_"):
            try:
                steps = int(tok.rsplit("_", 1)[-1], 16)
            except ValueError:
                continue
            current_time_seconds += steps * TIME_SHIFT_RESOLUTION
            continue
        if tok.startswith("VELOCITY_"):
            try:
                bin_v = int(tok.rsplit("_", 1)[-1], 16)
            except ValueError:
                continue
            bin_v = max(0, min(7, bin_v))
            current_velocity = max(1, min(127, round(bin_v / 7.0 * 127.0)))
            continue
        if tok.startswith("NOTE_ON_") or tok.startswith("NOTE_OFF_"):
            try:
                pitch = int(tok.rsplit("_", 1)[-1], 16) & 0x7f
            except ValueError:
                continue
            delta_seconds = max(0.0, current_time_seconds - last_event_seconds)
            delta_ticks = int(round(delta_seconds / seconds_per_tick))
            last_event_seconds = current_time_seconds
            if tok.startswith("NOTE_ON_"):
                track.append(mido.Message("note_on", note=pitch, velocity=current_velocity, time=delta_ticks))
            else:
                track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta_ticks))

    mid.save(str(out_path))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate MIDI from trained model. "
            "This build supports --min_note_ons and --note_on_logit_boost; if `python3 -m model.generate -h` "
            "does not list them, you are not running this checkout (wrong cwd, stale file, or shadowed package)."
        ),
        epilog=(
            "Example (match JUCE defaults for note guards): "
            "python3 -m model.generate --key C_MAJOR --seed 42 "
            "--min_note_ons 0 --note_on_logit_boost 0"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--vocab", type=str, default=str(DEFAULT_VOCAB))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--key", type=str, default="A_MINOR")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=25)
    p.add_argument("--top_p", type=float, default=0.93)
    p.add_argument("--repetition_penalty", type=float, default=1.15)
    p.add_argument("--no_repeat_ngram_size", type=int, default=3)
    p.add_argument("--harmony_bias", type=float, default=0.35)
    p.add_argument("--velocity_feel", type=float, default=0.0)
    p.add_argument("--groove_feel", type=float, default=0.3)
    p.add_argument("--max_polyphony", type=int, default=8)
    p.add_argument("--min_body_tokens", type=int, default=48)
    p.add_argument(
        "--min_note_ons",
        type=int,
        default=4,
        help="require at least this many NOTE_ON in the body before EOS or target_seconds "
        "may end generation (0 disables). Default 4 avoids mostly-silence MIDI when the LM "
        "samples long TIME_SHIFT runs.",
    )
    p.add_argument(
        "--note_on_logit_boost",
        type=float,
        default=0.2,
        help="while body has fewer than --min_note_ons NOTE_ON, add this to every NOTE_ON logit "
        "(only if --min_note_ons > 0). Set 0 to disable.",
    )
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--target_seconds", type=float, default=4.0)
    p.add_argument("--bpm", type=float, default=120.0)
    p.add_argument("--out", type=str, default=str(_DEFAULT_GEN_DIR / "sample.mid"))
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(generate(parse_args()))
