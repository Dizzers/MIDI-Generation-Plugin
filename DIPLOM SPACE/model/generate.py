import argparse
import json
import random
import sys
from pathlib import Path

import music21 as m21
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_TOKENS_PATH = PROJECT_ROOT / "dataset" / "processed" / "tokens" / "full.npy"
sys.path.insert(0, str(PROJECT_ROOT))

from model.transformer import TransformerLM

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
TIME_SHIFT_RESOLUTION = 0.05


def build_conditioning_maps(token2id):
    genre_tokens = sorted(token for token in token2id if token.startswith("<GENRE_"))
    if not genre_tokens:
        genre_tokens = ["<GENRE_NONE>"]
    genre_to_index = {token[7:-1]: idx for idx, token in enumerate(genre_tokens) if token.startswith("<GENRE_")}
    key_tokens = sorted(token for token in token2id if token.startswith("<KEY_"))
    return genre_to_index, len(genre_tokens), key_tokens


def build_token_groups(token2id):
    groups = {
        "velocity_ids": [],
        "note_on_ids": [],
        "note_off_ids": [],
        "time_shift_ids": [],
        "special_banned_ids": [],
        "note_on_pitch_to_id": {},
        "note_off_pitch_to_id": {},
    }
    for token, token_id in token2id.items():
        if token.startswith("VELOCITY_"):
            groups["velocity_ids"].append(token_id)
        elif token.startswith("NOTE_ON_"):
            groups["note_on_ids"].append(token_id)
            try:
                groups["note_on_pitch_to_id"][int(token.split("_")[2], 16)] = token_id
            except Exception:
                pass
        elif token.startswith("NOTE_OFF_"):
            groups["note_off_ids"].append(token_id)
            try:
                groups["note_off_pitch_to_id"][int(token.split("_")[2], 16)] = token_id
            except Exception:
                pass
        elif token.startswith("TIME_SHIFT_"):
            groups["time_shift_ids"].append(token_id)

        if token in {"<PAD>", "<BOS>", "<UNK>"} or token.startswith("<GENRE_") or token.startswith("<KEY_"):
            groups["special_banned_ids"].append(token_id)

    for key in ["velocity_ids", "note_on_ids", "note_off_ids", "time_shift_ids", "special_banned_ids"]:
        groups[key] = torch.tensor(sorted(groups[key]), dtype=torch.long, device=DEVICE)
    return groups


def top_k_top_p_filter(logits, top_k=0, top_p=1.0):
    filtered = logits.clone()
    vocab_size = filtered.shape[-1]
    if top_k > 0:
        top_k = min(top_k, vocab_size)
        threshold = torch.topk(filtered, top_k).values[..., -1, None]
        filtered[filtered < threshold] = float("-inf")
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cumulative > top_p
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False
        sorted_logits[remove_mask] = float("-inf")
        filtered = torch.full_like(filtered, float("-inf"))
        filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered


def apply_repetition_penalty(logits, generated, penalty=1.0, window=128):
    if penalty <= 1.0 or not generated:
        return logits
    adjusted = logits.clone()
    for token_id in set(generated[-window:]):
        adjusted[token_id] = adjusted[token_id] / penalty if adjusted[token_id] >= 0 else adjusted[token_id] * penalty
    return adjusted


def get_banned_next_tokens(generated, no_repeat_ngram_size):
    n = no_repeat_ngram_size
    if n <= 1 or len(generated) < n - 1:
        return set()
    prefix = tuple(generated[-(n - 1):])
    banned = set()
    for idx in range(len(generated) - n + 1):
        if tuple(generated[idx:idx + n - 1]) == prefix:
            banned.add(generated[idx + n - 1])
    return banned


def token_pitch(token_name):
    if token_name.startswith("NOTE_ON_") or token_name.startswith("NOTE_OFF_"):
        try:
            return int(token_name.split("_")[2], 16)
        except Exception:
            return None
    return None


def collect_active_pitches(generated, id2token):
    active = set()
    for token_id in generated:
        token_name = id2token[token_id]
        pitch = token_pitch(token_name)
        if pitch is None:
            continue
        if token_name.startswith("NOTE_ON_"):
            active.add(pitch)
        elif token_name.startswith("NOTE_OFF_"):
            active.discard(pitch)
    return active


def choose_primer_tokens(genre, key_token, primer_mode, primer_len, full_sequences_cache):
    if primer_mode == "none" or primer_len <= 0 or not DATASET_TOKENS_PATH.exists():
        return []

    if "all" not in full_sequences_cache:
        full_sequences_cache["all"] = np.load(DATASET_TOKENS_PATH, allow_pickle=True).tolist()

    sequences = full_sequences_cache["all"]
    genre_token = f"<GENRE_{genre}>"
    filtered = [seq for seq in sequences if len(seq) > 2 and seq[0] == genre_token and seq[1] == key_token]
    if not filtered:
        filtered = [seq for seq in sequences if len(seq) > 2 and seq[0] == genre_token]
    if not filtered:
        filtered = [seq for seq in sequences if len(seq) > 2]
    if not filtered:
        return []

    seq = random.choice(filtered)
    return list(seq[2:2 + primer_len])


def apply_generation_constraints(logits, generated, id2token, token_groups, eos_id=None, min_body_tokens=48, max_polyphony=8):
    constrained = logits.clone()

    if token_groups["special_banned_ids"].numel() > 0:
        constrained.index_fill_(0, token_groups["special_banned_ids"], float("-inf"))

    prefix_len = 3
    body_len = max(0, len(generated) - prefix_len)
    if eos_id is not None and body_len < min_body_tokens:
        constrained[eos_id] = float("-inf")

    active_pitches = collect_active_pitches(generated[prefix_len:], id2token)
    for pitch in active_pitches:
        note_on_id = token_groups["note_on_pitch_to_id"].get(pitch)
        if note_on_id is not None:
            constrained[note_on_id] = float("-inf")

    if len(active_pitches) >= max_polyphony:
        if token_groups["note_on_ids"].numel() > 0:
            constrained.index_fill_(0, token_groups["note_on_ids"], float("-inf"))

    inactive_pitches = set(token_groups["note_off_pitch_to_id"].keys()) - active_pitches
    for pitch in inactive_pitches:
        note_off_id = token_groups["note_off_pitch_to_id"].get(pitch)
        if note_off_id is not None:
            constrained[note_off_id] = float("-inf")

    if torch.isinf(constrained).all():
        return logits
    return constrained


def sample_next_token(
    logits,
    generated,
    id2token,
    token_groups,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    repetition_penalty=1.0,
    no_repeat_ngram_size=0,
    eos_id=None,
    min_body_tokens=48,
):
    adjusted = logits / max(1e-5, float(temperature))
    adjusted = apply_repetition_penalty(adjusted, generated, repetition_penalty)
    adjusted = apply_generation_constraints(adjusted, generated, id2token, token_groups, eos_id=eos_id, min_body_tokens=min_body_tokens)

    banned = get_banned_next_tokens(generated, no_repeat_ngram_size)
    if banned:
        adjusted.index_fill_(0, torch.tensor(sorted(banned), device=DEVICE, dtype=torch.long), float("-inf"))

    filtered = top_k_top_p_filter(adjusted, top_k=top_k, top_p=top_p)
    probs = torch.softmax(filtered, dim=-1)
    if torch.isnan(probs).any() or probs.sum() <= 0:
        return torch.argmax(adjusted).item()
    return torch.multinomial(probs, 1).item()


def _body_tokens(tokens):
    return [token for token in tokens if not (token in {"<BOS>", "<EOS>"} or token.startswith("<GENRE_") or token.startswith("<KEY_"))]


def _ngram_repeat_rate(tokens, n=4):
    body = _body_tokens(tokens)
    if len(body) < n:
        return 0.0
    grams = [tuple(body[idx:idx + n]) for idx in range(len(body) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def _note_on_off_balance(tokens):
    body = _body_tokens(tokens)
    note_on = sum(1 for token in body if token.startswith("NOTE_ON_"))
    note_off = sum(1 for token in body if token.startswith("NOTE_OFF_"))
    return 1.0 - abs(note_on - note_off) / max(1, note_on + note_off)


def _polyphony_stats(tokens):
    onset_counts = {}
    current_step = 0
    for token in _body_tokens(tokens):
        if token.startswith("TIME_SHIFT_"):
            try:
                current_step += int(token.split("_")[2], 16)
            except Exception:
                pass
        elif token.startswith("NOTE_ON_"):
            onset_counts[current_step] = onset_counts.get(current_step, 0) + 1
    if not onset_counts:
        return {"chord_ratio": 0.0, "max_simul": 0.0}
    counts = list(onset_counts.values())
    return {
        "chord_ratio": sum(1 for count in counts if count >= 2) / len(counts),
        "max_simul": max(counts),
    }


def _quick_quality_score(tokens):
    body = _body_tokens(tokens)
    if not body:
        return -1.0

    pitches = []
    time_shifts = []
    for token in body:
        if token.startswith("NOTE_ON_"):
            try:
                pitches.append(int(token.split("_")[2], 16))
            except Exception:
                pass
        elif token.startswith("TIME_SHIFT_"):
            try:
                time_shifts.append(int(token.split("_")[2], 16))
            except Exception:
                pass

    if not pitches:
        return -1.0

    repeat = _ngram_repeat_rate(tokens)
    balance = _note_on_off_balance(tokens)
    poly = _polyphony_stats(tokens)
    pitch_range = min(1.0, (max(pitches) - min(pitches)) / 36.0)
    rhythm_diversity = len(set(time_shifts)) / max(1, len(time_shifts))

    return (
        0.24 * balance
        + 0.20 * (1.0 - repeat)
        + 0.18 * pitch_range
        + 0.18 * rhythm_diversity
        + 0.12 * min(1.0, poly["chord_ratio"] / 0.20)
        + 0.08 * min(1.0, poly["max_simul"] / 4.0)
    )


def generate_tokens(
    model,
    token2id,
    id2token,
    genre="TRAP",
    max_len=512,
    temperature=0.92,
    top_k=40,
    top_p=0.92,
    repetition_penalty=1.15,
    no_repeat_ngram_size=4,
    genre_to_index=None,
    primer_mode="dataset",
    primer_len=64,
    full_sequences_cache=None,
    min_body_tokens=64,
    target_seconds=16.0,
    key_name="AUTO",
):
    if genre_to_index is None:
        genre_to_index, _, key_tokens = build_conditioning_maps(token2id)
    else:
        _, _, key_tokens = build_conditioning_maps(token2id)

    key_token = None
    if key_name is None or str(key_name).upper() == "AUTO":
        key_token = key_tokens[0] if key_tokens else "<KEY_UNKNOWN>"
    else:
        candidate = f"<KEY_{str(key_name).upper()}>"
        key_token = candidate if candidate in token2id else (key_tokens[0] if key_tokens else "<KEY_UNKNOWN>")

    genre_token = f"<GENRE_{genre}>"
    for token in ["<BOS>", genre_token, key_token]:
        if token not in token2id:
            raise ValueError(f"Token {token} not found in vocab")

    if full_sequences_cache is None:
        full_sequences_cache = {}

    genre_idx = genre_to_index.get(genre, 0)
    genre_id = torch.tensor([genre_idx], dtype=torch.long, device=DEVICE)
    token_groups = build_token_groups(token2id)
    generated = [token2id["<BOS>"], token2id[genre_token], token2id[key_token]]
    generated.extend(
        token2id.get(token, token2id["<UNK>"])
        for token in choose_primer_tokens(genre, key_token, primer_mode, primer_len, full_sequences_cache)
        if token in token2id
    )

    eos_id = token2id.get("<EOS>")
    elapsed = 0.0
    model.eval()
    with torch.no_grad():
        for _ in range(max_len):
            context = generated[-model.max_len:]
            x = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(DEVICE)
            logits = model(x, genre_id=genre_id)[0, -1]
            next_token_id = sample_next_token(
                logits,
                context,
                id2token,
                token_groups,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                eos_id=eos_id,
                min_body_tokens=min_body_tokens,
            )
            generated.append(next_token_id)

            token_name = id2token[next_token_id]
            if token_name.startswith("TIME_SHIFT_"):
                try:
                    elapsed += int(token_name.split("_")[2], 16) * TIME_SHIFT_RESOLUTION
                except Exception:
                    pass

            if eos_id is not None and next_token_id == eos_id:
                break
            if target_seconds is not None and elapsed >= float(target_seconds) and len(generated) >= (3 + min_body_tokens):
                if eos_id is not None:
                    generated.append(eos_id)
                break

    ended_by_eos = bool(eos_id is not None and generated[-1] == eos_id)
    return [id2token[idx] for idx in generated], ended_by_eos


def generate_best_candidate(
    model,
    token2id,
    id2token,
    genre,
    max_len,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    no_repeat_ngram_size,
    genre_to_index,
    primer_mode,
    primer_len,
    full_sequences_cache,
    min_body_tokens,
    target_seconds,
    key_name,
    candidates_per_sample=8,
    diversity_jitter=0.05,
):
    best = None
    best_score = None
    seen = set()
    for _ in range(max(1, candidates_per_sample)):
        temp = max(0.82, float(temperature) + random.uniform(-diversity_jitter, diversity_jitter))
        topp = min(0.98, max(0.86, float(top_p) + random.uniform(-0.03, 0.03)))
        topk = max(10, int(top_k + random.randint(-4, 4)))
        tokens, ended = generate_tokens(
            model=model,
            token2id=token2id,
            id2token=id2token,
            genre=genre,
            max_len=max_len,
            temperature=temp,
            top_k=topk,
            top_p=topp,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            genre_to_index=genre_to_index,
            primer_mode=primer_mode,
            primer_len=primer_len,
            full_sequences_cache=full_sequences_cache,
            min_body_tokens=min_body_tokens,
            target_seconds=target_seconds,
            key_name=key_name,
        )
        signature = tuple(_body_tokens(tokens))
        if signature in seen:
            continue
        seen.add(signature)
        score = _quick_quality_score(tokens)
        if best is None or score > best_score:
            best = (tokens, ended)
            best_score = score

    if best is None:
        return generate_tokens(
            model=model,
            token2id=token2id,
            id2token=id2token,
            genre=genre,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            genre_to_index=genre_to_index,
            primer_mode=primer_mode,
            primer_len=primer_len,
            full_sequences_cache=full_sequences_cache,
            min_body_tokens=min_body_tokens,
            target_seconds=target_seconds,
            key_name=key_name,
        )
    return best


def tokens_to_stream(tokens):
    stream = m21.stream.Stream()
    current_offset = 0.0
    active_notes = {}
    last_velocity = 80

    for token in tokens:
        if token.startswith("TIME_SHIFT_"):
            current_offset += int(token.split("_")[2], 16) * TIME_SHIFT_RESOLUTION
        elif token.startswith("VELOCITY_"):
            bucket = int(token.split("_")[1], 16)
            last_velocity = max(1, min(127, int(round((bucket / 7) * 127))))
        elif token.startswith("NOTE_ON_"):
            pitch = int(token.split("_")[2], 16)
            active_notes[pitch] = (current_offset, last_velocity)
        elif token.startswith("NOTE_OFF_"):
            pitch = int(token.split("_")[2], 16)
            state = active_notes.pop(pitch, None)
            if state is None:
                continue
            start, velocity = state
            note = m21.note.Note(pitch)
            note.offset = start
            note.quarterLength = max(TIME_SHIFT_RESOLUTION, current_offset - start)
            note.volume.velocity = velocity
            stream.append(note)

    final_offset = current_offset + (2 * TIME_SHIFT_RESOLUTION)
    for pitch, (start, velocity) in active_notes.items():
        note = m21.note.Note(pitch)
        note.offset = start
        note.quarterLength = max(TIME_SHIFT_RESOLUTION, final_offset - start)
        note.volume.velocity = velocity
        stream.append(note)

    return stream


def clean_state_dict(state_dict):
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict


def load_model_and_vocab(checkpoint_path, vocab_path):
    with open(vocab_path) as handle:
        vocab = json.load(handle)
    token2id = vocab["token2id"]
    id2token = {int(idx): token for idx, token in vocab["id2token"].items()}
    genre_to_index, num_genres, _ = build_conditioning_maps(token2id)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model_config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
    model_config = dict(model_config)
    model_config.pop("vocab_size", None)
    model_config.setdefault("pad_id", token2id.get("<PAD>"))
    model_config.setdefault("num_roles", 0)
    model_config.setdefault("num_genres", num_genres)

    model = TransformerLM(len(token2id), **model_config).to(DEVICE)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(clean_state_dict(state_dict))
    return model, token2id, id2token, genre_to_index


def parse_args():
    parser = argparse.ArgumentParser(description="Generate full MIDI samples")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_best.pth")
    parser.add_argument("--vocab", type=str, default="dataset/processed/vocab.json")
    parser.add_argument("--genre", type=str, default="TRAP")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.92)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--primer-mode", type=str, default="dataset", choices=["dataset", "none"])
    parser.add_argument("--primer-len", type=int, default=64)
    parser.add_argument("--min-body-tokens", type=int, default=64)
    parser.add_argument("--target-seconds", type=float, default=16.0)
    parser.add_argument("--key", type=str, default="AUTO")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--candidates-per-sample", type=int, default=8)
    parser.add_argument("--diversity-jitter", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="generated_full")
    return parser.parse_args()


def save_metadata(out_dir, sample_name, args, tokens):
    body = _body_tokens(tokens)
    payload = {
        "config": vars(args),
        "body_length": len(body),
        "ended_by_eos": bool(tokens and tokens[-1] == "<EOS>"),
        "quality_score": _quick_quality_score(tokens),
        "polyphony": _polyphony_stats(tokens),
    }
    with open(out_dir / f"{sample_name}.json", "w") as handle:
        json.dump(payload, handle, indent=2)


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if DEVICE == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    checkpoint_path = PROJECT_ROOT / args.checkpoint
    vocab_path = PROJECT_ROOT / args.vocab
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model, token2id, id2token, genre_to_index = load_model_and_vocab(checkpoint_path, vocab_path)
    cache = {}
    print(f"Device: {DEVICE}")
    print(f"Full-MIDI generation: genre={args.genre.upper()}, temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")

    for idx in range(1, args.samples + 1):
        tokens, _ = generate_best_candidate(
            model=model,
            token2id=token2id,
            id2token=id2token,
            genre=args.genre.upper(),
            max_len=args.max_len,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            genre_to_index=genre_to_index,
            primer_mode=args.primer_mode,
            primer_len=args.primer_len,
            full_sequences_cache=cache,
            min_body_tokens=args.min_body_tokens,
            target_seconds=args.target_seconds,
            key_name=args.key,
            candidates_per_sample=args.candidates_per_sample,
            diversity_jitter=args.diversity_jitter,
        )

        stream = tokens_to_stream(tokens)
        sample_name = f"sample_{idx:02d}_{args.genre.lower()}_full"
        midi_path = out_dir / f"{sample_name}.mid"
        stream.write("midi", fp=str(midi_path))
        save_metadata(out_dir, sample_name, args, tokens)
        print(f"Saved: {midi_path}")


if __name__ == "__main__":
    main()
