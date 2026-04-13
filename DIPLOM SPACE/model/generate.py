import argparse
import json
import random
import sys
from pathlib import Path

import music21 as m21
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_TOKENS_DIR = PROJECT_ROOT / "dataset" / "processed" / "tokens"
sys.path.insert(0, str(PROJECT_ROOT))

from model.transformer import TransformerLM

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

TIME_SHIFT_RESOLUTION = 0.05
ROLES = ["MELODY", "BASS", "CHORDS"]
ROLE_PITCH_RANGE = {
    "MELODY": (50, 96),
    "BASS": (28, 60),
    "CHORDS": (40, 84),
}


def build_conditioning_maps(token2id):
    role_to_index = {"MELODY": 0, "BASS": 1, "CHORDS": 2}
    genre_tokens = sorted([t for t in token2id.keys() if t.startswith("<GENRE_")])
    if not genre_tokens:
        genre_tokens = ["<GENRE_NONE>"]
    genre_to_index = {tok[7:-1]: i for i, tok in enumerate(genre_tokens) if tok.startswith("<GENRE_")}
    return role_to_index, genre_to_index, len(genre_tokens)


def build_token_groups(token2id):
    groups = {
        "velocity_ids": [],
        "note_on_ids": [],
        "note_off_ids": [],
        "time_shift_ids": [],
        "special_body_banned_ids": [],
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

        if (
            token in {"<PAD>", "<BOS>", "<UNK>"}
            or token.startswith("<ROLE_")
            or token.startswith("<GENRE_")
        ):
            groups["special_body_banned_ids"].append(token_id)

    for key in ["velocity_ids", "note_on_ids", "note_off_ids", "time_shift_ids", "special_body_banned_ids"]:
        groups[key] = torch.tensor(sorted(groups[key]), dtype=torch.long, device=DEVICE)
    return groups


def top_k_top_p_filter(logits, top_k=0, top_p=1.0):
    filtered = logits.clone()
    vocab_size = filtered.shape[-1]
    if top_k > 0:
        k = min(top_k, vocab_size)
        threshold = torch.topk(filtered, k).values[..., -1, None]
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
    out = logits.clone()
    for token_id in set(generated[-window:]):
        if out[token_id] < 0:
            out[token_id] *= penalty
        else:
            out[token_id] /= penalty
    return out


def get_banned_next_tokens(generated, no_repeat_ngram_size):
    n = no_repeat_ngram_size
    if n <= 1 or len(generated) < n - 1:
        return set()

    prefix = tuple(generated[-(n - 1):])
    banned = set()
    for i in range(len(generated) - n + 1):
        if tuple(generated[i:i + n - 1]) == prefix:
            banned.add(generated[i + n - 1])
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


def body_elapsed_seconds(generated, id2token, prefix_len=3):
    elapsed = 0.0
    for token_id in generated[prefix_len:]:
        token_name = id2token[token_id]
        if token_name.startswith("TIME_SHIFT_"):
            try:
                elapsed += int(token_name.split("_")[2], 16) * TIME_SHIFT_RESOLUTION
            except Exception:
                pass
    return elapsed


def last_note_on_pitch(generated, id2token, prefix_len=3):
    for token_id in reversed(generated[prefix_len:]):
        token_name = id2token[token_id]
        if token_name.startswith("NOTE_ON_"):
            try:
                return int(token_name.split("_")[2], 16)
            except Exception:
                return None
    return None


def apply_generation_constraints(logits, generated, id2token, token_groups, role=None, eos_id=None, min_body_tokens=24, max_melody_leap=12):
    constrained = logits.clone()

    if token_groups["special_body_banned_ids"].numel() > 0:
        constrained.index_fill_(0, token_groups["special_body_banned_ids"], float("-inf"))

    prefix_len = 3
    body_len = max(0, len(generated) - prefix_len)
    if eos_id is not None and body_len < min_body_tokens:
        constrained[eos_id] = float("-inf")

    active_pitches = collect_active_pitches(generated[prefix_len:], id2token)
    for pitch in active_pitches:
        note_on_id = token_groups["note_on_pitch_to_id"].get(pitch)
        if note_on_id is not None:
            constrained[note_on_id] = float("-inf")
    inactive_pitches = set(token_groups["note_off_pitch_to_id"].keys()) - active_pitches
    for pitch in inactive_pitches:
        note_off_id = token_groups["note_off_pitch_to_id"].get(pitch)
        if note_off_id is not None:
            constrained[note_off_id] = float("-inf")

    pitch_lo, pitch_hi = ROLE_PITCH_RANGE.get(role or "", (0, 127))
    for pitch, note_on_id in token_groups["note_on_pitch_to_id"].items():
        if pitch < pitch_lo or pitch > pitch_hi:
            constrained[note_on_id] = float("-inf")

    last_token = id2token[generated[-1]]
    if last_token.startswith("VELOCITY_"):
        candidate_pitches = set(token_groups["note_on_pitch_to_id"].keys()) - active_pitches

        if role == "MELODY":
            lp = last_note_on_pitch(generated, id2token, prefix_len=prefix_len)
            if lp is not None:
                candidate_pitches = {
                    p for p in candidate_pitches
                    if abs(p - lp) <= max_melody_leap
                }

        allowed = torch.tensor(
            sorted(token_groups["note_on_pitch_to_id"][pitch] for pitch in candidate_pitches),
            dtype=torch.long,
            device=DEVICE,
        )
        constrained.fill_(float("-inf"))
        if allowed.numel() > 0:
            constrained.index_copy_(0, allowed, logits.index_select(0, allowed))
    else:
        if token_groups["note_on_ids"].numel() > 0:
            constrained.index_fill_(0, token_groups["note_on_ids"], float("-inf"))

    if torch.isinf(constrained).all():
        return logits
    return constrained


def sample_next_token(
    logits,
    generated,
    id2token,
    token_groups,
    role=None,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    repetition_penalty=1.0,
    no_repeat_ngram_size=0,
    eos_id=None,
    min_body_tokens=24,
    max_melody_leap=12,
):
    temperature = max(1e-5, float(temperature))
    adjusted = logits / temperature
    adjusted = apply_repetition_penalty(adjusted, generated, repetition_penalty)
    adjusted = apply_generation_constraints(
        adjusted,
        generated,
        id2token=id2token,
        token_groups=token_groups,
        role=role,
        eos_id=eos_id,
        min_body_tokens=min_body_tokens,
        max_melody_leap=max_melody_leap,
    )

    banned = get_banned_next_tokens(generated, no_repeat_ngram_size)
    if banned:
        ban_tensor = torch.tensor(list(banned), device=adjusted.device, dtype=torch.long)
        adjusted.index_fill_(0, ban_tensor, float("-inf"))

    filtered = top_k_top_p_filter(adjusted, top_k=top_k, top_p=top_p)
    probs = torch.softmax(filtered, dim=-1)
    if torch.isnan(probs).any() or probs.sum() <= 0:
        return torch.argmax(adjusted).item()
    return torch.multinomial(probs, 1).item()


def load_role_token_sequences(role):
    role_path = DATASET_TOKENS_DIR / f"{role.lower()}.npy"
    if not role_path.exists():
        return []
    return np.load(role_path, allow_pickle=True).tolist()


def choose_primer_tokens(role, genre, primer_mode, primer_len, role_sequences_cache):
    if primer_mode == "none" or primer_len <= 0:
        return []

    if role not in role_sequences_cache:
        role_sequences_cache[role] = load_role_token_sequences(role)

    role_sequences = role_sequences_cache[role]
    genre_token = f"<GENRE_{genre}>"
    filtered = [seq for seq in role_sequences if len(seq) > 2 and seq[0] == f"<ROLE_{role}>" and seq[1] == genre_token]
    if not filtered:
        filtered = [seq for seq in role_sequences if len(seq) > 2]
    if not filtered:
        return []

    sequence = random.choice(filtered)
    return list(sequence[2:2 + primer_len])


def generate_tokens(
    model,
    token2id,
    id2token,
    role="MELODY",
    genre="TRAP",
    max_len=256,
    temperature=0.95,
    top_k=12,
    top_p=0.9,
    repetition_penalty=1.15,
    no_repeat_ngram_size=4,
    role_to_index=None,
    genre_to_index=None,
    primer_mode="dataset",
    primer_len=24,
    role_sequences_cache=None,
    min_body_tokens=24,
    target_seconds=2.5,
    max_melody_leap=12,
):
    role_token = f"<ROLE_{role}>"
    genre_token = f"<GENRE_{genre}>"
    for token in ["<BOS>", role_token, genre_token]:
        if token not in token2id:
            raise ValueError(f"Token {token} not found in vocab")

    if role_to_index is None or genre_to_index is None:
        role_to_index, genre_to_index, _ = build_conditioning_maps(token2id)
    if role_sequences_cache is None:
        role_sequences_cache = {}

    role_idx = role_to_index.get(role, 0)
    genre_idx = genre_to_index.get(genre, 0)
    role_id = torch.tensor([role_idx], dtype=torch.long, device=DEVICE)
    genre_id = torch.tensor([genre_idx], dtype=torch.long, device=DEVICE)
    token_groups = build_token_groups(token2id)

    generated = [token2id["<BOS>"], token2id[role_token], token2id[genre_token]]
    generated.extend(
        token2id.get(tok, token2id.get("<UNK>"))
        for tok in choose_primer_tokens(role, genre, primer_mode, primer_len, role_sequences_cache)
        if tok in token2id
    )
    eos_id = token2id.get("<EOS>")
    elapsed = body_elapsed_seconds(generated, id2token, prefix_len=3)

    model.eval()
    with torch.no_grad():
        for _ in range(max_len):
            context = generated[-model.max_len:]
            x = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(DEVICE)
            logits = model(x, role_id=role_id, genre_id=genre_id)[0, -1]
            next_token_id = sample_next_token(
                logits=logits,
                generated=context,
                id2token=id2token,
                token_groups=token_groups,
                role=role,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                eos_id=eos_id,
                min_body_tokens=min_body_tokens,
                max_melody_leap=max_melody_leap,
            )
            generated.append(next_token_id)

            next_tok = id2token[next_token_id]
            if next_tok.startswith("TIME_SHIFT_"):
                try:
                    elapsed += int(next_tok.split("_")[2], 16) * TIME_SHIFT_RESOLUTION
                except Exception:
                    pass

            if eos_id is not None and next_token_id == eos_id:
                break

            if target_seconds is not None and elapsed >= float(target_seconds) and len(generated) >= (3 + min_body_tokens):
                if eos_id is not None:
                    generated.append(eos_id)
                break

    ended_by_eos = bool(eos_id is not None and generated[-1] == eos_id)
    return [id2token[i] for i in generated], ended_by_eos


def _body_tokens(tokens):
    return [
        t for t in tokens
        if not (t in {"<BOS>", "<EOS>"} or t.startswith("<ROLE_") or t.startswith("<GENRE_"))
    ]


def _ngram_repeat_rate(tokens, n=4):
    body = _body_tokens(tokens)
    if len(body) < n:
        return 0.0
    grams = [tuple(body[i:i+n]) for i in range(len(body) - n + 1)]
    return 1.0 - (len(set(grams)) / max(1, len(grams)))


def _note_on_off_balance(tokens):
    body = _body_tokens(tokens)
    on = sum(1 for t in body if t.startswith("NOTE_ON_"))
    off = sum(1 for t in body if t.startswith("NOTE_OFF_"))
    return 1.0 - abs(on - off) / max(1, on + off)


def _quick_quality_score(tokens, role):
    body = _body_tokens(tokens)
    if not body:
        return -1.0

    repeat = _ngram_repeat_rate(tokens, n=4)
    balance = _note_on_off_balance(tokens)
    note_on = [t for t in body if t.startswith("NOTE_ON_")]
    pitches = []
    for t in note_on:
        try:
            pitches.append(int(t.split("_")[2], 16))
        except Exception:
            pass

    if not pitches:
        return -1.0

    pitch_range = max(pitches) - min(pitches)

    # role-sensitive soft priors
    if role == "BASS":
        role_score = sum(1 for p in pitches if p <= 55) / len(pitches)
    elif role == "MELODY":
        role_score = sum(1 for p in pitches if p >= 52) / len(pitches)
    else:
        role_score = min(1.0, pitch_range / 18.0)

    # higher is better
    return (0.45 * balance) + (0.30 * role_score) + (0.25 * (1.0 - repeat))


def generate_best_candidate(
    model,
    token2id,
    id2token,
    role,
    genre,
    max_len,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    no_repeat_ngram_size,
    role_to_index,
    genre_to_index,
    primer_mode,
    primer_len,
    role_sequences_cache,
    min_body_tokens,
    target_seconds,
    max_melody_leap,
    candidates_per_sample=4,
    diversity_jitter=0.08,
):
    best = None
    best_score = None
    seen = set()

    attempts = max(1, int(candidates_per_sample))
    for _ in range(attempts):
        t = max(0.75, float(temperature) + random.uniform(-diversity_jitter, diversity_jitter))
        p = min(0.98, max(0.84, float(top_p) + random.uniform(-0.03, 0.03)))
        k = max(8, int(top_k + random.randint(-3, 3)))

        tokens, ended = generate_tokens(
            model=model,
            token2id=token2id,
            id2token=id2token,
            role=role,
            genre=genre,
            max_len=max_len,
            temperature=t,
            top_k=k,
            top_p=p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            role_to_index=role_to_index,
            genre_to_index=genre_to_index,
            primer_mode=primer_mode,
            primer_len=primer_len,
            role_sequences_cache=role_sequences_cache,
            min_body_tokens=min_body_tokens,
            target_seconds=target_seconds,
            max_melody_leap=max_melody_leap,
        )

        sig = tuple(_body_tokens(tokens))
        if sig in seen:
            continue
        seen.add(sig)

        score = _quick_quality_score(tokens, role)
        if best is None or score > best_score:
            best = (tokens, ended)
            best_score = score

    if best is None:
        return generate_tokens(
            model=model,
            token2id=token2id,
            id2token=id2token,
            role=role,
            genre=genre,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            role_to_index=role_to_index,
            genre_to_index=genre_to_index,
            primer_mode=primer_mode,
            primer_len=primer_len,
            role_sequences_cache=role_sequences_cache,
            min_body_tokens=min_body_tokens,
            target_seconds=target_seconds,
            max_melody_leap=max_melody_leap,
        )

    return best


def tokens_to_part(tokens, part_name):
    part = m21.stream.Part()
    part.partName = part_name
    current_offset = 0.0
    active_notes = {}
    last_velocity = 80

    for tok in tokens:
        if tok.startswith("TIME_SHIFT_"):
            step = int(tok.split("_")[2], 16)
            current_offset += step * TIME_SHIFT_RESOLUTION
        elif tok.startswith("VELOCITY_"):
            bin_idx = int(tok.split("_")[1], 16)
            last_velocity = max(1, min(127, int(round((bin_idx / 7) * 127))))
        elif tok.startswith("NOTE_ON_"):
            pitch = int(tok.split("_")[2], 16)
            active_notes[pitch] = (current_offset, last_velocity)
        elif tok.startswith("NOTE_OFF_"):
            pitch = int(tok.split("_")[2], 16)
            state = active_notes.pop(pitch, None)
            if state is None:
                continue
            start, velocity = state
            dur = max(TIME_SHIFT_RESOLUTION, current_offset - start)
            note = m21.note.Note(pitch)
            note.offset = start
            note.quarterLength = dur
            note.volume.velocity = velocity
            part.append(note)

    if active_notes:
        final_offset = current_offset + (2 * TIME_SHIFT_RESOLUTION)
        for pitch, (start, velocity) in active_notes.items():
            note = m21.note.Note(pitch)
            note.offset = start
            note.quarterLength = max(TIME_SHIFT_RESOLUTION, final_offset - start)
            note.volume.velocity = velocity
            part.append(note)

    return part


def save_single(tokens, out_path, role):
    stream = m21.stream.Stream()
    stream.append(tokens_to_part(tokens, role))
    stream.write("midi", fp=str(out_path))


def save_arrangement(tokens_by_role, out_path):
    score = m21.stream.Score()
    for role in ROLES:
        score.append(tokens_to_part(tokens_by_role[role], role))
    score.write("midi", fp=str(out_path))


def clean_state_dict(state_dict):
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_model_and_vocab(checkpoint_path, vocab_path):
    with open(vocab_path) as f:
        vocab = json.load(f)
    token2id = vocab["token2id"]
    id2token = {int(k): v for k, v in vocab["id2token"].items()}

    role_to_index, genre_to_index, num_genres = build_conditioning_maps(token2id)
    num_roles = 3

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model_config = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
    
    if model_config is None:
        model = TransformerLM(
            len(token2id),
            pad_id=token2id.get("<PAD>"),
            num_roles=num_roles,
            num_genres=num_genres,
        ).to(DEVICE)
    else:
        # Создаём копию конфигурации
        model_config = dict(model_config)
        
        # Удаляем vocab_size из model_config, если он там есть
        if 'vocab_size' in model_config:
            del model_config['vocab_size']
        
        # Устанавливаем недостающие параметры
        if 'pad_id' not in model_config:
            model_config['pad_id'] = token2id.get("<PAD>")
        if 'num_roles' not in model_config:
            model_config['num_roles'] = num_roles
        if 'num_genres' not in model_config:
            model_config['num_genres'] = num_genres
        
        # Создаём модель, передавая vocab_size как первый аргумент
        model = TransformerLM(len(token2id), **model_config).to(DEVICE)

    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(clean_state_dict(state_dict))
    return model, token2id, id2token, role_to_index, genre_to_index


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MIDI samples")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_best.pth")
    parser.add_argument("--vocab", type=str, default="dataset/processed/vocab.json")
    parser.add_argument("--role", type=str, default="ALL", choices=ROLES + ["ALL"])
    parser.add_argument("--genre", type=str, default="TRAP")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.00)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--top-p", type=float, default=0.93)
    parser.add_argument("--repetition-penalty", type=float, default=1.20)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--primer-mode", type=str, default="dataset", choices=["dataset", "none"])
    parser.add_argument("--primer-len", type=int, default=24)
    parser.add_argument("--min-body-tokens", type=int, default=24)
    parser.add_argument("--target-seconds", type=float, default=2.5)
    parser.add_argument("--max-melody-leap", type=int, default=12)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--candidates-per-sample", type=int, default=4)
    parser.add_argument("--diversity-jitter", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="generated")
    return parser.parse_args()


def save_metadata(out_dir, sample_name, args, tokens_by_role):
    body_lengths = {}
    eos_flags = {}
    for role, tokens in tokens_by_role.items():
        body = [t for t in tokens if not (t in {"<BOS>", "<EOS>"} or t.startswith("<ROLE_") or t.startswith("<GENRE_"))]
        body_lengths[role] = len(body)
        eos_flags[role] = bool(tokens and tokens[-1] == "<EOS>")

    payload = {
        "config": vars(args),
        "body_lengths": body_lengths,
        "ended_by_eos": eos_flags,
    }
    with open(out_dir / f"{sample_name}.json", "w") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if DEVICE == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        if DEVICE == "mps":
            torch.mps.manual_seed(args.seed)

    checkpoint_path = PROJECT_ROOT / args.checkpoint
    vocab_path = PROJECT_ROOT / args.vocab
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model, token2id, id2token, role_to_index, genre_to_index = load_model_and_vocab(checkpoint_path, vocab_path)
    role_sequences_cache = {}
    print(f"Device: {DEVICE}")
    print(
        f"Generation config: role={args.role}, genre={args.genre.upper()}, temp={args.temperature}, "
        f"top_k={args.top_k}, top_p={args.top_p}, primer={args.primer_mode}:{args.primer_len}, "
        f"target_seconds={args.target_seconds}, max_melody_leap={args.max_melody_leap}, "
        f"candidates={args.candidates_per_sample}, jitter={args.diversity_jitter}"
    )

    eos_count = 0
    for i in range(1, args.samples + 1):
        if args.role == "ALL":
            tokens_by_role = {}
            all_eos = True
            for role in ROLES:
                tokens, ended_by_eos = generate_best_candidate(
                    model=model,
                    token2id=token2id,
                    id2token=id2token,
                    role=role,
                    genre=args.genre.upper(),
                    max_len=args.max_len,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    role_to_index=role_to_index,
                    genre_to_index=genre_to_index,
                    primer_mode=args.primer_mode,
                    primer_len=args.primer_len,
                    role_sequences_cache=role_sequences_cache,
                    min_body_tokens=args.min_body_tokens,
                    target_seconds=args.target_seconds,
                    max_melody_leap=args.max_melody_leap,
                    candidates_per_sample=args.candidates_per_sample,
                    diversity_jitter=args.diversity_jitter,
                )
                tokens_by_role[role] = tokens
                all_eos = all_eos and ended_by_eos
            sample_name = f"sample_{i:02d}_arrangement_{args.genre.lower()}"
            out_path = out_dir / f"{sample_name}.mid"
            save_arrangement(tokens_by_role, out_path)
            save_metadata(out_dir, sample_name, args, tokens_by_role)
            if all_eos:
                eos_count += 1
        else:
            tokens, ended_by_eos = generate_best_candidate(
                model=model,
                token2id=token2id,
                id2token=id2token,
                role=args.role.upper(),
                genre=args.genre.upper(),
                max_len=args.max_len,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                role_to_index=role_to_index,
                genre_to_index=genre_to_index,
                primer_mode=args.primer_mode,
                primer_len=args.primer_len,
                role_sequences_cache=role_sequences_cache,
                min_body_tokens=args.min_body_tokens,
                target_seconds=args.target_seconds,
                max_melody_leap=args.max_melody_leap,
                candidates_per_sample=args.candidates_per_sample,
                diversity_jitter=args.diversity_jitter,
            )
            sample_name = f"sample_{i:02d}_{args.role.lower()}_{args.genre.lower()}"
            out_path = out_dir / f"{sample_name}.mid"
            save_single(tokens, out_path, args.role.upper())
            save_metadata(out_dir, sample_name, args, {args.role.upper(): tokens})
            if ended_by_eos:
                eos_count += 1
        print(f"Saved: {out_path}")

    print(f"EOS rate: {eos_count / max(1, args.samples):.3f}")


if __name__ == "__main__":
    main()
