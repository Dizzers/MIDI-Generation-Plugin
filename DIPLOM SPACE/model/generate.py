import argparse
import json
import random
import sys
from pathlib import Path

import music21 as m21
import torch

PROJECT_ROOT = Path(__file__).parent.parent
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


def sample_next_token(
    logits,
    generated,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    repetition_penalty=1.0,
    no_repeat_ngram_size=0,
):
    temperature = max(1e-5, float(temperature))
    adjusted = logits / temperature
    adjusted = apply_repetition_penalty(adjusted, generated, repetition_penalty)

    banned = get_banned_next_tokens(generated, no_repeat_ngram_size)
    if banned:
        ban_tensor = torch.tensor(list(banned), device=adjusted.device, dtype=torch.long)
        adjusted.index_fill_(0, ban_tensor, float("-inf"))

    filtered = top_k_top_p_filter(adjusted, top_k=top_k, top_p=top_p)
    probs = torch.softmax(filtered, dim=-1)
    if torch.isnan(probs).any() or probs.sum() <= 0:
        return torch.argmax(adjusted).item()
    return torch.multinomial(probs, 1).item()


def generate_tokens(
    model,
    token2id,
    id2token,
    role="MELODY",
    genre="TRAP",
    max_len=512,
    temperature=1.0,
    top_k=20,
    top_p=0.95,
    repetition_penalty=1.1,
    no_repeat_ngram_size=4,
):
    role_token = f"<ROLE_{role}>"
    genre_token = f"<GENRE_{genre}>"
    for token in ["<BOS>", role_token, genre_token]:
        if token not in token2id:
            raise ValueError(f"Token {token} not found in vocab")

    generated = [token2id["<BOS>"], token2id[role_token], token2id[genre_token]]
    eos_id = token2id.get("<EOS>")

    model.eval()
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(DEVICE)
            logits = model(x)[0, -1]
            next_token_id = sample_next_token(
                logits=logits,
                generated=generated,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            generated.append(next_token_id)
            if eos_id is not None and next_token_id == eos_id:
                break

    ended_by_eos = bool(eos_id is not None and generated[-1] == eos_id)
    return [id2token[i] for i in generated], ended_by_eos


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
            last_velocity = int(bin_idx * (127 / 7))
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
            note.volume.velocity = max(1, min(127, velocity))
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
    model = TransformerLM(len(token2id), pad_id=token2id.get("<PAD>")).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(clean_state_dict(state_dict))
    return model, token2id, id2token


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MIDI samples")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_best.pth")
    parser.add_argument("--vocab", type=str, default="dataset/processed/vocab.json")
    parser.add_argument("--role", type=str, default="MELODY", choices=ROLES + ["ALL"])
    parser.add_argument("--genre", type=str, default="TRAP")
    parser.add_argument("--max-len", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="generated")
    return parser.parse_args()


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

    model, token2id, id2token = load_model_and_vocab(checkpoint_path, vocab_path)
    print(f"🖥️  Device: {DEVICE}")

    eos_count = 0
    for i in range(1, args.samples + 1):
        if args.role == "ALL":
            tokens_by_role = {}
            all_eos = True
            for role in ROLES:
                tokens, ended_by_eos = generate_tokens(
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
                )
                tokens_by_role[role] = tokens
                all_eos = all_eos and ended_by_eos
            out_path = out_dir / f"sample_{i:02d}_arrangement_{args.genre.lower()}.mid"
            save_arrangement(tokens_by_role, out_path)
            if all_eos:
                eos_count += 1
        else:
            tokens, ended_by_eos = generate_tokens(
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
            )
            out_path = out_dir / f"sample_{i:02d}_{args.role.lower()}_{args.genre.lower()}.mid"
            save_single(tokens, out_path, args.role.upper())
            if ended_by_eos:
                eos_count += 1
        print(f"🎵 Saved: {out_path}")

    print(f"📊 EOS rate: {eos_count / max(1, args.samples):.3f}")


if __name__ == "__main__":
    main()
