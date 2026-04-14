import argparse
import json
import random
import sys
from pathlib import Path

import music21 as m21
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.transformer import TransformerLM

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

TIME_SHIFT_RESOLUTION = 0.05


def clean_state_dict(state_dict):
    if not state_dict:
        return state_dict
    first = next(iter(state_dict.keys()))
    if first.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


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


def sample_next(logits, temperature=1.0, top_k=16, top_p=0.93):
    t = max(1e-5, float(temperature))
    adj = logits / t
    adj = top_k_top_p_filter(adj, top_k=top_k, top_p=top_p)
    probs = torch.softmax(adj, dim=-1)
    if torch.isnan(probs).any() or probs.sum() <= 0:
        return torch.argmax(logits).item()
    return torch.multinomial(probs, 1).item()


def load_model_and_vocab(checkpoint_path, vocab_path):
    with open(vocab_path) as f:
        vocab = json.load(f)
    token2id = vocab["token2id"]
    id2token = {int(k): v for k, v in vocab["id2token"].items()}

    cp = torch.load(checkpoint_path, map_location=DEVICE)
    cfg = cp.get("model_config", {}) if isinstance(cp, dict) else {}
    model = TransformerLM(
        vocab_size=len(token2id),
        d_model=cfg.get("d_model", 256),
        n_heads=cfg.get("n_heads", 8),
        n_layers=cfg.get("n_layers", 6),
        d_ff=cfg.get("d_ff", 1024),
        dropout=cfg.get("dropout", 0.2),
        max_len=cfg.get("max_len", 256),
        pad_id=cfg.get("pad_id", token2id.get("<PAD>")),
        num_roles=0,
        num_genres=cfg.get("num_genres", 1),
    ).to(DEVICE)

    state = cp.get("model_state_dict", cp) if isinstance(cp, dict) else cp
    model.load_state_dict(clean_state_dict(state))

    genre_tokens = sorted([t for t in token2id if t.startswith("<GENRE_")])
    genre_to_idx = {t[7:-1]: i for i, t in enumerate(genre_tokens)}

    return model, token2id, id2token, genre_to_idx


def tokens_to_part(tokens):
    part = m21.stream.Part()
    part.partName = "FULL"
    current_offset = 0.0
    active_notes = {}
    last_velocity = 80

    for tok in tokens:
        if tok.startswith("TIME_SHIFT_"):
            step = int(tok.split("_")[2], 16)
            current_offset += step * TIME_SHIFT_RESOLUTION
        elif tok.startswith("VELOCITY_"):
            b = int(tok.split("_")[1], 16)
            last_velocity = max(1, min(127, int(round((b / 7) * 127))))
        elif tok.startswith("NOTE_ON_"):
            pitch = int(tok.split("_")[2], 16)
            active_notes[pitch] = (current_offset, last_velocity)
        elif tok.startswith("NOTE_OFF_"):
            pitch = int(tok.split("_")[2], 16)
            state = active_notes.pop(pitch, None)
            if state is None:
                continue
            start, vel = state
            dur = max(TIME_SHIFT_RESOLUTION, current_offset - start)
            n = m21.note.Note(pitch)
            n.offset = start
            n.quarterLength = dur
            n.volume.velocity = vel
            part.append(n)

    if active_notes:
        final_offset = current_offset + (2 * TIME_SHIFT_RESOLUTION)
        for pitch, (start, vel) in active_notes.items():
            n = m21.note.Note(pitch)
            n.offset = start
            n.quarterLength = max(TIME_SHIFT_RESOLUTION, final_offset - start)
            n.volume.velocity = vel
            part.append(n)

    return part


def generate_one(model, token2id, id2token, genre, max_len=256, temperature=1.0, top_k=16, top_p=0.93):
    genre_token = f"<GENRE_{genre.upper()}>"
    key_tokens = [t for t in token2id if t.startswith("<KEY_")]
    key_token = random.choice(key_tokens) if key_tokens else "<KEY_UNKNOWN>"

    for tok in ["<BOS>", genre_token, key_token]:
        if tok not in token2id:
            raise ValueError(f"Missing token: {tok}")

    generated = [token2id["<BOS>"], token2id[genre_token], token2id[key_token]]
    eos_id = token2id.get("<EOS>")

    genre_tokens = sorted([t for t in token2id if t.startswith("<GENRE_")])
    genre_idx = genre_tokens.index(genre_token) if genre_token in genre_tokens else 0
    genre_id = torch.tensor([genre_idx], dtype=torch.long, device=DEVICE)

    model.eval()
    with torch.no_grad():
        for _ in range(max_len):
            context = generated[-model.max_len:]
            x = torch.tensor(context, dtype=torch.long, device=DEVICE).unsqueeze(0)
            logits = model(x, role_id=None, genre_id=genre_id)[0, -1]
            nxt = sample_next(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            generated.append(nxt)
            if eos_id is not None and nxt == eos_id:
                break

    return [id2token[i] for i in generated]


def parse_args():
    p = argparse.ArgumentParser("Generate full MIDI")
    p.add_argument("--checkpoint", type=str, default="checkpoints/model_best_full.pth")
    p.add_argument("--vocab", type=str, default="dataset/processed/vocab_full.json")
    p.add_argument("--genre", type=str, default="TRAP")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--max-len", type=int, default=192)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--top-p", type=float, default=0.93)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", type=str, default="generated_full")
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model, token2id, id2token, _genre_to_idx = load_model_and_vocab(PROJECT_ROOT / args.checkpoint, PROJECT_ROOT / args.vocab)

    print(f"Device: {DEVICE}")
    for i in range(1, args.samples + 1):
        toks = generate_one(
            model,
            token2id,
            id2token,
            genre=args.genre,
            max_len=args.max_len,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        part = tokens_to_part(toks)
        score = m21.stream.Score()
        score.append(part)
        out_path = out_dir / f"sample_{i:02d}_full_{args.genre.lower()}.mid"
        score.write("midi", fp=str(out_path))

        meta = {
            "config": vars(args),
            "length_tokens": len(toks),
            "ended_by_eos": bool(toks and toks[-1] == "<EOS>"),
        }
        with open(out_dir / f"sample_{i:02d}_full_{args.genre.lower()}.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
