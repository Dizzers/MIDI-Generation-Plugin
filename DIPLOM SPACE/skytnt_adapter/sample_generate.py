"""
Smoke-test SkyTNT generation locally.

Runs either:
  * `--mode torch`  -> uses the original PyTorch MIDIModel (sanity check)
  * `--mode onnx`   -> uses ONNX Runtime + the exported model_*.onnx
                      (validates the same generation path the C++ plugin uses)

Outputs N MIDI files into `artifacts/examples/`.

Usage:
    python -m skytnt_adapter.sample_generate \
        --mode onnx \
        --base artifacts/onnx/model_base.onnx \
        --token artifacts/onnx/model_token.onnx \
        --tokenizer artifacts/tokenizer/tokenizer_config.json \
        --out artifacts/examples \
        --num 2 --max-len 256
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import skytnt_adapter  # noqa: F401  - sys.path bootstrap

import numpy as np

import MIDI
from midi_tokenizer import MIDITokenizer


def _build_tokenizer_from_dump(path: str):
    """Reconstruct a tokenizer from the JSON dumped by export_skytnt_onnx."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    tok = MIDITokenizer(d["version"])
    tok.set_optimise_midi(bool(d.get("optimise_midi", False)))
    return tok


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - np.amax(x, axis=axis, keepdims=True)
    np.exp(x, out=x)
    return x / np.sum(x, axis=axis, keepdims=True)


def _sample_top_p_k(probs: np.ndarray, p: float, k: int, rng: np.random.Generator):
    probs_idx = np.argsort(-probs, axis=-1)
    probs_sort = np.take_along_axis(probs, probs_idx, -1)
    probs_sum = np.cumsum(probs_sort, axis=-1)
    mask = (probs_sum - probs_sort) > p
    probs_sort[mask] = 0.0
    cutoff_mask = np.zeros(probs_sort.shape[-1])
    cutoff_mask[:k] = 1
    probs_sort = probs_sort * cutoff_mask
    probs_sort /= np.sum(probs_sort, axis=-1, keepdims=True) + 1e-12
    flat_probs = probs_sort.reshape(-1, probs_sort.shape[-1])
    flat_idx = probs_idx.reshape(-1, probs_idx.shape[-1])
    out = np.stack([rng.choice(idxs, p=pvals) for pvals, idxs in zip(flat_probs, flat_idx)])
    return out.reshape(*probs_sort.shape[:-1])


def _empty_past_kv(num_layers: int, num_heads: int, head_size: int,
                   batch: int = 1, dtype=np.float32):
    return [
        (np.zeros((batch, num_heads, 0, head_size), dtype=dtype),
         np.zeros((batch, num_heads, 0, head_size), dtype=dtype))
        for _ in range(num_layers)
    ]


def _generate_onnx(base_session, token_session, tokenizer, *,
                   max_len: int, temperature: float, top_p: float, top_k: int,
                   seed: int):
    """Mirror upstream `app_onnx.generate` without IO-binding (CPU path)."""
    import onnxruntime as ort
    rng = np.random.default_rng(seed)
    max_token_seq = tokenizer.max_token_seq

    # Inspect base model layer/head dims so we can shape past_kv tensors.
    base_inputs = {i.name: i for i in base_session.get_inputs()}
    base_outputs = {o.name: o for o in base_session.get_outputs()}
    token_inputs = {i.name: i for i in token_session.get_inputs()}
    token_outputs = {o.name: o for o in token_session.get_outputs()}

    base_kv_keys = sorted([n for n in base_inputs if n.startswith("past_key_values")])
    base_present_keys = sorted([n for n in base_outputs if n.startswith("present")])
    token_kv_keys = sorted([n for n in token_inputs if n.startswith("past_key_values")])
    token_present_keys = sorted([n for n in token_outputs if n.startswith("present")])

    # Shapes: (batch, num_heads, past_seq, head_size). We pull num_heads/head_size
    # from the static dims of the input tensor descriptors.
    def _dims(name, model_inputs):
        s = model_inputs[name].shape
        # s typically = ['batch', num_heads, 'past_seq', head_size]
        return int(s[1]), int(s[3])

    base_nh, base_hs = _dims(base_kv_keys[0], base_inputs)
    token_nh, token_hs = _dims(token_kv_keys[0], token_inputs)
    base_layers = len(base_kv_keys) // 2
    token_layers = len(token_kv_keys) // 2

    base_past = _empty_past_kv(base_layers, base_nh, base_hs)
    base_past_dict = {}
    for i, (k, v) in enumerate(base_past):
        base_past_dict[f"past_key_values.{i}.key"] = k
        base_past_dict[f"past_key_values.{i}.value"] = v

    # Initial input: BOS row.
    input_tensor = np.full((1, max_token_seq), tokenizer.pad_id, dtype=np.int64)
    input_tensor[0, 0] = tokenizer.bos_id
    input_tensor = input_tensor[None, :, :]  # (1, mid_seq=1, max_token_seq)
    cur_len = input_tensor.shape[1]
    past_len = 0

    while cur_len < max_len:
        feed = {"x": input_tensor[:, past_len:].astype(np.int64)}
        feed.update(base_past_dict)
        outs = base_session.run(None, feed)
        # outs ordered as [hidden, present.0.key, present.0.value, ...]
        hidden = outs[0]
        new_present = outs[1:]
        base_past_dict = {}
        for i in range(base_layers):
            base_past_dict[f"past_key_values.{i}.key"] = new_present[2 * i]
            base_past_dict[f"past_key_values.{i}.value"] = new_present[2 * i + 1]

        last_hidden = hidden[:, -1:]  # (1, 1, n_embd)

        token_past = _empty_past_kv(token_layers, token_nh, token_hs)
        token_past_dict = {}
        for i, (k, v) in enumerate(token_past):
            token_past_dict[f"past_key_values.{i}.key"] = k
            token_past_dict[f"past_key_values.{i}.value"] = v

        next_token_seq = np.zeros((1, 0), dtype=np.int64)
        event_name = ""
        ended = False
        cur_hidden = last_hidden

        for i in range(max_token_seq):
            mask = np.zeros((1, tokenizer.vocab_size), dtype=np.int64)
            if ended:
                mask[0, tokenizer.pad_id] = 1
            elif i == 0:
                allowed = list(tokenizer.event_ids.values()) + [tokenizer.eos_id]
                mask[0, allowed] = 1
            else:
                params = tokenizer.events[event_name]
                if i > len(params):
                    mask[0, tokenizer.pad_id] = 1
                else:
                    mask[0, tokenizer.parameter_ids[params[i - 1]]] = 1
            mask = mask[:, None, :]

            x = next_token_seq
            if i != 0:
                if i == 1:
                    cur_hidden = np.zeros((1, 0, last_hidden.shape[-1]), dtype=np.float32)
                x = x[:, -1:]

            feed = {"hidden": cur_hidden.astype(np.float32),
                    "x": x.astype(np.int64)}
            feed.update(token_past_dict)
            outs = token_session.run(None, feed)
            logits = outs[0]
            new_present = outs[1:]
            token_past_dict = {}
            for li in range(token_layers):
                token_past_dict[f"past_key_values.{li}.key"] = new_present[2 * li]
                token_past_dict[f"past_key_values.{li}.value"] = new_present[2 * li + 1]

            scores = _softmax(logits / temperature, axis=-1) * mask
            sample = _sample_top_p_k(scores, top_p, top_k, rng)
            if i == 0:
                next_token_seq = sample
                eid = int(sample[0, 0])
                if eid == tokenizer.eos_id:
                    ended = True
                else:
                    event_name = tokenizer.id_events[eid]
            else:
                next_token_seq = np.concatenate([next_token_seq, sample], axis=1)
                if not ended and len(tokenizer.events[event_name]) == i:
                    break

        if next_token_seq.shape[1] < max_token_seq:
            next_token_seq = np.pad(
                next_token_seq,
                ((0, 0), (0, max_token_seq - next_token_seq.shape[1])),
                mode="constant", constant_values=tokenizer.pad_id,
            )
        next_token_seq = next_token_seq[:, None, :]
        input_tensor = np.concatenate([input_tensor, next_token_seq], axis=1)
        past_len = cur_len
        cur_len += 1
        if ended:
            break

    return input_tensor[0]


def _generate_torch(ckpt_path, config_name, *, max_len, temperature, top_p, top_k, seed, device):
    import torch
    from midi_model import MIDIModel, MIDIModelConfig
    config = MIDIModelConfig.from_name(config_name)
    model = MIDIModel(config).to(device=device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state_dict = {k.replace("model.", "", 1) if k.startswith("model.") else k: v
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    out = model.generate(batch_size=1, max_len=max_len, temp=temperature,
                         top_p=top_p, top_k=top_k, generator=g)
    return out[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["onnx", "torch"], default="onnx")
    parser.add_argument("--base", type=str, default="artifacts/onnx/model_base.onnx")
    parser.add_argument("--token", type=str, default="artifacts/onnx/model_token.onnx")
    parser.add_argument("--tokenizer", type=str,
                        default="artifacts/tokenizer/tokenizer_config.json")
    parser.add_argument("--ckpt", type=str, default="checkpoints/skytnt/model.ckpt",
                        help="(torch mode) checkpoint path")
    parser.add_argument("--config", type=str, default="tv2o-medium",
                        help="(torch mode) config name")
    parser.add_argument("--out", type=str, default="artifacts/examples")
    parser.add_argument("--num", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.94)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    opt = parser.parse_args()
    os.makedirs(opt.out, exist_ok=True)

    if opt.mode == "onnx":
        try:
            import onnxruntime as ort
        except ImportError:
            print("onnxruntime is required for --mode onnx", file=sys.stderr)
            sys.exit(2)
        if not os.path.exists(opt.base) or not os.path.exists(opt.token):
            print(f"ONNX models missing: {opt.base} / {opt.token}", file=sys.stderr)
            sys.exit(2)
        if not os.path.exists(opt.tokenizer):
            print(f"tokenizer config missing: {opt.tokenizer}", file=sys.stderr)
            sys.exit(2)
        tokenizer = _build_tokenizer_from_dump(opt.tokenizer)
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        base_sess = ort.InferenceSession(opt.base, sess_opts)
        token_sess = ort.InferenceSession(opt.token, sess_opts)
    else:
        from midi_model import MIDIModelConfig
        config = MIDIModelConfig.from_name(opt.config)
        tokenizer = config.tokenizer
        if not os.path.exists(opt.ckpt):
            print(f"checkpoint missing: {opt.ckpt}", file=sys.stderr)
            sys.exit(2)

    for i in range(opt.num):
        seed = opt.seed + i
        t0 = time.time()
        if opt.mode == "onnx":
            seq = _generate_onnx(base_sess, token_sess, tokenizer,
                                 max_len=opt.max_len, temperature=opt.temperature,
                                 top_p=opt.top_p, top_k=opt.top_k, seed=seed)
        else:
            seq = _generate_torch(opt.ckpt, opt.config,
                                  max_len=opt.max_len, temperature=opt.temperature,
                                  top_p=opt.top_p, top_k=opt.top_k, seed=seed,
                                  device="cpu")
        dt = time.time() - t0
        score = tokenizer.detokenize(seq.tolist() if hasattr(seq, 'tolist') else seq)
        out_path = os.path.join(opt.out, f"sample_{i:02d}.mid")
        with open(out_path, "wb") as f:
            f.write(MIDI.score2midi(score))
        print(f"[sample] {out_path} ({dt:.1f}s, {len(seq)} steps)")

    print("[sample] done.")


if __name__ == "__main__":
    main()
