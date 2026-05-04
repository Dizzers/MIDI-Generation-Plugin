"""PyTorch Dataset wrapping a numpy object-array of token-lists produced by
dataset/chunk_tokens.py. Adds <BOS> at the head, <EOS> at the tail, pads to
max_len with <PAD>, and returns (x, y, genre_id) where (x, y) is the standard
shifted next-token target pair.

Augmentation (training only):
    - transpose: shift all NOTE_ON/NOTE_OFF pitches by +/- N semitones
    - time-stretch: scale TIME_SHIFT step counts by a factor; round to nearest
      available token
    - velocity-jitter: shift VELOCITY bin by +/- k

Augmentations preserve the prefix [<GENRE_*>, <KEY_*>] and respect token
boundaries (only matching token-classes get rewritten).
"""
from __future__ import annotations

import bisect
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MIDITokenDataset(Dataset):
    def __init__(
        self,
        chunks_path: str | Path,
        vocab_path: str | Path,
        max_len: int = 1024,
        seed: int = 42,
        augment: bool = True,
        augment_config: dict | None = None,
    ) -> None:
        self.max_len = int(max_len)
        self.augment = bool(augment)
        self.aug_cfg = augment_config or {
            "transpose_prob": 0.8,
            "transpose_range": 11,
            "time_stretch_prob": 0.45,
            "time_stretch_range": (0.88, 1.12),
            "velocity_jitter_prob": 0.35,
            "velocity_jitter": 2,
        }
        self.rng = random.Random(seed)

        with open(vocab_path, encoding="utf-8") as handle:
            vocab = json.load(handle)
        self.token2id: dict[str, int] = vocab["token2id"]
        self.id2token: dict[str, str] = vocab["id2token"]

        self.pad = self.token2id["<PAD>"]
        self.bos = self.token2id["<BOS>"]
        self.eos = self.token2id["<EOS>"]
        self.unk = self.token2id["<UNK>"]

        self.genre_tokens = sorted(t for t in self.token2id if t.startswith("<GENRE_"))
        self.genre_token_to_index = {t: i for i, t in enumerate(self.genre_tokens)}
        self.num_genres = max(1, len(self.genre_tokens))

        self.time_shift_steps = sorted(
            int(t.rsplit("_", 1)[-1], 16)
            for t in self.token2id
            if t.startswith("TIME_SHIFT_")
        )
        self.max_time_step = self.time_shift_steps[-1] if self.time_shift_steps else 32
        self.velocity_bins = sorted(
            int(t.rsplit("_", 1)[-1], 16)
            for t in self.token2id
            if t.startswith("VELOCITY_")
        )
        self.max_vel_bin = self.velocity_bins[-1] if self.velocity_bins else 7

        self.data = np.load(chunks_path, allow_pickle=True)

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def _hex(value: int, width: int) -> str:
        spec = f"#0{width}x"
        return format(value, spec)

    _PC_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    _PC_NAME_TO_INDEX = {name: idx for idx, name in enumerate(_PC_NAMES)}

    def _transpose_token(self, token: str, shift: int) -> str:
        if token.startswith("NOTE_ON_") or token.startswith("NOTE_OFF_"):
            try:
                pitch = int(token.rsplit("_", 1)[-1], 16)
            except ValueError:
                return token
            new_pitch = max(0, min(127, pitch + shift))
            kind = "NOTE_ON_" if token.startswith("NOTE_ON_") else "NOTE_OFF_"
            return f"{kind}{self._hex(new_pitch, 4)}"
        if token.startswith("<KEY_") and token != "<KEY_UNKNOWN>":
            inner = token[len("<KEY_"):-1]
            if inner.endswith("_MAJ"):
                pc_name, suffix = inner[:-4], "_MAJ"
            elif inner.endswith("_MIN"):
                pc_name, suffix = inner[:-4], "_MIN"
            else:
                return token
            pc = self._PC_NAME_TO_INDEX.get(pc_name)
            if pc is None:
                return token
            new_pc = (pc + shift) % 12
            return f"<KEY_{self._PC_NAMES[new_pc]}{suffix}>"
        return token

    def _stretch_time_token(self, token: str, factor: float) -> str:
        if not token.startswith("TIME_SHIFT_"):
            return token
        try:
            steps = int(token.rsplit("_", 1)[-1], 16)
        except ValueError:
            return token
        new_steps = max(1, int(round(steps * factor)))
        new_steps = min(new_steps, self.max_time_step)
        # snap to nearest available step in vocab
        idx = bisect.bisect_left(self.time_shift_steps, new_steps)
        if idx >= len(self.time_shift_steps):
            idx = len(self.time_shift_steps) - 1
        elif idx > 0:
            before = self.time_shift_steps[idx - 1]
            after = self.time_shift_steps[idx]
            if abs(new_steps - before) <= abs(after - new_steps):
                idx -= 1
        chosen = self.time_shift_steps[idx]
        return f"TIME_SHIFT_{self._hex(chosen, 6)}"

    def _jitter_velocity_token(self, token: str, jitter: int) -> str:
        if not token.startswith("VELOCITY_"):
            return token
        try:
            value = int(token.rsplit("_", 1)[-1], 16)
        except ValueError:
            return token
        new_value = max(0, min(self.max_vel_bin, value + jitter))
        return f"VELOCITY_{self._hex(new_value, 4)}"

    def _augment(self, tokens: list[str]) -> list[str]:
        if not self.augment:
            return tokens
        out = list(tokens)
        cfg = self.aug_cfg

        if self.rng.random() < cfg.get("transpose_prob", 0.0):
            r = int(cfg.get("transpose_range", 0))
            if r > 0:
                shift = self.rng.randint(-r, r)
                if shift != 0:
                    out = [self._transpose_token(tok, shift) for tok in out]

        if self.rng.random() < cfg.get("time_stretch_prob", 0.0):
            low, high = cfg.get("time_stretch_range", (1.0, 1.0))
            factor = self.rng.uniform(low, high)
            if abs(factor - 1.0) > 1e-3:
                out = [self._stretch_time_token(tok, factor) for tok in out]

        if self.rng.random() < cfg.get("velocity_jitter_prob", 0.0):
            jmax = int(cfg.get("velocity_jitter", 0))
            if jmax > 0:
                jitter = self.rng.randint(-jmax, jmax)
                if jitter != 0:
                    out = [self._jitter_velocity_token(tok, jitter) for tok in out]

        return out

    def _genre_index(self, tokens: list[str]) -> int:
        for tok in tokens:
            if tok.startswith("<GENRE_"):
                return self.genre_token_to_index.get(tok, 0)
        return 0

    def key_token_per_sample(self) -> list[str]:
        """Return the <KEY_*> token of each chunk (used by WeightedRandomSampler).

        chunk_tokens.py guarantees the key sits within the first few prefix
        positions; we fall back to <KEY_UNKNOWN> if absent.
        """
        keys: list[str] = []
        for tokens in self.data:
            found = "<KEY_UNKNOWN>"
            for tok in list(tokens)[:5]:
                if tok.startswith("<KEY_"):
                    found = tok
                    break
            keys.append(found)
        return keys

    def encode(self, tokens: list[str]) -> list[int]:
        ids = [self.bos]
        ids.extend(self.token2id.get(tok, self.unk) for tok in tokens)
        ids.append(self.eos)
        return ids[: self.max_len]

    def __getitem__(self, idx: int):
        tokens = list(self.data[idx])
        tokens = self._augment(tokens)
        genre_idx = self._genre_index(tokens)
        ids = self.encode(tokens)
        if len(ids) < self.max_len:
            ids = ids + [self.pad] * (self.max_len - len(ids))
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y, torch.tensor(genre_idx, dtype=torch.long)
