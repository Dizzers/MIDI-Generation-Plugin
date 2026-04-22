import bisect
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset

_USE_EXISTING_SAMPLES_PER_EPOCH = object()


class MIDIDataset(Dataset):
    def __init__(
        self,
        chunks_path,
        vocab_path,
        max_len=512,
        samples_per_epoch=None,
        seed=42,
        augment_config=None,
        data_override=None,
        apply_augmentation=True,
    ):
        self.max_len = max_len
        self.samples_per_epoch = samples_per_epoch
        self.rng = random.Random(seed)
        self.augment_config = augment_config or {}
        self.apply_augmentation = apply_augmentation
        self.base_seed = seed
        self._vocab_path = vocab_path
        self._chunks_path = chunks_path

        with open(vocab_path) as handle:
            vocab = json.load(handle)
            self.token2id = vocab["token2id"]

        self.pad = self.token2id["<PAD>"]
        self.bos = self.token2id["<BOS>"]
        self.eos = self.token2id["<EOS>"]
        self.unk = self.token2id["<UNK>"]

        self.genre_tokens = sorted(token for token in self.token2id if token.startswith("<GENRE_"))
        if not self.genre_tokens:
            self.genre_tokens = ["<GENRE_NONE>"]
        self.genre_to_index_by_token = {token: idx for idx, token in enumerate(self.genre_tokens)}
        self.genre_to_index = {token[7:-1]: idx for idx, token in enumerate(self.genre_tokens) if token.startswith("<GENRE_")}
        self.num_genres = max(1, len(self.genre_tokens))

        self.time_shift_values = sorted({self._parse_hex_token(token) for token in self.token2id if token.startswith("TIME_SHIFT_")})
        self.time_shift_values = [value for value in self.time_shift_values if value is not None]
        self.velocity_values = sorted({self._parse_hex_token(token) for token in self.token2id if token.startswith("VELOCITY_")})
        self.velocity_values = [value for value in self.velocity_values if value is not None]

        if data_override is None:
            self.data = np.load(chunks_path, allow_pickle=True)
        else:
            self.data = np.array(data_override, dtype=object)

        self.samples = self._build_samples()

    def clone_with_data(self, data_override, samples_per_epoch=_USE_EXISTING_SAMPLES_PER_EPOCH, apply_augmentation=None, seed_offset=0):
        next_samples_per_epoch = self.samples_per_epoch if samples_per_epoch is _USE_EXISTING_SAMPLES_PER_EPOCH else samples_per_epoch
        return MIDIDataset(
            chunks_path=self._chunks_path,
            vocab_path=self._vocab_path,
            max_len=self.max_len,
            samples_per_epoch=next_samples_per_epoch,
            seed=self.base_seed + seed_offset,
            augment_config=self.augment_config,
            data_override=data_override,
            apply_augmentation=self.apply_augmentation if apply_augmentation is None else apply_augmentation,
        )

    def _parse_hex_token(self, token):
        try:
            parts = token.split("_")
            return int(parts[-1], 16)
        except Exception:
            return None

    def _build_samples(self):
        indices = list(range(len(self.data)))
        self.rng.shuffle(indices)
        if self.samples_per_epoch is None or self.samples_per_epoch <= 0:
            return indices
        return indices[: min(len(indices), int(self.samples_per_epoch))]

    def encode(self, tokens):
        ids = [self.bos]
        ids.extend(self.token2id.get(token, self.unk) for token in tokens)
        ids.append(self.eos)
        return ids[: self.max_len]

    def _nearest_value(self, values, target):
        if not values:
            return target
        idx = bisect.bisect_left(values, target)
        if idx == 0:
            return values[0]
        if idx >= len(values):
            return values[-1]
        before = values[idx - 1]
        after = values[idx]
        return before if abs(target - before) <= abs(after - target) else after

    def _shift_note_token(self, token, semitones):
        if not (token.startswith("NOTE_ON_") or token.startswith("NOTE_OFF_")):
            return token
        try:
            pitch = int(token.split("_")[2], 16)
        except Exception:
            return token
        new_pitch = max(0, min(127, pitch + semitones))
        return f"{token.split('_')[0]}_{token.split('_')[1]}_{new_pitch:#04x}"

    def _stretch_time_token(self, token, factor):
        if not token.startswith("TIME_SHIFT_"):
            return token
        try:
            value = int(token.split("_")[2], 16)
        except Exception:
            return token
        new_value = max(1, int(round(value * factor)))
        new_value = self._nearest_value(self.time_shift_values, new_value)
        return f"TIME_SHIFT_{new_value:#06x}"

    def _jitter_velocity_token(self, token, jitter):
        if not token.startswith("VELOCITY_"):
            return token
        try:
            value = int(token.split("_")[1], 16)
        except Exception:
            return token
        max_value = max(self.velocity_values) if self.velocity_values else 7
        new_value = max(0, min(max_value, value + jitter))
        return f"VELOCITY_{new_value:#02x}"

    def _apply_augmentation(self, tokens):
        out = list(tokens)
        if not self.augment_config:
            return out

        if self.rng.random() < self.augment_config.get("transpose_prob", 0.0):
            transpose_range = int(self.augment_config.get("transpose_range", 0))
            if transpose_range > 0:
                shift = self.rng.randint(-transpose_range, transpose_range)
                if shift != 0:
                    out = [self._shift_note_token(token, shift) for token in out]

        if self.rng.random() < self.augment_config.get("time_stretch_prob", 0.0):
            low, high = self.augment_config.get("time_stretch_range", (1.0, 1.0))
            factor = self.rng.uniform(low, high)
            if factor != 1.0:
                out = [self._stretch_time_token(token, factor) for token in out]

        if self.rng.random() < self.augment_config.get("velocity_jitter_prob", 0.0):
            jitter_max = int(self.augment_config.get("velocity_jitter", 0))
            if jitter_max > 0:
                jitter = self.rng.randint(-jitter_max, jitter_max)
                if jitter != 0:
                    out = [self._jitter_velocity_token(token, jitter) for token in out]

        return out

    def _extract_genre_index(self, tokens):
        for token in tokens:
            if token.startswith("<GENRE_"):
                return self.genre_to_index_by_token.get(token, 0)
        return 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_idx = self.samples[idx]
        tokens = list(self.data[sample_idx])
        if self.apply_augmentation:
            tokens = self._apply_augmentation(tokens)

        genre_idx = self._extract_genre_index(tokens)
        ids = self.encode(tokens)
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [self.pad] * pad_len

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y, torch.tensor(genre_idx, dtype=torch.long)
