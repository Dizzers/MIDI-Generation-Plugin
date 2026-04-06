import json
import random
import bisect
import numpy as np
import torch
from torch.utils.data import Dataset


class MIDIDataset(Dataset):
    def __init__(
        self,
        chunks_dir,
        vocab_path,
        max_len=512,
        samples_per_epoch=3000,
        sampling_strategy="balanced",
        seed=42,
        augment_config=None,
        role_sampling_max_fraction=None,
        data_override=None,
        apply_augmentation=True,
    ):
        self.max_len = max_len
        self.samples_per_epoch = samples_per_epoch
        self.sampling_strategy = sampling_strategy
        self.rng = random.Random(seed)
        self.augment_config = augment_config or {}
        self.role_sampling_max_fraction = role_sampling_max_fraction or {}
        self.apply_augmentation = apply_augmentation

        self.base_seed = seed
        self._vocab_path = vocab_path

        with open(vocab_path) as f:
            vocab = json.load(f)
            self.token2id = vocab["token2id"]

        self.pad = self.token2id["<PAD>"]
        self.bos = self.token2id["<BOS>"]
        self.eos = self.token2id["<EOS>"]
        self.unk = self.token2id["<UNK>"]

        self.data = {}
        if data_override is not None:
            for role in ["chords", "melody", "bass"]:
                self.data[role] = np.array(data_override.get(role, []), dtype=object)
        else:
            for role in ["chords", "melody", "bass"]:
                path = f"{chunks_dir}/{role}_chunks.npy"
                self.data[role] = np.load(path, allow_pickle=True)

        self.roles = list(self.data.keys())
        self.role_to_index = {"melody": 0, "bass": 1, "chords": 2}

        self.genre_tokens = sorted([t for t in self.token2id.keys() if t.startswith("<GENRE_")])
        if not self.genre_tokens:
            self.genre_tokens = ["<GENRE_NONE>"]
        self.genre_to_index_by_token = {tok: i for i, tok in enumerate(self.genre_tokens)}
        self.genre_to_index = {tok[7:-1]: i for i, tok in enumerate(self.genre_tokens) if tok.startswith("<GENRE_")}
        self.num_genres = max(1, len(self.genre_tokens))

        self.time_shift_values = sorted({self._parse_hex_token(t) for t in self.token2id if t.startswith("TIME_SHIFT_")})
        self.time_shift_values = [v for v in self.time_shift_values if v is not None]
        self.velocity_values = sorted({self._parse_hex_token(t) for t in self.token2id if t.startswith("VELOCITY_")})
        self.velocity_values = [v for v in self.velocity_values if v is not None]

        self.samples = self._build_samples()

    def clone_with_data(self, data_override, samples_per_epoch=None, apply_augmentation=None, seed_offset=0):
        return MIDIDataset(
            chunks_dir="",
            vocab_path=self._vocab_path,
            max_len=self.max_len,
            samples_per_epoch=self.samples_per_epoch if samples_per_epoch is None else samples_per_epoch,
            sampling_strategy=self.sampling_strategy,
            seed=self.base_seed + seed_offset,
            augment_config=self.augment_config,
            role_sampling_max_fraction=self.role_sampling_max_fraction,
            data_override=data_override,
            apply_augmentation=self.apply_augmentation if apply_augmentation is None else apply_augmentation,
        )

    def _parse_hex_token(self, token):
        try:
            parts = token.split("_")
            if len(parts) >= 3:
                return int(parts[2], 16)
            if len(parts) == 2:
                return int(parts[1], 16)
            return None
        except Exception:
            return None

    def _build_samples(self):
        role_to_items = {
            role: [(role, i) for i in range(len(self.data[role]))]
            for role in self.roles
        }
        all_items = []
        for role in self.roles:
            all_items.extend(role_to_items[role])

        if self.samples_per_epoch is None or self.samples_per_epoch <= 0:
            target_size = len(all_items)
        else:
            target_size = int(self.samples_per_epoch)

        if self.sampling_strategy == "balanced":
            per_role = max(1, target_size // len(self.roles))
            sampled = []
            for role in self.roles:
                pool = role_to_items[role]
                if len(pool) >= per_role:
                    sampled.extend(self.rng.sample(pool, per_role))
                else:
                    sampled.extend(self.rng.choices(pool, k=per_role))
            if len(sampled) < target_size:
                extra = target_size - len(sampled)
                sampled.extend(self.rng.choices(all_items, k=extra))
            self.rng.shuffle(sampled)
            return sampled[:target_size]

        if self.sampling_strategy == "role_cap":
            return self._sample_with_role_cap(role_to_items, all_items, target_size)

        self.rng.shuffle(all_items)
        return all_items[:min(target_size, len(all_items))]

    def _sample_with_role_cap(self, role_to_items, all_items, target_size):
        default_cap = 1.0 / len(self.roles)
        caps = {role: self.role_sampling_max_fraction.get(role, default_cap) for role in self.roles}
        total_cap = sum(caps.values())
        if total_cap <= 0:
            caps = {role: default_cap for role in self.roles}
            total_cap = 1.0
        caps = {role: v / total_cap for role, v in caps.items()}

        sampled = []
        remaining = target_size
        for role in self.roles:
            cap_n = max(1, int(target_size * caps[role]))
            pool = role_to_items[role]
            if len(pool) >= cap_n:
                sampled.extend(self.rng.sample(pool, cap_n))
            else:
                sampled.extend(self.rng.choices(pool, k=cap_n))
            remaining -= cap_n

        if remaining > 0:
            sampled.extend(self.rng.choices(all_items, k=remaining))

        self.rng.shuffle(sampled)
        return sampled[:target_size]

    def encode(self, tokens):
        ids = [self.bos]
        for t in tokens:
            ids.append(self.token2id.get(t, self.unk))
        ids.append(self.eos)
        return ids[:self.max_len]

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

    def _shift_note_token(self, tok, semitones):
        if not (tok.startswith("NOTE_ON_") or tok.startswith("NOTE_OFF_")):
            return tok
        try:
            pitch = int(tok.split("_")[2], 16)
        except Exception:
            return tok
        new_pitch = max(0, min(127, pitch + semitones))
        return f"{tok.split('_')[0]}_{tok.split('_')[1]}_{new_pitch:02X}"

    def _stretch_time_token(self, tok, factor):
        if not tok.startswith("TIME_SHIFT_"):
            return tok
        try:
            val = int(tok.split("_")[2], 16)
        except Exception:
            return tok
        new_val = max(1, int(round(val * factor)))
        new_val = self._nearest_value(self.time_shift_values, new_val)
        return f"TIME_SHIFT_{new_val:02X}"

    def _jitter_velocity_token(self, tok, jitter):
        if not tok.startswith("VELOCITY_"):
            return tok
        try:
            val = int(tok.split("_")[1], 16)
        except Exception:
            return tok
        new_val = max(0, min(max(self.velocity_values) if self.velocity_values else 7, val + jitter))
        return f"VELOCITY_{new_val:02X}"

    def _apply_augmentation(self, tokens):
        tokens_out = list(tokens)
        if not self.augment_config:
            return tokens_out

        if self.rng.random() < self.augment_config.get("transpose_prob", 0.0):
            tr = int(self.augment_config.get("transpose_range", 0))
            if tr > 0:
                semitones = self.rng.randint(-tr, tr)
                if semitones != 0:
                    tokens_out = [self._shift_note_token(t, semitones) for t in tokens_out]

        if self.rng.random() < self.augment_config.get("time_stretch_prob", 0.0):
            low, high = self.augment_config.get("time_stretch_range", (1.0, 1.0))
            if low != 1.0 or high != 1.0:
                factor = self.rng.uniform(low, high)
                tokens_out = [self._stretch_time_token(t, factor) for t in tokens_out]

        if self.rng.random() < self.augment_config.get("velocity_jitter_prob", 0.0):
            jitter = int(self.augment_config.get("velocity_jitter", 0))
            if jitter > 0:
                j = self.rng.randint(-jitter, jitter)
                if j != 0:
                    tokens_out = [self._jitter_velocity_token(t, j) for t in tokens_out]

        return tokens_out

    def _extract_genre_index(self, tokens):
        for t in tokens:
            if t.startswith("<GENRE_"):
                return self.genre_to_index_by_token.get(t, 0)
        return 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        role, item_idx = self.samples[idx]
        tokens = self.data[role][item_idx]

        if self.apply_augmentation:
            tokens = self._apply_augmentation(tokens)

        role_idx = self.role_to_index.get(role, 0)
        genre_idx = self._extract_genre_index(tokens)

        ids = self.encode(tokens)
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [self.pad] * pad_len

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y, torch.tensor(role_idx, dtype=torch.long), torch.tensor(genre_idx, dtype=torch.long)
