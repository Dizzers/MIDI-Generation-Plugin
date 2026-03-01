import json
import random
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
    ):
        self.max_len = max_len
        self.samples_per_epoch = samples_per_epoch
        self.sampling_strategy = sampling_strategy
        self.rng = random.Random(seed)

        with open(vocab_path) as f:
            vocab = json.load(f)
            self.token2id = vocab["token2id"]

        self.pad = self.token2id["<PAD>"]
        self.bos = self.token2id["<BOS>"]
        self.eos = self.token2id["<EOS>"]
        self.unk = self.token2id["<UNK>"]

        self.data = {}
        for role in ["chords", "melody", "bass"]:
            path = f"{chunks_dir}/{role}_chunks.npy"
            self.data[role] = np.load(path, allow_pickle=True)

        self.roles = list(self.data.keys())
        self.samples = self._build_samples()

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
            target_size = min(int(self.samples_per_epoch), len(all_items))

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

        self.rng.shuffle(all_items)
        return all_items[:target_size]

    def encode(self, tokens):
        ids = [self.bos]
        for t in tokens:
            ids.append(self.token2id.get(t, self.unk))
        ids.append(self.eos)
        return ids[:self.max_len]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        role, item_idx = self.samples[idx]
        tokens = self.data[role][item_idx]

        ids = self.encode(tokens)
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [self.pad] * pad_len

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y
