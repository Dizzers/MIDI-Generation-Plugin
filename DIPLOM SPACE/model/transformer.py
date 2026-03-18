import torch
import torch.nn as nn
import math


class TransformerLM(nn.Module):
 
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.2,
        max_len=512,
        use_checkpoint=False,
        pad_id=None,
        num_roles=0,
        num_genres=0,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.use_checkpoint = use_checkpoint
        self.pad_id = pad_id

        self.num_roles = int(num_roles or 0)
        self.num_genres = int(num_genres or 0)

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.role_emb = nn.Embedding(self.num_roles, d_model) if self.num_roles > 0 else None
        self.genre_emb = nn.Embedding(self.num_genres, d_model) if self.num_genres > 0 else None

        self.emb_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )

        self.fc = nn.Linear(d_model, vocab_size)
        self.scale = math.sqrt(d_model)
        self.register_buffer(
            "_causal_mask_cache",
            torch.empty(0, 0, dtype=torch.bool),
            persistent=False
        )

        self._init_weights()
        self.fc.weight = self.token_emb.weight

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def _generate_causal_mask(self, T, device):
        if self._causal_mask_cache.size(0) < T or self._causal_mask_cache.device != device:
            mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
            self._causal_mask_cache = mask
        return self._causal_mask_cache[:T, :T]

    def forward(self, x, role_id=None, genre_id=None):
        """
        Args:
            x: (B, T) токены
            role_id: (B,) индексы ролей
            genre_id: (B,) индексы жанров
        Returns:
            logits: (B, T, vocab_size)
        """
        input_ids = x
        B, T = input_ids.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} exceeds max_len {self.max_len}")

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)

        token_emb = self.token_emb(input_ids) * self.scale
        pos_emb = self.pos_emb(pos)

        x = token_emb + pos_emb

        if self.role_emb is not None and role_id is not None:
            role_vec = self.role_emb(role_id).unsqueeze(1)
            x = x + role_vec
        if self.genre_emb is not None and genre_id is not None:
            genre_vec = self.genre_emb(genre_id).unsqueeze(1)
            x = x + genre_vec

        x = self.emb_dropout(x)

        causal_mask = self._generate_causal_mask(T, x.device)
        key_padding_mask = (input_ids == self.pad_id) if self.pad_id is not None else None

        if self.use_checkpoint and self.training:
            x = self.transformer(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        else:
            x = self.transformer(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)

        logits = self.fc(x)
        return logits


class ReferenceEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(ReferenceEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, emb_dim)

    def forward(self, x):
        return self.fc(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
