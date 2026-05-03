"""Causal Transformer LM with the exact forward signature the existing C++
plugin expects:

    auto out = module.forward({x, g}).toTensor();   // (B, T, V)

where x is int64 [B, T] of token ids and g is int64 [B] of genre indices.
The whole class must remain torch.jit.script-compatible because
DIPLOM SPACE/plugin/juce/Source/ModelInference.cpp loads the resulting
TorchScript module via torch::jit::load.

We avoid any non-scriptable constructs: only nn.Embedding, nn.LayerNorm,
nn.TransformerEncoder/EncoderLayer, learned positional embedding, and a
weight-tied linear head.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class TransformerLM(nn.Module):
    """Decoder-only LM implemented via nn.TransformerEncoder + causal mask.

    Args:
        vocab_size: total number of tokens in vocab.json
        num_genres: number of <GENRE_*> tokens (>=1, currently 1)
        d_model, n_heads, n_layers, d_ff, dropout, max_len: standard
        pad_id: id of <PAD>; if not None, used for src_key_padding_mask
    """

    def __init__(
        self,
        vocab_size: int,
        num_genres: int = 1,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.2,
        max_len: int = 1024,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if num_genres < 1:
            raise ValueError("num_genres must be >= 1")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_len = max_len
        self.pad_id = pad_id

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.genre_emb = nn.Embedding(num_genres, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.scale = math.sqrt(d_model)
        self._reset_parameters()
        self.fc.weight = self.token_emb.weight  # weight tying

    def _reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _causal_mask(self, T: int, device: torch.device) -> Tensor:
        # nn.TransformerEncoder with batch_first expects float mask of shape (T, T)
        # where -inf disallows attention; bool mask also works (True = mask).
        mask = torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)
        return mask

    def forward(self, x: Tensor, genre_id: Tensor) -> Tensor:
        """x: (B, T) long, genre_id: (B,) long. Returns (B, T, V) logits."""
        B = x.size(0)
        T = x.size(1)
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} exceeds max_len {self.max_len}")

        pos = torch.arange(T, device=x.device).unsqueeze(0)

        tok = self.token_emb(x) * self.scale
        pos_e = self.pos_emb(pos)
        gen_e = self.genre_emb(genre_id).unsqueeze(1)
        h = tok + pos_e + gen_e
        h = self.emb_dropout(h)

        causal_mask = self._causal_mask(T, h.device)
        key_padding_mask = x.eq(self.pad_id)

        h = self.transformer(h, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        return self.fc(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
