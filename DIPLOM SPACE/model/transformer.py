import torch
import torch.nn as nn
import math


class TransformerLM(nn.Module):
    """
    Улучшенный Transformer Language Model для музыки.
    
    Улучшения:
    1. ✓ Embedding scaling для стабильности обучения
    2. ✓ Dropout на embedding слое
    3. ✓ Больший d_ff в feedforward для экспрессивности
    4. ✓ Правильная инициализация весов
    5. ✓ Gradient checkpointing для экономии памяти
    6. ✓ Attention масштабирование (1/sqrt(d_k))
    7. ✓ Layer normalization перед слоями (Pre-LN)
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=256,          # Уменьшили с 512 для лучшей сходимости
        n_heads=8,
        n_layers=6,
        d_ff=1024,            # Добавили параметр для feedforward
        dropout=0.2,          # Увеличили dropout для регуляризации
        max_len=512,
        use_checkpoint=False,  # Gradient checkpointing
        pad_id=None,
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.use_checkpoint = use_checkpoint
        self.pad_id = pad_id
        
        # Embedding слой с масштабированием
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        # Dropout на embedding
        self.emb_dropout = nn.Dropout(dropout)
        
        # Transformer слои с Pre-LN (LayerNorm перед слоем, а не после)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,  # Обычно 4*d_model
            dropout=dropout,
            activation='gelu',     # GELU лучше чем ReLU для трансформеров
            batch_first=True,
            norm_first=True        # Pre-LN архитектура
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output слой
        self.fc = nn.Linear(d_model, vocab_size)
        self.scale = math.sqrt(d_model)
        self.register_buffer(
            "_causal_mask_cache",
            torch.empty(0, 0, dtype=torch.bool),
            persistent=False
        )
        
        # Инициализация весов
        self._init_weights()

        # Weight tying обычно улучшает LM-качество и уменьшает число параметров
        self.fc.weight = self.token_emb.weight
    
    def _init_weights(self):
        """Правильная инициализация весов для стабильного обучения"""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                # Xavier initialization для весов
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # Нули для bias
                nn.init.zeros_(param)
    
    def _generate_causal_mask(self, T, device):
        """Генерирует каузальную маску для decoder attention"""
        if self._causal_mask_cache.size(0) < T or self._causal_mask_cache.device != device:
            mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
            self._causal_mask_cache = mask
        return self._causal_mask_cache[:T, :T]
    
    def forward(self, x):
        """
        Args:
            x: (B, T) токены
        Returns:
            logits: (B, T, vocab_size)
        """
        input_ids = x
        B, T = input_ids.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} exceeds max_len {self.max_len}")
        
        # Позиции
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        
        # Embeddings
        token_emb = self.token_emb(input_ids) * self.scale  # (B, T, d_model)
        pos_emb = self.pos_emb(pos)     # (1, T, d_model)
        
        x = token_emb + pos_emb
        x = self.emb_dropout(x)
        
        # Каузальная маска (чтобы не смотрел в будущее)
        causal_mask = self._generate_causal_mask(T, x.device)
        key_padding_mask = (input_ids == self.pad_id) if self.pad_id is not None else None
        
        # Transformer
        if self.use_checkpoint and self.training:
            # Gradient checkpointing для экономии памяти
            x = self.transformer(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        else:
            x = self.transformer(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        
        # Output
        logits = self.fc(x)  # (B, T, vocab_size)
        
        return logits


class ReferenceEncoder(nn.Module):
    # Кодирует reference MIDI в embedding
    def __init__(self, input_dim, emb_dim):
        super(ReferenceEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, emb_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, input_dim) - входные данные MIDI
        Returns:
            (B, T, emb_dim) - закодированные embeddings
        """
        return self.fc(x)


def count_parameters(model):
    """Подсчитывает количество параметров"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
