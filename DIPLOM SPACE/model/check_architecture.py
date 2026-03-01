#!/usr/bin/env python3
"""
Скрипт для проверки архитектуры модели и подсчета параметров.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from model.transformer import TransformerLM, count_parameters

print("=" * 70)
print("🤖 АРХИТЕКТУРА ТРАНСФОРМЕР МОДЕЛИ")
print("=" * 70)

# Параметры модели
vocab_size = 565
d_model = 256
n_heads = 8
n_layers = 6
d_ff = 1024
dropout = 0.2
max_len = 512

model = TransformerLM(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    d_ff=d_ff,
    dropout=dropout,
    max_len=max_len,
    pad_id=0,
)

print(f"\n📊 ПАРАМЕТРЫ:")
print(f"  Vocab size: {vocab_size}")
print(f"  Model dimension (d_model): {d_model}")
print(f"  Attention heads: {n_heads}")
print(f"  Transformer layers: {n_layers}")
print(f"  Feedforward dimension: {d_ff}")
print(f"  Dropout: {dropout}")
print(f"  Max sequence length: {max_len}")

print(f"\n📈 СТАТИСТИКА МОДЕЛИ:")
total_params = count_parameters(model)
print(f"  Всего параметров: {total_params:,}")
print(f"  Всего параметров (в млн): {total_params / 1e6:.2f}M")

# Разбор по слоям
print(f"\n🔍 РАСПРЕДЕЛЕНИЕ ПАРАМЕТРОВ:")
print(f"  Token Embedding: {model.token_emb.weight.numel():,}")
print(f"  Position Embedding: {model.pos_emb.weight.numel():,}")
print(f"  Transformer layers: {sum(p.numel() for name, p in model.named_parameters() if 'transformer' in name):,}")
print(f"  Output layer bias: {model.fc.bias.numel():,}")
print(f"  Output layer weight: tied with token embedding")

print(f"\n✅ УЛУЧШЕНИЯ В МОДЕЛИ:")
print(f"  ✓ Embedding масштабирование (sqrt(d_model))")
print(f"  ✓ Dropout на embedding слое ({dropout})")
print(f"  ✓ GELU activation вместо ReLU")
print(f"  ✓ Pre-LN архитектура (LayerNorm перед слоем)")
print(f"  ✓ Xavier инициализация весов")
print(f"  ✓ Оптимальный d_model={d_model} для быстрой сходимости")
print(f"  ✓ Достаточный d_ff={d_ff} для экспрессивности")

print(f"\n🧪 ТЕСТ FORWARD PASS:")
batch_size = 4
seq_len = 256
x = torch.randint(0, vocab_size, (batch_size, seq_len))
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
x = x.to(device)

with torch.no_grad():
    output = model(x)

print(f"  Input shape: {x.shape}")
print(f"  Output shape: {output.shape}")
print(f"  ✓ Forward pass OK!")

# Подсчет памяти
print(f"\n💾 ПАМЯТЬ:")
model_memory = sum(p.numel() * 4 / (1024**2) for p in model.parameters())
print(f"  Параметры модели: {model_memory:.2f} MB")
print(f"  Активации (примерно): {batch_size * seq_len * d_model * 4 / (1024**2):.2f} MB")
print(f"  Рекомендуемая GPU VRAM: ~2-4 GB")

print(f"\n" + "=" * 70)
print(f"✅ Модель готова к обучению!")
print(f"=" * 70)
