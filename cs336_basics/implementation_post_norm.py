import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import typing
import argparse
import time
import logging

from cs336_basics.implementation import (
    AdamW, 
    cross_entropy, 
    learning_rate_schedule, 
    gradient_clipping,
    data_loading,
    save_checkpoint,
    load_checkpoint,
    Linear,
    Embedding,
    RMSNorm,
    SwiGLU,
    RotaryPositionEmbedding,
    MultiHeadSelfAttention
)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RotaryPositionEmbedding | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.rmsnorm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.rmsnorm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, token_positions: torch.Tensor = None) -> torch.Tensor:
        # Multi-head Self-Attention with Residual Connection
        # x_norm = self.rmsnorm1(x)
        attn_output = self.attention(x, mask=mask, token_positions=token_positions) # Pass x directly
        x = x + attn_output
        x = self.rmsnorm1(x)

        # Feed-Forward Network with Residual Connection
        # x_norm = self.rmsnorm2(x)
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.rmsnorm2(x)

        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, max_seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(TransformerLM, self).__init__()

        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.rope = RotaryPositionEmbedding(theta=rope_theta, d_k=d_model // num_heads, max_seq_len=max_seq_len, device=device)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope=self.rope, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)

        self.output_linear = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(token_ids)

        batch_size, seq_len, _ = x.size()
        token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        for block in self.transformer_blocks:
            x = block(x, mask=mask, token_positions=token_positions)

        x = self.final_norm(x)
        logits = self.output_linear(x)

        return logits