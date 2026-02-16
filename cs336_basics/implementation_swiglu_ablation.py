import torch
import torch.nn as nn
import torch.nn.functional as F

from cs336_basics.implementation import (
    Linear,
    Embedding,
    RMSNorm,
    RotaryPositionEmbedding,
    MultiHeadSelfAttention,
)

class FeedForwardSiLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope=None, device=None, dtype=None):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.rmsnorm1 = RMSNorm(d_model, device=device, dtype=dtype)
        
        self.ffn = FeedForwardSiLU(d_model, d_ff, device=device, dtype=dtype)
        
        self.rmsnorm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, mask=None, token_positions=None) -> torch.Tensor:
        x_norm = self.rmsnorm1(x)
        attn_output = self.attention(x_norm, mask=mask, token_positions=token_positions)
        x = x + attn_output
        
        x_norm = self.rmsnorm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, max_seq_len, device=None, dtype=None):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionEmbedding(theta=rope_theta, d_k=d_model // num_heads, max_seq_len=max_seq_len, device=device)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope=self.rope, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_linear = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids, mask=None):
        x = self.embedding(token_ids)
        batch_size, seq_len, _ = x.size()
        token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        for block in self.transformer_blocks:
            x = block(x, mask=mask, token_positions=token_positions)

        x = self.final_norm(x)
        return self.output_linear(x)