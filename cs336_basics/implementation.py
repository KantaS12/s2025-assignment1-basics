import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

# Linear layer implementation
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(Linear, self).__init__()
        self.weights = nn.Parameter(torch.rand((out_features, in_features), device=device, dtype=dtype))

        std = math.sqrt(2.0 / (in_features + out_features))

        limit = 3 * std

        nn.init.trunc_normal_(
            self.weights, 
            mean=0.0,
            std=std,
            a=-limit,
            b=limit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weights.T)


# Embedding layer implementation
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(Embedding, self).__init__()
        self.weights = nn.Parameter(torch.rand((num_embeddings, embedding_dim), device=device, dtype=dtype))

        nn.init.trunc_normal_(
            self.weights,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0
        )
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]


# RMS Normalization implementation
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype

        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

        rmsNorm = (x / rms) * self.scale

        return rmsNorm.to(in_dtype)


# SwiGLU activation function
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(SwiGLU, self).__init__()

        non_rounded_dff = (8 / 3) * d_model

        d_ff = ((int(non_rounded_dff) + 63) // 64) * 64

        self.w1 = nn.Parameter(torch.rand((d_ff, d_model), device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.rand((d_model, d_ff), device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.rand((d_ff, d_model), device=device, dtype=dtype))

        for w in [self.w1, self.w2, self.w3]:
            d_in, d_out = w.shape[1], w.shape[0]

            std = math.sqrt(2.0 / (d_in + d_out))
            limit = 3 * std

            nn.init.trunc_normal_(
                w,
                mean=0.0,
                std=std,
                a=-limit,
                b=limit
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        gate = F.linear(x, self.w1)
        gate = gate * torch.sigmoid(gate)

        hidden = F.linear(x, self.w3)

        element_wise = gate * hidden

        return F.linear(element_wise, self.w2)


# RoPE positional encoding implementation
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super(RotaryPositionEmbedding, self).__init__()

        theta_i_k = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))

        positions = torch.arange(0, max_seq_len, device=device).float()

        freqs = torch.einsum('i , j -> i j', positions, theta_i_k)

        double_freq = torch.repeat_interleave(freqs, repeats=2, dim=-1)

        self.register_buffer('cos', torch.cos(double_freq), persistent=False)
        self.register_buffer('sin', torch.sin(double_freq), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        x_rotated = torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)

        x_out = (x * cos) + (x_rotated * sin)

        return x_out


# Softmax function
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:

    # i-th dimension of input tensor x
    max_x = torch.max(x, dim=dim, keepdim=True).values

    # Ensure numerical stability subtracting maximum value of i-th dimension from all elements of i-th dimension
    x = x - max_x

    exp_x = torch.exp(x)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)

    return exp_x / sum_exp_x


# Scaled dot-product attention
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:

    # keys & queries of shape (batch_size, ..., seq_len, d_k)
    # values of shape (batch_size, ..., seq_len, d_v)

    d_k = q.size(-1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))

    attn_weights = softmax(scores, dim=-1)

    output = torch.matmul(attn_weights, v)

    return output


# Multi-head self-attention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionEmbedding | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(MultiHeadSelfAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope

        assert d_model % num_heads == 0, "d_model needs to be divisible"

        self.d_k = d_model // num_heads # dk = dv = dmodel/heads

        self.q_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_linear = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, token_positions: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # perform linear operation
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        if mask is None:
            mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))

        # apply attention 
        attn_output = scaled_dot_product_attention(q, k, v, mask=mask)

        # concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.out_linear(attn_output)

        return output

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RotaryPositionEmbedding | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.rmsnorm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.rmsnorm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, token_positions: torch.Tensor = None) -> torch.Tensor:
        # Multi-head Self-Attention with Residual Connection
        attn_output = self.attention(self.rmsnorm1(x), mask=mask, token_positions=token_positions)
        x = x + attn_output

        # Feed-Forward Network with Residual Connection
        ffn_output = self.ffn(self.rmsnorm2(x))
        x = x + ffn_output

        return x

# Transformer LM
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