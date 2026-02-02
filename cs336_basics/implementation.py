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

    




    