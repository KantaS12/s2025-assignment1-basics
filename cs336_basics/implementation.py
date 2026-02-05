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


# Cross Entropy Implementation
def cross_entropy(predicted_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

    # Predicted logits (o_i) is in R^(vocab_size)
    # Targets is a sequence x of length m + 1 and i = 1, ..., m

    # Cross Entropy Equation: l_i = -log softmax(o_i)[x_(i+1)]

    # 1. We want to subtract the largest element for numerical stability

    # 2. Cancel out log and exp if possible

    # 3. Handle any additional batch dimensions and return the average across the batch. 

    # 3etc. assume batch-like dimensions come first before vocab size dimension

    largest_logits = torch.max(predicted_logits, dim=-1, keepdim=True).values

    stabalized_logits = predicted_logits - largest_logits

    exp_logits = torch.exp(stabalized_logits)

    sum_exp_logits = torch.sum(exp_logits, dim=-1, keepdim=True)

    log_softmax = stabalized_logits - torch.log(sum_exp_logits)

    # output[i,j] = A[i, I[i, j]] where A is log_softmax and I is targets (I is index tensor)
    target_log_probs = log_softmax.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # Calcualte Cross Entropy Loss
    cross_entropy_loss = -torch.mean(target_log_probs)

    return cross_entropy_loss
 

# AdamW Optimizer
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, weight_decay: float = 0.01, betas = (0.9, 0.999), eps: float = 1e-8):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # Set defaults
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }

        super().__init__(params, defaults=defaults)

    def step(self, closure = None):
        loss = None if closure is None else closure()

        # Iterate over parameter groups
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['betas'][0]
            beta2 = group['betas'][1]
            eps = group['eps']
            weight_decay = group['weight_decay']

            # Iterate over parameters in this group
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # Get state for this parameter (creates empty dict if first time)
                state = self.state[p]

                # Initialize state on first step
                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                # Get state variables
                t = state['t']
                m = state['m']
                v = state['v']

                # Increment timestep
                t += 1
                state['t'] = t

                # Update biased first moment estimate
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second moment estimate
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected learning rate
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                # Update parameters:
                p.data.addcdiv_(m, torch.sqrt(v) + eps, value=-lr_t)

                # Apply weight decay:
                p.data.mul_(1 - lr * weight_decay)

        return loss


# Learning Rate Scheduling
def learning_rate_schedule(step: int, max_learning_rate: float, minimum_learning_rate: float, warmup_iterations: int, cos_anneal_iterations: int) -> float:
    # Warmup Phase
    if step < warmup_iterations:
        lr = max_learning_rate * (step / warmup_iterations)
    # Cosine Annealing Phase  
    elif step < cos_anneal_iterations:
    
        decay_duration = cos_anneal_iterations - warmup_iterations
        
        progress = (step - warmup_iterations) / decay_duration
        cosine_term = math.cos(progress * math.pi)
        
        lr = minimum_learning_rate + 0.5 * (1 + cosine_term) * (max_learning_rate - minimum_learning_rate)
    # Post Annealing Phase
    else:
        lr = minimum_learning_rate

    return lr


# Gradient Clipping
def gradient_clipping(parameters, max_norm: float):
    result_norm = 0.0

    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            result_norm += param_norm.item() ** 2

    # Compute the total norm
    result_norm = result_norm ** 0.5

    # Clip the gradients if the norm exceeds the maximum
    if result_norm > max_norm:
        clip_value = max_norm / result_norm
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_value)

    return result_norm


