import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention split over ``num_heads`` heads.

    ``q`` attends over ``k``/``v``: pass the same tensor three times for
    self-attention, or decoder states and encoder states for cross-attention.
    ``mask`` is broadcastable to ``(batch, heads, q_len, k_len)``; positions
    where it is 0/False get zero attention weight.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.W_query = nn.Linear(hidden_dim, hidden_dim)
        self.W_key = nn.Linear(hidden_dim, hidden_dim)
        self.W_value = nn.Linear(hidden_dim, hidden_dim)
        self.W_out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, q, k, v, mask=None, return_weights: bool = False):
        queries = self._split_heads(self.W_query(q))  # (b, h, q_len, d_head)
        keys = self._split_heads(self.W_key(k))  # (b, h, k_len, d_head)
        values = self._split_heads(self.W_value(v))

        scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.head_dim)  # (b, h, q_len, k_len)
        if mask is not None:
            # Most negative finite value rather than -inf: a fully masked row then
            # yields uniform weights instead of NaN.
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)

        out = self.dropout(weights) @ values  # (b, h, q_len, d_head)
        out = out.transpose(1, 2).reshape(q.shape[0], q.shape[1], self.hidden_dim)
        out = self.W_out(out)
        return (out, weights) if return_weights else out


class LayerNorm(nn.Module):
    """Normalises each token over its features, then applies a learned scale and shift."""

    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(hidden_dim))
        self.shift = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class FeedForward(nn.Module):
    """Position-wise MLP: expand to ``expansion * hidden_dim``, ReLU, project back."""

    def __init__(self, hidden_dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, expansion * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.layers(x)


class Embedding(nn.Module):
    """Token embedding plus learned absolute position embedding."""

    def __init__(self, vocab_size: int, max_length: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.token = nn.Embedding(vocab_size, hidden_dim)
        self.position = nn.Embedding(max_length, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        return self.dropout(self.token(ids) + self.position(positions))
