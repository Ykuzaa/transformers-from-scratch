import torch
import torch.nn as nn

from minitransformer.layers import FeedForward, LayerNorm, MultiHeadAttention


def causal_mask(size: int, device=None) -> torch.Tensor:
    """Lower-triangular boolean mask: position i may attend to positions <= i."""
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))


def padding_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """(batch, seq_len) 1/0 mask -> (batch, 1, 1, seq_len), broadcast over heads and queries."""
    return attention_mask[:, None, None, :].bool()


class _Residual(nn.Module):
    """``x + sublayer(x)`` with LayerNorm after (post-norm) or before (pre-norm) the sublayer."""

    def __init__(self, hidden_dim: int, dropout: float, norm_first: bool):
        super().__init__()
        self.norm = LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x, sublayer):
        if self.norm_first:
            return x + self.dropout(sublayer(self.norm(x)))
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderBlock(nn.Module):
    """Self-attention + feed-forward, each wrapped in a residual connection and LayerNorm."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, norm_first: bool = False):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = FeedForward(hidden_dim, dropout=dropout)
        self.res_attention = _Residual(hidden_dim, dropout, norm_first)
        self.res_feed_forward = _Residual(hidden_dim, dropout, norm_first)
        self.final_norm = LayerNorm(hidden_dim) if norm_first else nn.Identity()

    def forward(self, x, mask=None):
        x = self.res_attention(x, lambda h: self.attention(h, h, h, mask))
        x = self.res_feed_forward(x, self.feed_forward)
        return self.final_norm(x)


class DecoderBlock(nn.Module):
    """Masked self-attention, cross-attention over the encoder output, then feed-forward."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, norm_first: bool = False):
        super().__init__()
        self.self_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = FeedForward(hidden_dim, dropout=dropout)
        self.res_self = _Residual(hidden_dim, dropout, norm_first)
        self.res_cross = _Residual(hidden_dim, dropout, norm_first)
        self.res_feed_forward = _Residual(hidden_dim, dropout, norm_first)
        self.final_norm = LayerNorm(hidden_dim) if norm_first else nn.Identity()

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        """``tgt_mask`` must already include the causal mask; ``memory_mask`` hides source padding."""
        x = self.res_self(x, lambda h: self.self_attention(h, h, h, tgt_mask))
        x = self.res_cross(x, lambda h: self.cross_attention(h, memory, memory, memory_mask))
        x = self.res_feed_forward(x, self.feed_forward)
        return self.final_norm(x)
