import pytest
import torch
import torch.nn as nn

from minitransformer import Embedding, FeedForward, LayerNorm, MultiHeadAttention


def test_attention_shapes_self_and_cross():
    mha = MultiHeadAttention(64, 8)
    x, memory = torch.randn(2, 5, 64), torch.randn(2, 9, 64)
    assert mha(x, x, x).shape == (2, 5, 64)
    out, weights = mha(x, memory, memory, return_weights=True)
    assert out.shape == (2, 5, 64)
    assert weights.shape == (2, 8, 5, 9)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 8, 5))


def test_masked_keys_get_zero_weight():
    mha = MultiHeadAttention(32, 4)
    x = torch.randn(1, 4, 32)
    mask = torch.tensor([1, 1, 1, 0]).view(1, 1, 1, 4)
    _, weights = mha(x, x, x, mask, return_weights=True)
    assert torch.all(weights[..., 3] == 0)


def test_head_dim_must_divide():
    with pytest.raises(ValueError):
        MultiHeadAttention(30, 8)


def test_layernorm_matches_pytorch():
    x = torch.randn(3, 7, 16)
    assert torch.allclose(LayerNorm(16)(x), nn.LayerNorm(16)(x), atol=1e-5)


def test_feedforward_and_embedding_shapes():
    assert FeedForward(16)(torch.randn(2, 3, 16)).shape == (2, 3, 16)
    assert Embedding(100, 10, 16)(torch.randint(0, 100, (2, 7))).shape == (2, 7, 16)
