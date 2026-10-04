import torch
import torch.nn as nn

from minitransformer.blocks import DecoderBlock, EncoderBlock, causal_mask, padding_mask
from minitransformer.layers import Embedding


class TransformerClassifier(nn.Module):
    """Encoder-only Transformer; classifies a sequence from the hidden state of its first ([CLS]) token."""

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        norm_first: bool = False,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, max_length, hidden_dim, dropout)
        self.layers = nn.ModuleList(
            EncoderBlock(hidden_dim, num_heads, dropout, norm_first) for _ in range(num_layers)
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask=None):
        mask = padding_mask(attention_mask) if attention_mask is not None else None
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, mask)
        return self.head(x[:, 0])  # (batch, num_classes)


class Seq2SeqTransformer(nn.Module):
    """Encoder-decoder Transformer with a language-modelling head over the target vocabulary."""

    def __init__(
        self,
        vocab_size: int,
        src_max_length: int,
        tgt_max_length: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        norm_first: bool = False,
    ):
        super().__init__()
        self.tgt_max_length = tgt_max_length
        self.src_embedding = Embedding(vocab_size, src_max_length, hidden_dim, dropout)
        self.tgt_embedding = Embedding(vocab_size, tgt_max_length, hidden_dim, dropout)
        self.encoder = nn.ModuleList(
            EncoderBlock(hidden_dim, num_heads, dropout, norm_first) for _ in range(num_layers)
        )
        self.decoder = nn.ModuleList(
            DecoderBlock(hidden_dim, num_heads, dropout, norm_first) for _ in range(num_layers)
        )
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def encode(self, src_ids, src_mask):
        mask = padding_mask(src_mask)
        x = self.src_embedding(src_ids)
        for layer in self.encoder:
            x = layer(x, mask)
        return x

    def decode(self, tgt_ids, memory, src_mask, tgt_mask=None):
        """Logits for the next token at every target position, shape (batch, tgt_len, vocab)."""
        tgt_len = tgt_ids.shape[1]
        self_mask = causal_mask(tgt_len, tgt_ids.device)[None, None]  # (1, 1, t, t)
        if tgt_mask is not None:
            self_mask = self_mask & padding_mask(tgt_mask)
        memory_mask = padding_mask(src_mask)
        x = self.tgt_embedding(tgt_ids)
        for layer in self.decoder:
            x = layer(x, memory, self_mask, memory_mask)
        return self.lm_head(x)

    def forward(self, src_ids, tgt_ids, src_mask, tgt_mask=None):
        return self.decode(tgt_ids, self.encode(src_ids, src_mask), src_mask, tgt_mask)

    @torch.no_grad()
    def generate(self, src_ids, src_mask, bos_id: int, eos_id: int, max_new_tokens: int | None = None):
        """Greedy decoding for a batch; sequences that emitted ``eos_id`` keep emitting it."""
        max_new_tokens = min(max_new_tokens or self.tgt_max_length, self.tgt_max_length) - 1
        memory = self.encode(src_ids, src_mask)
        out = torch.full((src_ids.shape[0], 1), bos_id, dtype=torch.long, device=src_ids.device)
        done = torch.zeros(src_ids.shape[0], dtype=torch.bool, device=src_ids.device)
        for _ in range(max_new_tokens):
            next_token = self.decode(out, memory, src_mask)[:, -1].argmax(dim=-1)
            next_token = torch.where(done, torch.full_like(next_token, eos_id), next_token)
            out = torch.cat([out, next_token[:, None]], dim=1)
            done |= next_token == eos_id
            if done.all():
                break
        return out
