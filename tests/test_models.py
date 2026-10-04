import torch

from minitransformer import Seq2SeqTransformer, TransformerClassifier, causal_mask

torch.manual_seed(0)


def test_causal_mask():
    assert causal_mask(3).int().tolist() == [[1, 0, 0], [1, 1, 0], [1, 1, 1]]


def test_classifier_ignores_padding():
    model = TransformerClassifier(
        vocab_size=50, max_length=16, num_classes=3, hidden_dim=32, num_heads=4
    ).eval()
    ids = torch.tensor([[1, 7, 8, 9]])
    padded = torch.tensor([[1, 7, 8, 9, 0, 0]])
    out = model(ids, torch.ones_like(ids))
    out_padded = model(padded, torch.tensor([[1, 1, 1, 1, 0, 0]]))
    assert out.shape == (1, 3)
    assert torch.allclose(out, out_padded, atol=1e-5)


def make_seq2seq():
    return Seq2SeqTransformer(
        vocab_size=40, src_max_length=20, tgt_max_length=10, hidden_dim=32, num_heads=4
    ).eval()


def test_decoder_cannot_see_the_future():
    model = make_seq2seq()
    src = torch.randint(3, 40, (1, 6))
    src_mask = torch.ones_like(src)
    tgt = torch.randint(3, 40, (1, 5))
    changed = tgt.clone()
    changed[0, 3:] = (changed[0, 3:] + 1) % 40
    a, b = model(src, tgt, src_mask), model(src, changed, src_mask)
    assert torch.allclose(a[:, :3], b[:, :3], atol=1e-5)
    assert not torch.allclose(a[:, 3:], b[:, 3:])


def test_seq2seq_ignores_source_padding():
    model = make_seq2seq()
    src = torch.randint(3, 40, (1, 6))
    padded = torch.cat([src, torch.zeros(1, 4, dtype=torch.long)], dim=1)
    tgt = torch.randint(3, 40, (1, 4))
    a = model(src, tgt, torch.ones_like(src))
    b = model(padded, tgt, torch.cat([torch.ones(1, 6), torch.zeros(1, 4)], dim=1).long())
    assert torch.allclose(a, b, atol=1e-5)


def test_generate_respects_length_and_starts_with_bos():
    model = make_seq2seq()
    src = torch.randint(3, 40, (2, 6))
    out = model.generate(src, torch.ones_like(src), bos_id=0, eos_id=2)
    assert out.shape[0] == 2 and out.shape[1] <= 10
    assert torch.all(out[:, 0] == 0)


def test_pre_norm_variant_runs():
    model = TransformerClassifier(50, 16, 3, hidden_dim=32, num_heads=4, num_layers=2, norm_first=True)
    assert model(torch.randint(0, 50, (2, 5))).shape == (2, 3)
