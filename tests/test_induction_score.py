"""Tests for the induction score metric."""
from __future__ import annotations
import pytest
import torch
import transformer_lens


@pytest.fixture(scope="module")
def small_model():
    model = transformer_lens.HookedTransformer.from_pretrained(
        "attn-only-2l", center_unembed=True, center_writing_weights=True,
        fold_ln=True, refactor_factored_attn_matrices=True,
    )
    model.eval()
    return model


def test_score_shape(small_model):
    from src.circuits.induction_score import compute_induction_score
    scores = compute_induction_score(small_model, sequence_length=20, num_sequences=10, seed=42, device="cpu")
    assert scores.shape == (small_model.cfg.n_layers, small_model.cfg.n_heads)


def test_score_range(small_model):
    from src.circuits.induction_score import compute_induction_score
    scores = compute_induction_score(small_model, sequence_length=20, num_sequences=10, seed=42, device="cpu")
    assert float(scores.min()) >= 0.0
    assert float(scores.max()) <= 1.0 + 1e-5


def test_layer1_has_high_induction_head(small_model):
    """attn-only-2l layer 1 must have at least one head with IS >= 0.5."""
    from src.circuits.induction_score import compute_induction_score
    scores = compute_induction_score(small_model, sequence_length=30, num_sequences=50, seed=42, device="cpu")
    assert float(scores[1].max()) >= 0.5, f"Expected IS >= 0.5 in L1, got {float(scores[1].max()):.3f}"


def test_score_reproducibility(small_model):
    from src.circuits.induction_score import compute_induction_score
    a = compute_induction_score(small_model, sequence_length=20, num_sequences=20, seed=42, device="cpu")
    b = compute_induction_score(small_model, sequence_length=20, num_sequences=20, seed=42, device="cpu")
    assert torch.allclose(a, b, atol=1e-6)


def test_raises_on_short_sequence(small_model):
    from src.circuits.induction_score import compute_induction_score
    with pytest.raises(ValueError, match="sequence_length"):
        compute_induction_score(small_model, sequence_length=1, num_sequences=5, seed=42, device="cpu")


def test_identify_induction_heads(small_model):
    from src.circuits.induction_score import compute_induction_score, identify_induction_heads
    scores = compute_induction_score(small_model, sequence_length=30, num_sequences=50, seed=42, device="cpu")
    heads = identify_induction_heads(scores, threshold=0.4)
    assert isinstance(heads, list)
    for lh in heads:
        assert len(lh) == 2
        assert 0 <= lh[0] < small_model.cfg.n_layers
        assert 0 <= lh[1] < small_model.cfg.n_heads
