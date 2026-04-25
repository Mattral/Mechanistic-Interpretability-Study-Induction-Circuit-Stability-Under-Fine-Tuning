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


def test_score_known_non_induction_head(small_model: transformer_lens.HookedTransformer) -> None:
    """Layer-0 heads (previous-token heads) must score below 0.3.

    In the pretrained attn-only-2l model, layer-0 heads are previous-token
    heads, not induction heads. Their mean induction score must be below 0.3.
    This guards against regressions that accidentally make all heads look like
    induction heads.
    """
    from src.circuits.induction_score import compute_induction_score

    scores = compute_induction_score(
        model=small_model,
        sequence_length=30,
        num_sequences=50,
        seed=42,
        device="cpu",
    )
    layer_0_mean = float(scores[0].mean())
    assert layer_0_mean < 0.3, (
        f"Layer-0 mean induction score should be < 0.3 (previous-token heads), "
        f"got {layer_0_mean:.3f}. Possible regression in induction score computation."
    )

def test_score_known_induction_head(small_model: transformer_lens.HookedTransformer) -> None:
    """Known induction head in attn-only-2l must score above 0.7.

    Spec Section 8.1: "known head scores above 0.7"

    The attn-only-2l pretrained model is specifically constructed to exhibit
    strong induction heads. At least one head in layer 1 must score > 0.7
    on the canonical repeated-token task. Scores below 0.7 indicate either
    a model loading error or a regression in induction score computation.
    """
    from src.circuits.induction_score import compute_induction_score

    scores = compute_induction_score(
        model=small_model,
        sequence_length=30,
        num_sequences=100,
        seed=42,
        device="cpu",
    )
    # Layer 1 contains the induction heads per Olsson et al. (2022)
    layer_1_max = float(scores[1].max())
    assert layer_1_max > 0.7, (
        f"Expected at least one layer-1 head with induction score > 0.7 "
        f"(spec Section 8.1), got max = {layer_1_max:.4f}. "
        "This may indicate a model loading error or a bug in compute_induction_score."
    )
