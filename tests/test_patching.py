"""Tests for activation patching and circuit attribution."""
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


def test_clean_corrupted_shapes(small_model):
    from src.circuits.patching import _build_clean_and_corrupted
    clean, corrupted = _build_clean_and_corrupted(10, 8, small_model.cfg.d_vocab, 42, "cpu")
    assert clean.shape == (8, 20)
    assert corrupted.shape == (8, 20)


def test_clean_is_repeated(small_model):
    from src.circuits.patching import _build_clean_and_corrupted
    clean, _ = _build_clean_and_corrupted(15, 16, small_model.cfg.d_vocab, 42, "cpu")
    assert torch.all(clean[:, :15] == clean[:, 15:])


def test_attribution_shape(small_model):
    from src.circuits.patching import compute_circuit_attribution
    scores = compute_circuit_attribution(small_model, seq_len=10, batch_size=8, seed=42, device="cpu")
    assert scores.shape == (small_model.cfg.n_layers, small_model.cfg.n_heads)


def test_circuit_head_attribution(small_model):
    from src.circuits.patching import CIRCUIT_THRESHOLD, compute_circuit_attribution
    scores = compute_circuit_attribution(small_model, seq_len=15, batch_size=16, seed=42, device="cpu")
    assert float(scores.max()) >= CIRCUIT_THRESHOLD, f"Expected attribution >= {CIRCUIT_THRESHOLD}, got {float(scores.max()):.3f}"


def test_attribution_reproducibility(small_model):
    from src.circuits.patching import compute_circuit_attribution
    a = compute_circuit_attribution(small_model, seq_len=10, batch_size=8, seed=42, device="cpu")
    b = compute_circuit_attribution(small_model, seq_len=10, batch_size=8, seed=42, device="cpu")
    assert torch.allclose(a, b, atol=1e-5)


def test_get_circuit_heads_sorted(small_model):
    from src.circuits.patching import compute_circuit_attribution, get_circuit_heads
    scores = compute_circuit_attribution(small_model, seq_len=10, batch_size=8, seed=42, device="cpu")
    heads = get_circuit_heads(scores)
    assert heads == sorted(heads)
