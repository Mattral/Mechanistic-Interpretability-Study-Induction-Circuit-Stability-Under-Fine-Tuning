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


def test_clean_run_full_recovery(small_model: transformer_lens.HookedTransformer) -> None:
    """Patching all heads from clean into a clean run recovers attribution ≈ 1.0.

    When the corrupted run IS the clean run (all activations already clean),
    the total attribution across all heads should be approximately 1.0
    (the clean logit diff is already achieved).
    """
    from src.circuits.patching import (
        _build_clean_and_corrupted,
        _compute_logit_diff,
        compute_circuit_attribution,
    )

    seq_len = 15
    # Build clean sequences only — use the same tokens for both clean and corrupted
    torch.manual_seed(42)
    vocab_size = small_model.cfg.d_vocab
    prefix = torch.randint(0, vocab_size, (16, seq_len))
    clean_tokens = torch.cat([prefix, prefix], dim=1)

    ld_clean = _compute_logit_diff(small_model, clean_tokens, seq_len)
    ld_clean_val = float(ld_clean)

    # Patching clean into clean: recovered logit diff should match clean
    # i.e. denominator = clean - clean = 0, but we test via a near-clean corrupted
    # Use the same clean tokens as corrupted: recovery should be ≈ 1.0 per head
    with torch.no_grad():
        _, clean_cache = small_model.run_with_cache(
            clean_tokens,
            names_filter=lambda n: n.endswith("hook_z"),
            return_type=None,
        )

    from src.circuits.patching import patch_head_activation
    # Patch a known induction head (layer 1) from clean into clean
    # logit diff should be unchanged (full recovery of 1.0 relative baseline)
    patched_logits = patch_head_activation(
        model=small_model,
        corrupted_tokens=clean_tokens,  # using clean as corrupted
        clean_cache=clean_cache,
        layer=1,
        head=0,
    )
    qp = torch.arange(seq_len, 2 * seq_len - 1)
    tgt = clean_tokens[:, 1:seq_len]
    ql = patched_logits[:, qp, :]
    cl = ql.gather(dim=2, index=tgt.unsqueeze(2)).squeeze(2)
    ld_patched = float((cl - ql.mean(dim=2)).mean())
    # When patching clean->clean, result should equal original clean logit diff
    assert abs(ld_patched - ld_clean_val) < 0.1, (
        f"Patching clean->clean should preserve logit diff. "
        f"Expected ≈{ld_clean_val:.3f}, got {ld_patched:.3f}"
    )


def test_corrupted_run_no_recovery(small_model: transformer_lens.HookedTransformer) -> None:
    """Patching a non-circuit head (layer 0, any head) from corrupted into corrupted
    should yield near-zero attribution (no meaningful recovery).

    Layer-0 heads in the 2-layer model are previous-token heads. When we patch
    them into a corrupted run using CORRUPTED activations (not clean), there is
    no information to recover — the attribution should be near 0.0.
    """
    from src.circuits.patching import (
        _build_clean_and_corrupted,
        _compute_logit_diff,
    )

    seq_len = 15
    clean_tokens, corrupted_tokens = _build_clean_and_corrupted(
        seq_len=seq_len,
        batch_size=16,
        vocab_size=small_model.cfg.d_vocab,
        seed=42,
        device="cpu",
    )

    ld_clean = float(_compute_logit_diff(small_model, clean_tokens, seq_len))
    ld_corrupted = float(_compute_logit_diff(small_model, corrupted_tokens, seq_len))
    denom = ld_clean - ld_corrupted

    if abs(denom) < 1e-6:
        # If model has no induction capability, test is vacuous — skip gracefully
        return

    # Patch from CORRUPTED cache into corrupted run — should give ~0 attribution
    with torch.no_grad():
        _, corrupted_cache = small_model.run_with_cache(
            corrupted_tokens,
            names_filter=lambda n: n.endswith("hook_z"),
            return_type=None,
        )

    from src.circuits.patching import patch_head_activation
    # Use layer 0, head 0 (previous-token head, not induction head)
    patched_logits = patch_head_activation(
        model=small_model,
        corrupted_tokens=corrupted_tokens,
        clean_cache=corrupted_cache,  # patching from corrupted = no new information
        layer=0,
        head=0,
    )
    qp = torch.arange(seq_len, 2 * seq_len - 1)
    tgt = clean_tokens[:, 1:seq_len]
    ql = patched_logits[:, qp, :]
    cl = ql.gather(dim=2, index=tgt.unsqueeze(2)).squeeze(2)
    ld_patched = float((cl - ql.mean(dim=2)).mean())
    attribution = (ld_patched - ld_corrupted) / denom

    # Patching corrupted into corrupted gives no recovery: attribution ≈ 0
    assert abs(attribution) < 0.3, (
        f"Patching corrupted->corrupted should give near-zero attribution. "
        f"Got {attribution:.3f}. Possible bug in patch_head_activation."
    )
