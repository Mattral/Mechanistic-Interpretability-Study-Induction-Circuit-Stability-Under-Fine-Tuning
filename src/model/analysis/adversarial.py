"""Adversarial prompt probes for induction circuit stress testing.

Builds >= 20 distinct sequence types that challenge the induction mechanism.
All probe scores are normalised against the clean_repeated baseline (1.0 = clean, 0.0 = no induction).
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import torch
import transformer_lens

from src.circuits.patching import _compute_logit_diff

logger = logging.getLogger(__name__)
MIN_PROMPT_PATTERNS = 20


def build_adversarial_suite(
    vocab_size: int,
    seq_len: int = 20,
    batch_size: int = 50,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Build the adversarial prompt token suite with >= MIN_PROMPT_PATTERNS types.

    Args:
        vocab_size: Model vocabulary size.
        seq_len: Half-length of each sequence.
        batch_size: Number of instances per prompt type.
        seed: Random seed.
        device: Torch device string.

    Returns:
        Dict mapping prompt type name -> token tensor [batch, 2*seq_len].
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    def _rand(shape):
        return torch.randint(0, vocab_size, shape, device=device, generator=rng)

    suite: dict[str, torch.Tensor] = {}
    prefix = _rand((batch_size, seq_len))

    # Baseline
    suite["clean_repeated"] = torch.cat([prefix, prefix], dim=1)

    # Single-token substitutions at start positions 0-4
    for pos in range(5):
        s = prefix.clone()
        s[:, pos] = _rand((batch_size,))
        suite[f"sub_pos_{pos}"] = torch.cat([prefix, s], dim=1)

    # Single-token substitutions at end positions -5 to -1
    for pos in range(-5, 0):
        s = prefix.clone()
        s[:, pos] = _rand((batch_size,))
        suite[f"sub_end_{abs(pos)}"] = torch.cat([prefix, s], dim=1)

    # Multi-token substitutions
    for n_subs in [2, 3]:
        s = prefix.clone()
        positions = torch.randperm(seq_len, generator=rng)[:n_subs]
        s[:, positions] = _rand((batch_size, n_subs))
        suite[f"multi_sub_{n_subs}"] = torch.cat([prefix, s], dim=1)

    # Structural perturbations
    suite["reversed_second_half"] = torch.cat([prefix, prefix.flip(dims=[1])], dim=1)
    for k in [1, 3]:
        suite[f"cyclic_shift_{k}"] = torch.cat([prefix, torch.roll(prefix, shifts=k, dims=1)], dim=1)

    # Competing patterns
    pa = _rand((batch_size, seq_len))
    pb = _rand((batch_size, seq_len))
    interleaved = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    interleaved[:, ::2] = pa[:, ::2]
    interleaved[:, 1::2] = pb[:, 1::2]
    suite["competing_a"] = torch.cat([pa, interleaved], dim=1)
    suite["competing_b"] = torch.cat([pb, interleaved], dim=1)

    # Long lags
    for lag in [2, 3]:
        if seq_len > lag + 2:
            pl = _rand((batch_size, seq_len + lag))
            suite[f"long_lag_{lag}"] = torch.cat([pl[:, :seq_len], pl[:, lag:lag + seq_len]], dim=1)

    # Degenerate cases
    suite["all_same_token"] = _rand((batch_size, 1)).expand(-1, 2 * seq_len).contiguous()
    suite["fully_random"] = _rand((batch_size, 2 * seq_len))

    assert len(suite) >= MIN_PROMPT_PATTERNS, (
        f"Need >= {MIN_PROMPT_PATTERNS} prompt types, got {len(suite)}"
    )
    logger.info("Adversarial suite: %d prompt types", len(suite))
    return suite


def evaluate_adversarial_suite(
    model: transformer_lens.HookedTransformer,
    suite: dict[str, torch.Tensor],
    seq_len: int = 20,
    device: Optional[str] = None,
) -> dict[str, float]:
    """Evaluate normalised logit diff on each probe type.

    Scores are normalised: 1.0 = clean induction, 0.0 = no induction.

    Raises:
        KeyError: If clean_repeated is not in suite.
    """
    if "clean_repeated" not in suite:
        raise KeyError("Suite must contain clean_repeated baseline.")
    if device is None:
        device = next(model.parameters()).device.type
    model.eval()

    with torch.no_grad():
        clean_val = float(_compute_logit_diff(model, suite["clean_repeated"].to(device), seq_len))

    scores = {}
    for name, tokens in suite.items():
        with torch.no_grad():
            raw = float(_compute_logit_diff(model, tokens.to(device), seq_len))
        scores[name] = raw / clean_val if clean_val != 0 else 0.0
    return scores


def compare_adversarial_pre_post(
    model_pre: transformer_lens.HookedTransformer,
    model_post: transformer_lens.HookedTransformer,
    vocab_size: int,
    seq_len: int = 20,
    batch_size: int = 50,
    seed: int = 42,
    device: Optional[str] = None,
) -> dict[str, dict[str, float]]:
    """Compare adversarial probe scores before and after fine-tuning.

    Returns:
        Dict with keys pre, post, delta — each mapping prompt name -> score.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    suite = build_adversarial_suite(vocab_size=vocab_size, seq_len=seq_len,
                                     batch_size=batch_size, seed=seed, device=device)
    pre = evaluate_adversarial_suite(model_pre, suite, seq_len=seq_len, device=device)
    post = evaluate_adversarial_suite(model_post, suite, seq_len=seq_len, device=device)
    delta = {k: post.get(k, 0.0) - pre.get(k, 0.0) for k in pre}
    return {"pre": pre, "post": post, "delta": delta}
