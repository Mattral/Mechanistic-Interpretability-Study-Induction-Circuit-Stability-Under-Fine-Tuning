"""Induction score metric — canonical Olsson et al. (2022) definition.

For a repeated-token sequence [t_1,...,t_n, t_1,...,t_n], the induction score
for head (l,h) is the mean attention weight A^(l,h)[n+i, i] averaged over all
positions i and all sequences. This definition must not be altered.
"""
from __future__ import annotations
import logging
from typing import Optional

import torch
import transformer_lens
from jaxtyping import Float
from torch import Tensor

logger = logging.getLogger(__name__)


def compute_induction_score(
    model: transformer_lens.HookedTransformer,
    sequence_length: int = 50,
    num_sequences: int = 100,
    seed: int = 42,
    device: Optional[str] = None,
) -> Float[Tensor, "layer head"]:
    """Compute per-head induction scores using repeated-token sequences.

    Args:
        model: A loaded HookedTransformer instance.
        sequence_length: Half-length of each test sequence (full = 2*sequence_length).
        num_sequences: Number of random sequences to average over.
        seed: Random seed for reproducibility.
        device: Torch device. Inferred from model if None.

    Returns:
        Float tensor [n_layers, n_heads].

    Raises:
        ValueError: If sequence_length < 2.
    """
    if sequence_length < 2:
        raise ValueError(f"sequence_length must be >= 2, got {sequence_length}.")

    if device is None:
        device = next(model.parameters()).device.type

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    vocab_size = model.cfg.d_vocab

    torch.manual_seed(seed)
    prefix = torch.randint(0, vocab_size, (num_sequences, sequence_length), device=device)
    sequences = torch.cat([prefix, prefix], dim=1)  # [num_seq, 2*seq_len]

    with torch.no_grad():
        _, cache = model.run_with_cache(
            sequences,
            names_filter=lambda name: "hook_attn" in name or "hook_pattern" in name,
            return_type=None,
        )

    scores = torch.zeros(n_layers, n_heads, device=device)
    query_positions = torch.arange(sequence_length, 2 * sequence_length, device=device)
    key_positions = torch.arange(0, sequence_length, device=device)

    for layer in range(n_layers):
        attn_key = f"blocks.{layer}.attn.hook_attn"
        if attn_key not in cache:
            attn_key = f"blocks.{layer}.attn.hook_pattern"
        attn = cache[attn_key]  # [batch, heads, query, key]
        for head in range(n_heads):
            head_attn = attn[:, head, :, :]  # [batch, query, key]
            induction_attn = head_attn[:, query_positions, key_positions]  # [batch, seq_len]
            scores[layer, head] = induction_attn.mean()

    logger.debug("Induction scores: mean=%.4f max=%.4f", float(scores.mean()), float(scores.max()))
    return scores


def compute_induction_score_with_stats(
    model: transformer_lens.HookedTransformer,
    sequence_length: int = 50,
    num_sequences: int = 100,
    seed: int = 42,
    device: Optional[str] = None,
) -> tuple[Float[Tensor, "layer head"], Float[Tensor, "layer head"]]:
    """Compute induction scores with per-sequence standard deviations.

    Returns:
        Tuple (mean_scores, std_scores), each [n_layers, n_heads].
    """
    if sequence_length < 2:
        raise ValueError(f"sequence_length must be >= 2, got {sequence_length}.")

    if device is None:
        device = next(model.parameters()).device.type

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    vocab_size = model.cfg.d_vocab

    torch.manual_seed(seed)
    prefix = torch.randint(0, vocab_size, (num_sequences, sequence_length), device=device)
    sequences = torch.cat([prefix, prefix], dim=1)

    with torch.no_grad():
        _, cache = model.run_with_cache(
            sequences,
            names_filter=lambda name: "hook_attn" in name or "hook_pattern" in name,
            return_type=None,
        )

    means = torch.zeros(n_layers, n_heads, device=device)
    stds = torch.zeros(n_layers, n_heads, device=device)
    query_positions = torch.arange(sequence_length, 2 * sequence_length, device=device)
    key_positions = torch.arange(0, sequence_length, device=device)

    for layer in range(n_layers):
        attn_key = f"blocks.{layer}.attn.hook_attn"
        if attn_key not in cache:
            attn_key = f"blocks.{layer}.attn.hook_pattern"
        attn = cache[attn_key]
        for head in range(n_heads):
            head_attn = attn[:, head, :, :]
            induction_attn = head_attn[:, query_positions, key_positions]
            per_seq = induction_attn.mean(dim=1)
            means[layer, head] = per_seq.mean()
            stds[layer, head] = per_seq.std()

    return means, stds


def identify_induction_heads(
    scores: Float[Tensor, "layer head"],
    threshold: float = 0.4,
) -> list[tuple[int, int]]:
    """Return (layer, head) pairs with induction score above threshold."""
    indices = (scores > threshold).nonzero(as_tuple=False)
    result = [(int(idx[0]), int(idx[1])) for idx in indices]
    result.sort()
    logger.info("Induction heads (threshold=%.2f): %s", threshold, result)
    return result
