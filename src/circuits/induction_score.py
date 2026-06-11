"""Induction score metric — canonical Olsson et al. (2022) definition.

For a repeated-token sequence [t_1,...,t_n, t_1,...,t_n], the induction score
for head (l,h) is the mean attention weight A^(l,h)[n+i, i] averaged over all
positions i in [0,n-1] and all sequences. This definition must not be altered.

CRITICAL IMPLEMENTATION NOTE — BOS token:
TransformerLens prepends a BOS token by default (prepend_bos=True), which
shifts all position indices by +1 and corrupts the IS formula.
We pass prepend_bos=False to run_with_cache so the sequence the model
attends over is exactly [t_1,...,t_n, t_1,...,t_n] with no offset.
This matches the Neel Nanda TransformerLens tutorial implementation.
See decisions/DECISION_LOG.md DECISION-005.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import transformer_lens
from torch import Tensor

logger = logging.getLogger(__name__)


def compute_induction_score(
    model: transformer_lens.HookedTransformer,
    sequence_length: int = 50,
    num_sequences: int = 100,
    seed: int = 42,
    device: Optional[str] = None,
) -> Tensor:
    """Compute per-head induction scores using repeated-token sequences.

    Uses prepend_bos=False to avoid BOS-token position offset that would
    corrupt the A[n+i, i] index formula. See module docstring.

    Args:
        model: A loaded HookedTransformer instance.
        sequence_length: Half-length n of each test sequence.
            Full sequence length is 2*sequence_length (no BOS).
        num_sequences: Number of random sequences to average over.
        seed: Random seed for reproducibility.
        device: Torch device. Inferred from model if None.

    Returns:
        Float tensor of shape [n_layers, n_heads]. Entry [l, h] is the
        induction score for layer l, head h.

    Raises:
        ValueError: If sequence_length < 2.
    """
    if sequence_length < 2:
        raise ValueError(f"sequence_length must be >= 2, got {sequence_length}.")

    if device is None:
        device = next(model.parameters()).device.type

    n_layers: int = model.cfg.n_layers
    n_heads: int = model.cfg.n_heads
    vocab_size: int = model.cfg.d_vocab

    torch.manual_seed(seed)
    prefix = torch.randint(0, vocab_size, (num_sequences, sequence_length), device=device)
    sequences = torch.cat([prefix, prefix], dim=1)  # [batch, 2*seq_len], no BOS

    with torch.no_grad():
        _, cache = model.run_with_cache(
            sequences,
            names_filter=lambda name: "hook_pattern" in name,
            prepend_bos=False,   # ← CRITICAL: prevents BOS position offset
            return_type=None,
        )

    scores = torch.zeros(n_layers, n_heads, device=device)

    # Validate cache shape — must be [batch, n_heads, 2n, 2n] (no BOS)
    key0 = "blocks.0.attn.hook_pattern"
    if key0 in cache:
        expected_seq = 2 * sequence_length
        actual_seq = cache[key0].shape[2]
        if actual_seq != expected_seq:
            raise RuntimeError(
                f"Unexpected attention pattern seq length {actual_seq} "
                f"(expected {expected_seq}). BOS token may still be prepended. "
                "Check TransformerLens version and prepend_bos behaviour."
            )

    # IS formula: score[l,h] = mean_i( A^(l,h)[n+i, i] ) for i in [0, n-1]
    # query positions: [n, n+1, ..., 2n-1]  (second copy of each token)
    # key positions:   [0,  1,  ...,  n-1]  (first copy of each token)
    n = sequence_length
    query_positions = torch.arange(n, 2 * n, device=device)
    key_positions = torch.arange(0, n, device=device)

    for layer in range(n_layers):
        attn_key = f"blocks.{layer}.attn.hook_pattern"
        if attn_key not in cache:
            raise KeyError(
                f"Expected cache key '{attn_key}' not found. "
                f"Available keys: {list(cache.keys())}"
            )
        attn = cache[attn_key]  # [batch, n_heads, 2n, 2n]
        for head in range(n_heads):
            head_attn = attn[:, head, :, :]  # [batch, 2n, 2n]
            # Advanced indexing: reads A[batch, n+i, i] for each i
            induction_weights = head_attn[:, query_positions, key_positions]  # [batch, n]
            scores[layer, head] = induction_weights.mean()

    logger.debug(
        "IS computed (n=%d, n_seq=%d): mean=%.4f, max=%.4f [layer=%d, head=%d]",
        sequence_length, num_sequences,
        float(scores.mean()), float(scores.max()),
        *divmod(int(scores.argmax()), n_heads),
    )
    return scores


def compute_induction_score_with_stats(
    model: transformer_lens.HookedTransformer,
    sequence_length: int = 50,
    num_sequences: int = 100,
    seed: int = 42,
    device: Optional[str] = None,
) -> tuple[Tensor, Tensor]:
    """Compute induction scores with per-head standard deviations.

    Same BOS fix as compute_induction_score. Std is computed across
    sequences (after averaging over positions within each sequence),
    providing error bars for paper figures.

    Args:
        model: A loaded HookedTransformer instance.
        sequence_length: Half-length of each test sequence.
        num_sequences: Number of random sequences.
        seed: Random seed.
        device: Torch device.

    Returns:
        Tuple (mean_scores, std_scores), each [n_layers, n_heads].

    Raises:
        ValueError: If sequence_length < 2.
    """
    if sequence_length < 2:
        raise ValueError(f"sequence_length must be >= 2, got {sequence_length}.")

    if device is None:
        device = next(model.parameters()).device.type

    n_layers: int = model.cfg.n_layers
    n_heads: int = model.cfg.n_heads
    vocab_size: int = model.cfg.d_vocab

    torch.manual_seed(seed)
    prefix = torch.randint(0, vocab_size, (num_sequences, sequence_length), device=device)
    sequences = torch.cat([prefix, prefix], dim=1)

    with torch.no_grad():
        _, cache = model.run_with_cache(
            sequences,
            names_filter=lambda name: "hook_pattern" in name,
            prepend_bos=False,   # ← CRITICAL: prevents BOS position offset
            return_type=None,
        )

    means = torch.zeros(n_layers, n_heads, device=device)
    stds = torch.zeros(n_layers, n_heads, device=device)

    n = sequence_length
    query_positions = torch.arange(n, 2 * n, device=device)
    key_positions = torch.arange(0, n, device=device)

    for layer in range(n_layers):
        attn_key = f"blocks.{layer}.attn.hook_pattern"
        attn = cache[attn_key]  # [batch, n_heads, 2n, 2n]
        for head in range(n_heads):
            head_attn = attn[:, head, :, :]
            induction_weights = head_attn[:, query_positions, key_positions]  # [batch, n]
            per_seq_score = induction_weights.mean(dim=1)  # [batch]
            means[layer, head] = per_seq_score.mean()
            stds[layer, head] = per_seq_score.std()

    logger.info(
        "IS with stats (n=%d): max=%.4f at L%dH%d",
        sequence_length, float(means.max()),
        *divmod(int(means.argmax()), n_heads),
    )
    return means, stds


def identify_induction_heads(
    scores: Tensor,
    threshold: float = 0.4,
) -> list[tuple[int, int]]:
    """Return (layer, head) pairs whose induction score exceeds the threshold.

    Args:
        scores: Float tensor [n_layers, n_heads].
        threshold: Minimum score to classify as induction head.
            Olsson et al. (2022) use ~0.4; scores > 0.7 are strongly inductive.

    Returns:
        Sorted list of (layer, head) integer tuples.
    """
    indices = (scores > threshold).nonzero(as_tuple=False)
    result = [(int(idx[0]), int(idx[1])) for idx in indices]
    result.sort()
    logger.info("Induction heads (threshold=%.2f): %s", threshold, result)
    return result
