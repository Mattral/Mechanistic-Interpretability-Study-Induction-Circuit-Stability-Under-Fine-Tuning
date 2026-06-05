"""Induction score metric implementation.

Implements the exact canonical definition from Olsson et al. (2022):

    For a repeated-token sequence [t_1, ..., t_n, t_1, ..., t_n], the induction
    score is the mean attention weight paid by position n+i to position i,
    averaged over all i and over all sequences.

This definition must not be altered. Alternative metrics must be defined
separately and clearly labelled as non-canonical.
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

    Induction score is defined as the mean attention paid by each head
    to the token that follows the previous occurrence of the current token.
    See Olsson et al. (2022), Section 3.

    The sequence construction is:
        [t_1, ..., t_n, t_1, ..., t_n]
    where each t_i is drawn uniformly at random from the vocabulary.
    For position n+i (0-indexed: n+i where i in [0, n-1]), the target
    attention position is i (the previous occurrence of t_i), so the
    induction score measures attention weight at offset -(n) relative to
    the current position — specifically, the weight at position i when
    the query is at position n+i.

    Args:
        model: A loaded HookedTransformer instance.
        sequence_length: Half-length of each test sequence. The full
            sequence will be 2 * sequence_length tokens. Must be even.
        num_sequences: Number of random sequences to average over.
        seed: Random seed for reproducibility.
        device: Torch device string. Inferred from model if None.

    Returns:
        Float tensor of shape [n_layers, n_heads] where entry [l, h] is
        the induction score for layer l, head h.

    Raises:
        ValueError: If sequence_length is less than 2.
    """
    if sequence_length < 2:
        raise ValueError(
            f"sequence_length must be at least 2, got {sequence_length}."
        )

    if device is None:
        device = next(model.parameters()).device.type

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    vocab_size = model.cfg.d_vocab

    torch.manual_seed(seed)

    # Shape: [num_sequences, 2 * sequence_length]
    prefix = torch.randint(0, vocab_size, (num_sequences, sequence_length), device=device)
    sequences = torch.cat([prefix, prefix], dim=1)  # repeated-token sequences

    # Run model and cache all attention patterns
    with torch.no_grad():
        _, cache = model.run_with_cache(
            sequences,
            names_filter=lambda name: name.endswith("attn.hook_attn"),
            return_type=None,
        )

    scores = torch.zeros(n_layers, n_heads, device=device)

    for layer in range(n_layers):
        # attn shape: [batch, n_heads, seq_len, seq_len]
        attn_key = f"blocks.{layer}.attn.hook_attn"
        if attn_key not in cache:
            # Fallback key name used by some TransformerLens versions
            attn_key = f"blocks.{layer}.attn.hook_pattern"

        attn: torch.Tensor = cache[attn_key]  # [batch, heads, query_pos, key_pos]

        for head in range(n_heads):
            head_attn = attn[:, head, :, :]  # [batch, query_pos, key_pos]

            # For each position n+i (i in [0, seq_len-1]), extract attention
            # weight at position i. This is the induction score for that position.
            # query positions: sequence_length, sequence_length+1, ..., 2*sequence_length-1
            # key positions:   0,              1,                  ..., sequence_length-1

            query_positions = torch.arange(sequence_length, 2 * sequence_length, device=device)
            key_positions = torch.arange(0, sequence_length, device=device)

            # head_attn[:, query_positions, key_positions] → [batch, seq_len]
            induction_attn = head_attn[:, query_positions, key_positions]
            scores[layer, head] = induction_attn.mean()

    logger.debug(
        "Induction scores computed: mean=%.4f, max=%.4f",
        float(scores.mean()),
        float(scores.max()),
    )
    return scores


def compute_induction_score_with_stats(
    model: transformer_lens.HookedTransformer,
    sequence_length: int = 50,
    num_sequences: int = 100,
    seed: int = 42,
    device: Optional[str] = None,
) -> tuple[
    Float[Tensor, "layer head"],
    Float[Tensor, "layer head"],
]:
    """Compute induction scores along with standard deviation across sequences.

    Identical to compute_induction_score but also returns per-head standard
    deviations, which are required for paper figures with error bars.

    Args:
        model: A loaded HookedTransformer instance.
        sequence_length: Half-length of each test sequence.
        num_sequences: Number of random sequences to average over.
        seed: Random seed for reproducibility.
        device: Torch device string. Inferred from model if None.

    Returns:
        Tuple of (mean_scores, std_scores), each of shape [n_layers, n_heads].

    Raises:
        ValueError: If sequence_length is less than 2.
    """
    if sequence_length < 2:
        raise ValueError(
            f"sequence_length must be at least 2, got {sequence_length}."
        )

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
            names_filter=lambda name: name.endswith("attn.hook_attn"),
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
            # Per-sequence mean (averaged over positions), then across sequences
            per_seq_score = induction_attn.mean(dim=1)  # [batch]
            means[layer, head] = per_seq_score.mean()
            stds[layer, head] = per_seq_score.std()

    return means, stds


def identify_induction_heads(
    scores: Float[Tensor, "layer head"],
    threshold: float = 0.4,
) -> list[tuple[int, int]]:
    """Return (layer, head) pairs whose induction score exceeds the threshold.

    Per Olsson et al. (2022), induction heads in the 2-layer attention-only
    model typically have scores above 0.4–0.7. Scores below ~0.1 are
    considered non-induction heads.

    Args:
        scores: Induction score tensor of shape [n_layers, n_heads].
        threshold: Minimum score to be classified as an induction head.

    Returns:
        Sorted list of (layer, head) integer tuples.
    """
    indices = (scores > threshold).nonzero(as_tuple=False)
    result = [(int(idx[0]), int(idx[1])) for idx in indices]
    result.sort()
    logger.info(
        "Identified %d induction heads (threshold=%.2f): %s",
        len(result),
        threshold,
        result,
    )
    return result
