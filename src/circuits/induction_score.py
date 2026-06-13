"""Induction score metric — canonical Olsson et al. (2022) definition.

For a repeated-token sequence [t_1,...,t_n, t_1,...,t_n] (length n each
half, 2n total), the induction score (prefix-matching score) for head
(l,h) is the mean attention weight A^(l,h)[n+j, j+1] averaged over all
j in [0, n-2] and all sequences — i.e. the average attention from the
second occurrence of a token back to the position immediately AFTER its
first occurrence (the token it is "copying"), normalised by (n-1).

This is the formula given in Olsson et al. (2022): "its average attention
from the source token x_i to the next token of its previous occurrence",
with normalisation 1/(|x|-1). See DECISION-005 (REVISED) in
decisions/DECISION_LOG.md for the full derivation and the off-by-one bug
this fixes (the previous implementation read A[n+i, i], the attention to
the SAME token as the first occurrence, rather than A[n+j, j+1], the
attention to the token that FOLLOWS the first occurrence — which is what
an induction head actually needs to copy).

Implementation note — BOS token:
Test sequences are constructed as raw integer tensors (torch.randint), not
strings, so TransformerLens's tokenizer/BOS-prepending logic is never
invoked for this code path; prepend_bos has no effect here. We still pass
prepend_bos=False defensively and validate the cache shape is exactly
[batch, n_heads, 2n, 2n], in case future callers pass string input.
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
    """Compute per-head induction (prefix-matching) scores.

    Uses repeated-token sequences [t_1,...,t_n, t_1,...,t_n] and reads
    A[n+j, j+1] for j in [0, n-2] — the attention from the second
    occurrence of a token back to the token that followed its first
    occurrence. See module docstring for the full formula derivation.

    Args:
        model: A loaded HookedTransformer instance.
        sequence_length: Half-length n of each test sequence.
            Full sequence length is 2*sequence_length (no BOS).
            Must be >= 2 (n-1 >= 1 valid (query, key) pairs).
        num_sequences: Number of random sequences to average over.
        seed: Random seed for reproducibility.
        device: Torch device. Inferred from model if None.

    Returns:
        Float tensor of shape [n_layers, n_heads]. Entry [l, h] is the
        induction (prefix-matching) score for layer l, head h, in [0, 1].

    Raises:
        ValueError: If sequence_length < 2.
        RuntimeError: If the cached attention pattern does not have the
            expected shape [batch, n_heads, 2n, 2n].
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
            prepend_bos=False,
            return_type=None,
        )

    scores = torch.zeros(n_layers, n_heads, device=device)

    # Validate cache shape — must be [batch, n_heads, 2n, 2n]
    key0 = "blocks.0.attn.hook_pattern"
    if key0 in cache:
        expected_seq = 2 * sequence_length
        actual_seq = cache[key0].shape[2]
        if actual_seq != expected_seq:
            raise RuntimeError(
                f"Unexpected attention pattern seq length {actual_seq} "
                f"(expected {expected_seq}). Check TransformerLens version "
                "and prepend_bos behaviour for the input type used."
            )

    # IS formula (Olsson et al. 2022, prefix-matching score):
    #   score[l,h] = (1/(n-1)) * sum_{j=0}^{n-2} A^(l,h)[n+j, j+1]
    #
    # query positions: [n, n+1, ..., 2n-2]   -- second occurrence of t_0..t_{n-2}
    # key positions:   [1, 2, ...,  n-1]     -- position of t_1..t_{n-1} in the
    #                                            first copy (the token that
    #                                            FOLLOWED the first occurrence
    #                                            of the query's token)
    n = sequence_length
    query_positions = torch.arange(n, 2 * n - 1, device=device)  # n-1 positions
    key_positions = torch.arange(1, n, device=device)            # n-1 positions

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
            # Advanced indexing: reads A[batch, n+j, j+1] for each j
            induction_weights = head_attn[:, query_positions, key_positions]  # [batch, n-1]
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

    Same formula as compute_induction_score (A[n+j, j+1], j in [0, n-2]).
    Std is computed across sequences (after averaging over positions
    within each sequence), providing error bars for paper figures.

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
            prepend_bos=False,
            return_type=None,
        )

    means = torch.zeros(n_layers, n_heads, device=device)
    stds = torch.zeros(n_layers, n_heads, device=device)

    n = sequence_length
    query_positions = torch.arange(n, 2 * n - 1, device=device)  # n-1 positions
    key_positions = torch.arange(1, n, device=device)            # n-1 positions

    for layer in range(n_layers):
        attn_key = f"blocks.{layer}.attn.hook_pattern"
        attn = cache[attn_key]  # [batch, n_heads, 2n, 2n]
        for head in range(n_heads):
            head_attn = attn[:, head, :, :]
            induction_weights = head_attn[:, query_positions, key_positions]  # [batch, n-1]
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
