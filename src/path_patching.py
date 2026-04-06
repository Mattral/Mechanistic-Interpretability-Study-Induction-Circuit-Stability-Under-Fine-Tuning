"""Path patching utilities for indirect effect decomposition.

Path patching extends activation patching by isolating the causal effect
of a specific *path* through the computation graph, rather than the total
effect of a node. This allows decomposing whether a head's contribution
flows through direct logit attribution or through downstream heads.

Reference: Wang et al. (2022), "Interpretability in the Wild."
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import transformer_lens
from jaxtyping import Float
from torch import Tensor

from src.circuits.patching import (
    CIRCUIT_THRESHOLD,
    _build_clean_and_corrupted,
    _compute_logit_diff,
)

logger = logging.getLogger(__name__)


def compute_path_patching_scores(
    model: transformer_lens.HookedTransformer,
    seq_len: int = 20,
    batch_size: int = 32,
    seed: int = 42,
    device: Optional[str] = None,
) -> Float[Tensor, "src_layer src_head dst_layer dst_head"]:
    """Compute path patching scores for all head-to-head paths.

    For each (src, dst) head pair, measures how much of the clean→corrupted
    recovery is mediated by the path src_head → dst_head (via residual stream).

    The score is computed by:
    1. Running the clean run and caching src_head outputs.
    2. In the corrupted run, patching only src_head.
    3. Measuring the change in dst_head's output logit contribution.

    This is an approximation via independent patching rather than full
    path patching (which requires differentiable forward passes). The
    approximation is acceptable for circuit verification purposes.

    Args:
        model: A loaded HookedTransformer.
        seq_len: Half-length of repeated-token test sequences.
        batch_size: Number of sequence pairs.
        seed: Random seed.
        device: Torch device string. Inferred from model if None.

    Returns:
        Float tensor [n_layers, n_heads, n_layers, n_heads]. Entry [l1, h1, l2, h2]
        is the path patching score for the path from head (l1, h1) to head (l2, h2).
        Only paths where l1 < l2 are meaningful (causal order).
    """
    if device is None:
        device = next(model.parameters()).device.type

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    vocab_size = model.cfg.d_vocab

    clean_tokens, corrupted_tokens = _build_clean_and_corrupted(
        seq_len=seq_len,
        batch_size=batch_size,
        vocab_size=vocab_size,
        seed=seed,
        device=device,
    )

    # Clean run: cache all hook_z activations
    with torch.no_grad():
        _, clean_cache = model.run_with_cache(
            clean_tokens,
            names_filter=lambda name: name.endswith("hook_z"),
            return_type=None,
        )

    logit_diff_clean = _compute_logit_diff(model, clean_tokens, seq_len)
    logit_diff_corrupted = _compute_logit_diff(model, corrupted_tokens, seq_len)
    denominator = logit_diff_clean - logit_diff_corrupted

    if abs(float(denominator)) < 1e-8:
        logger.warning("Near-zero denominator in path patching. Returning zeros.")
        return torch.zeros(n_layers, n_heads, n_layers, n_heads, device=device)

    path_scores = torch.zeros(n_layers, n_heads, n_layers, n_heads, device=device)

    for src_layer in range(n_layers):
        for src_head in range(n_heads):
            # Patch src_head from clean into corrupted run
            src_hook_name = f"blocks.{src_layer}.attn.hook_z"

            def _src_patch(
                value: torch.Tensor,
                hook: transformer_lens.hook_points.HookPoint,
                _sh: int = src_head,
                _shn: str = src_hook_name,
            ) -> torch.Tensor:
                value[:, :, _sh, :] = clean_cache[_shn][:, :, _sh, :]
                return value

            # Run with src patched; cache all dst hook_z values
            with torch.no_grad():
                _, src_patched_cache = model.run_with_cache(
                    corrupted_tokens,
                    fwd_hooks=[(src_hook_name, _src_patch)],
                    names_filter=lambda name: name.endswith("hook_z"),
                    return_type=None,
                )

            # Measure how each downstream head's output changed
            for dst_layer in range(src_layer + 1, n_layers):
                for dst_head in range(n_heads):
                    dst_hook_name = f"blocks.{dst_layer}.attn.hook_z"

                    # Patch dst_head with the src-patched value (to isolate path)
                    dst_patched_val = src_patched_cache[dst_hook_name]

                    def _dst_patch(
                        value: torch.Tensor,
                        hook: transformer_lens.hook_points.HookPoint,
                        _dh: int = dst_head,
                        _dpv: torch.Tensor = dst_patched_val,
                    ) -> torch.Tensor:
                        value[:, :, _dh, :] = _dpv[:, :, _dh, :]
                        return value

                    with torch.no_grad():
                        patched_logits = model(
                            corrupted_tokens,
                            fwd_hooks=[(dst_hook_name, _dst_patch)],
                        )

                    query_positions = torch.arange(
                        seq_len, 2 * seq_len - 1, device=device
                    )
                    target_tokens = clean_tokens[:, 1:seq_len]
                    query_logits = patched_logits[:, query_positions, :]
                    correct_logits = query_logits.gather(
                        dim=2, index=target_tokens.unsqueeze(2)
                    ).squeeze(2)
                    mean_logits = query_logits.mean(dim=2)
                    logit_diff_path = (correct_logits - mean_logits).mean()

                    path_scores[src_layer, src_head, dst_layer, dst_head] = (
                        (logit_diff_path - logit_diff_corrupted) / denominator
                    )

    logger.info("Path patching scores computed.")
    return path_scores


def get_significant_paths(
    path_scores: Float[Tensor, "src_layer src_head dst_layer dst_head"],
    threshold: float = 0.3,
) -> list[tuple[int, int, int, int]]:
    """Return (src_layer, src_head, dst_layer, dst_head) tuples above threshold.

    Args:
        path_scores: Output of compute_path_patching_scores.
        threshold: Minimum path score to consider significant.

    Returns:
        Sorted list of 4-tuples (src_layer, src_head, dst_layer, dst_head).
    """
    indices = (path_scores > threshold).nonzero(as_tuple=False)
    result = [tuple(int(idx[i]) for i in range(4)) for idx in indices]
    result.sort()
    logger.info("Significant paths (threshold=%.2f): %s", threshold, result)
    return result  # type: ignore[return-value]
