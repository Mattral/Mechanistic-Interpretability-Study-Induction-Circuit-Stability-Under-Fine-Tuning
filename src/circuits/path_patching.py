"""Path patching for indirect effect decomposition.

Measures causal effect of the path src_head -> dst_head via residual stream.
Reference: Wang et al. (2022), "Interpretability in the Wild."
"""
from __future__ import annotations
import logging
from typing import Optional

import torch
import transformer_lens
from jaxtyping import Float
from torch import Tensor
from src.circuits.patching import _build_clean_and_corrupted, _compute_logit_diff

logger = logging.getLogger(__name__)


def compute_path_patching_scores(
    model: transformer_lens.HookedTransformer,
    seq_len: int = 20,
    batch_size: int = 32,
    seed: int = 42,
    device: Optional[str] = None,
) -> Float[Tensor, "src_layer src_head dst_layer dst_head"]:
    """Compute path patching scores for all head-to-head paths.

    For each (src, dst) pair, measures recovery of induction behaviour
    mediated by the path src -> dst through the residual stream.
    Only entries where src_layer < dst_layer are causally meaningful.

    Returns:
        Float tensor [n_layers, n_heads, n_layers, n_heads].
    """
    if device is None:
        device = next(model.parameters()).device.type

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    clean_tokens, corrupted_tokens = _build_clean_and_corrupted(
        seq_len=seq_len, batch_size=batch_size,
        vocab_size=model.cfg.d_vocab, seed=seed, device=device,
    )

    with torch.no_grad():
        _, clean_cache = model.run_with_cache(
            clean_tokens,
            names_filter=lambda n: n.endswith("hook_z"),
            return_type=None,
        )

    ld_clean = _compute_logit_diff(model, clean_tokens, seq_len)
    ld_corrupted = _compute_logit_diff(model, corrupted_tokens, seq_len)
    denom = ld_clean - ld_corrupted

    if abs(float(denom)) < 1e-8:
        logger.warning("Near-zero denominator in path patching. Returning zeros.")
        return torch.zeros(n_layers, n_heads, n_layers, n_heads, device=device)

    path_scores = torch.zeros(n_layers, n_heads, n_layers, n_heads, device=device)

    for src_layer in range(n_layers):
        for src_head in range(n_heads):
            src_hook = f"blocks.{src_layer}.attn.hook_z"

            def _src_patch(value, hook, _sh=src_head, _shn=src_hook):
                value[:, :, _sh, :] = clean_cache[_shn][:, :, _sh, :]
                return value

            with torch.no_grad():
                _, src_patched_cache = model.run_with_cache(
                    corrupted_tokens,
                    fwd_hooks=[(src_hook, _src_patch)],
                    names_filter=lambda n: n.endswith("hook_z"),
                    return_type=None,
                )

            for dst_layer in range(src_layer + 1, n_layers):
                for dst_head in range(n_heads):
                    dst_hook = f"blocks.{dst_layer}.attn.hook_z"
                    dst_val = src_patched_cache[dst_hook]

                    def _dst_patch(value, hook, _dh=dst_head, _dv=dst_val):
                        value[:, :, _dh, :] = _dv[:, :, _dh, :]
                        return value

                    with torch.no_grad():
                        pl = model(corrupted_tokens, fwd_hooks=[(dst_hook, _dst_patch)])

                    qp = torch.arange(seq_len, 2 * seq_len - 1, device=device)
                    tgt = clean_tokens[:, 1:seq_len]
                    ql = pl[:, qp, :]
                    cl = ql.gather(dim=2, index=tgt.unsqueeze(2)).squeeze(2)
                    ld_path = (cl - ql.mean(dim=2)).mean()
                    path_scores[src_layer, src_head, dst_layer, dst_head] = (ld_path - ld_corrupted) / denom

    logger.info("Path patching scores computed.")
    return path_scores


def get_significant_paths(
    path_scores: Float[Tensor, "src_layer src_head dst_layer dst_head"],
    threshold: float = 0.3,
) -> list[tuple[int, int, int, int]]:
    """Return (src_layer, src_head, dst_layer, dst_head) tuples above threshold."""
    indices = (path_scores > threshold).nonzero(as_tuple=False)
    result = [tuple(int(idx[i]) for i in range(4)) for idx in indices]
    result.sort()
    return result
