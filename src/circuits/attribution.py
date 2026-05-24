"""Direct Logit Attribution (DLA) helpers.

Decomposes each head's contribution to the final logit via the residual stream.
Reference: Elhage et al. (2021), "A Mathematical Framework for Transformer Circuits."
"""
from __future__ import annotations
import logging
from typing import Optional

import torch
import transformer_lens
from torch import Tensor

logger = logging.getLogger(__name__)


def compute_direct_logit_attribution(
    model: transformer_lens.HookedTransformer,
    tokens: torch.Tensor,
    target_positions: Optional[torch.Tensor] = None,
    target_tokens: Optional[torch.Tensor] = None,
    device: Optional[str] = None,
) -> Tensor:
    """Compute direct logit attribution for each attention head.

    DLA[l,h] = mean over target positions of (head output projected through W_O W_U)
    dotted with the one-hot correct-token direction.

    Args:
        model: HookedTransformer (fold_ln=True required for validity).
        tokens: Input token tensor [batch, seq_len].
        target_positions: Positions to measure [n_pos]. Default: all positions.
        target_tokens: Correct next tokens [batch, n_pos]. Default: actual next tokens.
        device: Torch device.

    Returns:
        Float tensor [n_layers, n_heads].
    """
    if tokens.shape[1] < 2:
        raise ValueError(f"tokens must have >= 2 positions, got {tokens.shape}.")

    if device is None:
        device = next(model.parameters()).device.type
    tokens = tokens.to(device)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    if target_positions is None:
        target_positions = torch.arange(tokens.shape[1] - 1, device=device)
    if target_tokens is None:
        target_tokens = tokens[:, 1:][:, target_positions]

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: n.endswith("hook_z"),
        prepend_bos=False,
        return_type=None,
        )

    W_U = model.W_U  # [d_model, vocab]
    dla = torch.zeros(n_layers, n_heads, device=device)

    for layer in range(n_layers):
        z = cache[f"blocks.{layer}.attn.hook_z"]  # [batch, seq, heads, d_head]
        W_O = model.blocks[layer].attn.W_O  # [heads, d_head, d_model]
        for head in range(n_heads):
            head_resid = z[:, :, head, :] @ W_O[head]  # [batch, seq, d_model]
            head_logits = head_resid[:, target_positions, :] @ W_U  # [batch, n_pos, vocab]
            correct = head_logits.gather(dim=2, index=target_tokens.unsqueeze(2)).squeeze(2)
            dla[layer, head] = correct.mean()

    return dla
