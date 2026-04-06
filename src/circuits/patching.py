"""Activation patching for causal circuit verification.

Protocol (AGENT_INSTRUCTIONS Section 4.3):
1. Clean run on repeated-token sequences.
2. Corrupted run: second half of each sequence randomly shuffled.
3. Patch one head at a time from clean into corrupted run.
4. Attribution = (recovered - corrupted) / (clean - corrupted).
Heads with attribution >= 0.5 are in the circuit.
"""
from __future__ import annotations
import logging
from typing import Optional

import torch
import transformer_lens
from jaxtyping import Float
from torch import Tensor

logger = logging.getLogger(__name__)
CIRCUIT_THRESHOLD = 0.5


def _build_clean_and_corrupted(
    seq_len: int, batch_size: int, vocab_size: int, seed: int, device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build matched clean (repeated) and corrupted (shuffled second half) sequences."""
    torch.manual_seed(seed)
    prefix = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    clean = torch.cat([prefix, prefix], dim=1)
    shuffled = torch.stack(
        [prefix[b][torch.randperm(seq_len, device=device)] for b in range(batch_size)]
    )
    corrupted = torch.cat([prefix, shuffled], dim=1)
    return clean, corrupted


def _compute_logit_diff(
    model: transformer_lens.HookedTransformer,
    tokens: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Mean logit difference at induction positions (correct - mean_other)."""
    with torch.no_grad():
        logits = model(tokens)  # [batch, 2*seq_len, vocab]
    query_positions = torch.arange(seq_len, 2 * seq_len - 1, device=tokens.device)
    target_tokens = tokens[:, 1:seq_len]  # [batch, seq_len-1]
    query_logits = logits[:, query_positions, :]  # [batch, seq_len-1, vocab]
    correct_logits = query_logits.gather(dim=2, index=target_tokens.unsqueeze(2)).squeeze(2)
    mean_logits = query_logits.mean(dim=2)
    return (correct_logits - mean_logits).mean()


def patch_head_activation(
    model: transformer_lens.HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache: dict,
    layer: int,
    head: int,
) -> torch.Tensor:
    """Patch one head output from clean into a corrupted forward pass."""
    hook_name = f"blocks.{layer}.attn.hook_z"

    def patch_hook(value, hook, _sh=head, _hn=hook_name):
        value[:, :, _sh, :] = clean_cache[_hn][:, :, _sh, :]
        return value

    with model.hooks(fwd_hooks=[(hook_name, patch_hook)]):
        patched_logits = model(corrupted_tokens)
    return patched_logits


def compute_circuit_attribution(
    model: transformer_lens.HookedTransformer,
    seq_len: int = 20,
    batch_size: int = 32,
    seed: int = 42,
    device: Optional[str] = None,
) -> Float[Tensor, "layer head"]:
    """Compute per-head attribution scores via activation patching.

    Args:
        model: A loaded HookedTransformer.
        seq_len: Half-length of test sequences.
        batch_size: Number of sequence pairs.
        seed: Random seed.
        device: Torch device. Inferred if None.

    Returns:
        Float tensor [n_layers, n_heads]. Score >= CIRCUIT_THRESHOLD -> in circuit.
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
            names_filter=lambda name: name.endswith("hook_z"),
            return_type=None,
        )

    logit_diff_clean = _compute_logit_diff(model, clean_tokens, seq_len)
    logit_diff_corrupted = _compute_logit_diff(model, corrupted_tokens, seq_len)
    denominator = logit_diff_clean - logit_diff_corrupted

    if abs(float(denominator)) < 1e-8:
        logger.warning("Near-zero denominator in attribution. Returning zeros.")
        return torch.zeros(n_layers, n_heads, device=device)

    attribution_scores = torch.zeros(n_layers, n_heads, device=device)

    for layer in range(n_layers):
        for head in range(n_heads):
            with torch.no_grad():
                patched_logits = patch_head_activation(
                    model=model, corrupted_tokens=corrupted_tokens,
                    clean_cache=clean_cache, layer=layer, head=head,
                )
            q_pos = torch.arange(seq_len, 2 * seq_len - 1, device=device)
            tgt = clean_tokens[:, 1:seq_len]
            q_logits = patched_logits[:, q_pos, :]
            correct = q_logits.gather(dim=2, index=tgt.unsqueeze(2)).squeeze(2)
            ld_patched = (correct - q_logits.mean(dim=2)).mean()
            attribution_scores[layer, head] = (ld_patched - logit_diff_corrupted) / denominator

    circuit = [(int(l), int(h)) for l, h in (attribution_scores >= CIRCUIT_THRESHOLD).nonzero(as_tuple=False)]
    logger.info("Circuit heads (>=%.1f): %s", CIRCUIT_THRESHOLD, sorted(circuit))
    return attribution_scores


def get_circuit_heads(
    attribution_scores: Float[Tensor, "layer head"],
    threshold: float = CIRCUIT_THRESHOLD,
) -> list[tuple[int, int]]:
    """Extract circuit heads above threshold."""
    indices = (attribution_scores >= threshold).nonzero(as_tuple=False)
    result = [(int(idx[0]), int(idx[1])) for idx in indices]
    result.sort()
    return result
