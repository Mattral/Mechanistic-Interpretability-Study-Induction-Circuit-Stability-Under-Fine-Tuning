"""Activation patching utilities for causal circuit verification.

Implements the patching protocol defined in AGENT_INSTRUCTIONS Section 4.3:

1. Clean run: model behaviour on the induction task.
2. Corrupted run: same model, repeated-token structure broken (second half shuffled).
3. Patch one component at a time: replace activations of one head/layer in the
   corrupted run with activations from the clean run.
4. Measure recovery: logit difference on the target token, normalised between
   the fully corrupted and fully clean values.
5. Attribution score = (recovered logit diff) / (clean - corrupted).

A head with attribution score ≥ 0.5 is considered part of the circuit.
This threshold is documented here and must be stated in the paper.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import transformer_lens
from jaxtyping import Float
from torch import Tensor

logger = logging.getLogger(__name__)

CIRCUIT_THRESHOLD = 0.5  # heads with attribution score ≥ this are in the circuit


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------


def _build_clean_and_corrupted(
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    seed: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build matched clean and corrupted repeated-token sequences.

    Clean:     [t_1, ..., t_n, t_1, ..., t_n]  — proper repetition
    Corrupted: [t_1, ..., t_n, t_σ(1), ..., t_σ(n)] — second half shuffled

    Args:
        seq_len: Length of each half (total sequence will be 2*seq_len).
        batch_size: Number of sequence pairs to construct.
        vocab_size: Model vocabulary size.
        seed: Random seed.
        device: Torch device string.

    Returns:
        Tuple of (clean_tokens, corrupted_tokens), each [batch, 2*seq_len].
    """
    torch.manual_seed(seed)
    prefix = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    clean = torch.cat([prefix, prefix], dim=1)

    # Shuffle the second half independently for each batch element
    shuffled = torch.stack(
        [prefix[b][torch.randperm(seq_len, device=device)] for b in range(batch_size)]
    )
    corrupted = torch.cat([prefix, shuffled], dim=1)

    return clean, corrupted


# ---------------------------------------------------------------------------
# Logit difference computation
# ---------------------------------------------------------------------------


def _compute_logit_diff(
    model: transformer_lens.HookedTransformer,
    tokens: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Compute the logit difference at the induction positions.

    For position n+i, the correct next token is t_{i+1} (the token that
    followed t_i in the prefix). The logit diff is
        logit(correct_token) - mean(logit(all_other_tokens))
    averaged over all induction positions and all batch elements.

    Args:
        model: The HookedTransformer.
        tokens: Input token tensor [batch, 2*seq_len].
        seq_len: Half-length of the sequence.

    Returns:
        Scalar tensor: mean logit difference across positions and batch.
    """
    with torch.no_grad():
        logits = model(tokens)  # [batch, 2*seq_len, vocab]

    # Target positions: seq_len, seq_len+1, ..., 2*seq_len-2
    # (last position has no ground truth next token in this construction)
    query_positions = torch.arange(seq_len, 2 * seq_len - 1, device=tokens.device)

    # For position n+i, the correct next token is tokens[b, i+1]
    # (the token that followed t_i in the original prefix)
    # i = 0, 1, ..., seq_len-2 → target tokens = tokens[:, 1:seq_len]
    target_tokens = tokens[:, 1:seq_len]  # [batch, seq_len-1]

    query_logits = logits[:, query_positions, :]  # [batch, seq_len-1, vocab]

    # Gather logit at the correct token
    correct_logits = query_logits.gather(
        dim=2, index=target_tokens.unsqueeze(2)
    ).squeeze(2)  # [batch, seq_len-1]

    # Mean logit across vocabulary (approximate baseline)
    mean_logits = query_logits.mean(dim=2)  # [batch, seq_len-1]

    logit_diff = (correct_logits - mean_logits).mean()
    return logit_diff


# ---------------------------------------------------------------------------
# Core patching function
# ---------------------------------------------------------------------------


def patch_head_activation(
    model: transformer_lens.HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache: dict,
    layer: int,
    head: int,
) -> torch.Tensor:
    """Patch one attention head's output activation in a corrupted forward pass.

    Replaces the output of head (layer, head) in the corrupted run with
    the corresponding activation from the clean run cache.

    Args:
        model: The HookedTransformer.
        corrupted_tokens: Token tensor for the corrupted run [batch, seq].
        clean_cache: Activation cache from the clean run (from run_with_cache).
        layer: Layer index of the head to patch.
        head: Head index to patch.

    Returns:
        Logits tensor [batch, seq, vocab] from the patched corrupted run.
    """
    hook_name = f"blocks.{layer}.attn.hook_z"

    def patch_hook(value: torch.Tensor, hook: transformer_lens.hook_points.HookPoint) -> torch.Tensor:
        # value shape: [batch, seq, n_heads, d_head]
        value[:, :, head, :] = clean_cache[hook_name][:, :, head, :]
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

    Implements the full protocol from AGENT_INSTRUCTIONS Section 4.3.

    Attribution score for head (l, h) = (recovered - corrupted) / (clean - corrupted)

    where:
    - clean = logit diff on the clean repeated-token sequence
    - corrupted = logit diff on the corrupted (shuffled second half) sequence
    - recovered = logit diff after patching head (l, h) from clean → corrupted

    Args:
        model: A loaded HookedTransformer instance.
        seq_len: Half-length of test sequences.
        batch_size: Number of sequence pairs to average over.
        seed: Random seed.
        device: Torch device. Inferred from model if None.

    Returns:
        Float tensor of shape [n_layers, n_heads]. Entry [l, h] is the
        attribution score for head (l, h). Score ≥ 0.5 → in the circuit.
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

    # Compute clean activations and logit diff
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
        logger.warning(
            "Near-zero denominator in attribution (clean=%.4f, corrupted=%.4f). "
            "Returning zeros.",
            float(logit_diff_clean),
            float(logit_diff_corrupted),
        )
        return torch.zeros(n_layers, n_heads, device=device)

    attribution_scores = torch.zeros(n_layers, n_heads, device=device)

    for layer in range(n_layers):
        for head in range(n_heads):
            with torch.no_grad():
                patched_logits = patch_head_activation(
                    model=model,
                    corrupted_tokens=corrupted_tokens,
                    clean_cache=clean_cache,
                    layer=layer,
                    head=head,
                )
            # Compute logit diff from patched logits
            seq_len_here = seq_len
            query_positions = torch.arange(
                seq_len_here, 2 * seq_len_here - 1, device=device
            )
            target_tokens = clean_tokens[:, 1:seq_len_here]
            query_logits = patched_logits[:, query_positions, :]
            correct_logits = query_logits.gather(
                dim=2, index=target_tokens.unsqueeze(2)
            ).squeeze(2)
            mean_logits = query_logits.mean(dim=2)
            logit_diff_patched = (correct_logits - mean_logits).mean()

            attribution_scores[layer, head] = (
                (logit_diff_patched - logit_diff_corrupted) / denominator
            )

    logger.info(
        "Attribution scores computed. Circuit heads (≥%.1f): %s",
        CIRCUIT_THRESHOLD,
        [(int(l), int(h)) for l, h in (attribution_scores >= CIRCUIT_THRESHOLD).nonzero(as_tuple=False)],
    )
    return attribution_scores


def get_circuit_heads(
    attribution_scores: Float[Tensor, "layer head"],
    threshold: float = CIRCUIT_THRESHOLD,
) -> list[tuple[int, int]]:
    """Extract the list of circuit heads based on attribution threshold.

    Args:
        attribution_scores: Output of compute_circuit_attribution.
        threshold: Attribution score cutoff (default: CIRCUIT_THRESHOLD = 0.5).

    Returns:
        Sorted list of (layer, head) tuples for heads in the circuit.
    """
    indices = (attribution_scores >= threshold).nonzero(as_tuple=False)
    result = [(int(idx[0]), int(idx[1])) for idx in indices]
    result.sort()
    return result
