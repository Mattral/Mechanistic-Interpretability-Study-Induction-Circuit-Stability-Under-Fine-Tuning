"""Phase transition detection in induction score curves."""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


def smooth_curve(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Savitzky-Golay smoothing. Returns same-length array."""
    if window % 2 == 0:
        window += 1
    if window > len(values):
        return values.copy()
    return savgol_filter(values, window_length=window, polyorder=2)


def detect_phase_transitions(
    steps: np.ndarray,
    induction_scores: np.ndarray,
    threshold: float = 0.1,
    window: int = 5,
) -> list[dict]:
    """Detect phase transitions (|delta| >= threshold) for all heads.

    Args:
        steps: 1D array of training steps [n_ckpts].
        induction_scores: Array [n_ckpts, n_layers, n_heads].
        threshold: Minimum absolute score change to flag (0.1 = practically meaningful).
        window: Savitzky-Golay smoothing window size.

    Returns:
        List of transition dicts sorted by step, each with keys:
        layer, head, step, step_fraction, score_before, score_after, delta.
    """
    n_ckpts, n_layers, n_heads = induction_scores.shape
    max_step = int(steps[-1]) if len(steps) > 0 else 1
    transitions = []

    for layer in range(n_layers):
        for head in range(n_heads):
            curve = induction_scores[:, layer, head]
            smoothed = smooth_curve(curve, window=window)
            diffs = np.diff(smoothed)
            for i, diff in enumerate(diffs):
                if abs(diff) >= threshold:
                    step = int(steps[i + 1])
                    transitions.append({
                        "layer": layer, "head": head,
                        "step": step, "step_fraction": step / max_step,
                        "score_before": float(smoothed[i]),
                        "score_after": float(smoothed[i + 1]),
                        "delta": float(diff),
                    })

    transitions.sort(key=lambda x: x["step"])
    logger.info("Phase transitions detected: %d (threshold=%.2f)", len(transitions), threshold)
    return transitions


def summarise_transitions(transitions: list[dict]) -> dict:
    """Produce a summary of detected phase transitions."""
    if not transitions:
        return {"n_degradations": 0, "n_improvements": 0,
                "earliest_step": None, "earliest_step_fraction": None,
                "heads_affected": [], "mean_delta": 0.0}

    degradations = [t for t in transitions if t["delta"] < 0]
    improvements = [t for t in transitions if t["delta"] > 0]
    heads_affected = sorted(set((t["layer"], t["head"]) for t in transitions))
    earliest = transitions[0]
    return {
        "n_degradations": len(degradations),
        "n_improvements": len(improvements),
        "earliest_step": earliest["step"],
        "earliest_step_fraction": earliest["step_fraction"],
        "heads_affected": heads_affected,
        "mean_delta": float(np.mean([t["delta"] for t in transitions])),
    }


def detect_circuit_dissolution_step(
    steps: np.ndarray,
    attribution_scores: np.ndarray,
    threshold: float = 0.5,
) -> Optional[int]:
    """Return first training step where any baseline circuit head drops below threshold."""
    baseline = attribution_scores[0]
    mask = baseline >= threshold
    if not np.any(mask):
        logger.warning("No circuit heads at baseline (threshold=%.2f).", threshold)
        return None
    circuit_heads = list(zip(*np.where(mask)))
    for step_idx in range(1, len(steps)):
        for layer, head in circuit_heads:
            if attribution_scores[step_idx, layer, head] < threshold:
                step = int(steps[step_idx])
                logger.info("Circuit dissolution: L%dH%d < %.2f at step %d", layer, head, threshold, step)
                return step
    logger.info("No circuit dissolution detected.")
    return None
