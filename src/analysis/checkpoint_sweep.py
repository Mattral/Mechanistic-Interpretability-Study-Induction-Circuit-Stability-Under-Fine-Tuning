"""Checkpoint sweep: compute all circuit metrics across saved checkpoints."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.circuits.induction_score import compute_induction_score
from src.circuits.patching import compute_circuit_attribution
from src.model.config import EvalConfig, ModelConfig
from src.model.train import load_checkpoint, load_pretrained_model, set_global_seed

logger = logging.getLogger(__name__)


def sweep_checkpoints(
    checkpoint_dir: Path,
    model_config: ModelConfig,
    eval_config: EvalConfig,
    output_path: Path,
    device: Optional[str] = None,
) -> dict:
    """Load all step_*.pt checkpoints and compute induction + attribution metrics.

    Args:
        checkpoint_dir: Directory containing step_NNNNNN.pt files.
        model_config: ModelConfig for constructing the model.
        eval_config: EvalConfig with metric parameters.
        output_path: Path to save sweep results (.npz).
        device: Torch device. Auto-detected if None.

    Returns:
        Dict with arrays: steps, train_losses, induction_scores, attribution_scores.

    Raises:
        FileNotFoundError: If directory or checkpoint files are missing.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    ckpt_files = sorted(checkpoint_dir.glob("step_*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files in {checkpoint_dir}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_global_seed(eval_config.seed)
    model = load_pretrained_model(model_config, device=device)
    steps, train_losses, ind_list, attr_list = [], [], [], []

    for ckpt_path in ckpt_files:
        logger.info("Processing: %s", ckpt_path.name)
        payload = load_checkpoint(ckpt_path, model)
        model.eval()

        ind = compute_induction_score(
            model=model, sequence_length=eval_config.induction_seq_length,
            num_sequences=eval_config.induction_num_sequences,
            seed=eval_config.seed, device=device,
        )
        attr = compute_circuit_attribution(
            model=model, seq_len=min(eval_config.induction_seq_length, 20),
            batch_size=32, seed=eval_config.seed, device=device,
        )

        steps.append(payload["step"])
        train_losses.append(float(payload.get("loss", float("nan"))))
        ind_list.append(ind.cpu().numpy())
        attr_list.append(attr.cpu().numpy())

        logger.info("  step=%d | loss=%.4f | mean_IS=%.4f | n_circuit=%d",
            payload["step"], train_losses[-1], float(ind.mean()),
            int((attr >= eval_config.induction_circuit_threshold).sum()))

    result = {
        "steps": np.array(steps),
        "train_losses": np.array(train_losses),
        "induction_scores": np.stack(ind_list),    # [n_ckpts, n_layers, n_heads]
        "attribution_scores": np.stack(attr_list),  # [n_ckpts, n_layers, n_heads]
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **result)
    logger.info("Sweep saved to %s", output_path)
    return result


def load_sweep_results(path: Path) -> dict:
    """Load a saved sweep .npz file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sweep results not found: {path}")
    return dict(np.load(path))


def compute_induction_score_delta(sweep_results: dict) -> np.ndarray:
    """Per-head delta from step-0 baseline. Returns [n_ckpts, n_layers, n_heads]."""
    ind = sweep_results["induction_scores"]
    return ind - ind[0][np.newaxis, :, :]
