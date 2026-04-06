"""Pre-training loop and checkpointing utilities.

This module handles the training loop for the baseline attention-only
transformer and provides checkpoint save/load functionality. All experiments
that involve training from scratch (as opposed to fine-tuning a pretrained
model) use this module.
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import transformer_lens
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.model.config import ModelConfig, TrainConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_global_seed(seed: int = 42) -> None:
    """Set all random seeds for full determinism.

    Must be called at the top of every notebook cell that trains or evaluates
    a model.

    Args:
        seed: Integer random seed applied to Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to %d", seed)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_pretrained_model(
    config: ModelConfig,
    device: Optional[str] = None,
) -> transformer_lens.HookedTransformer:
    """Load the pretrained attention-only transformer from TransformerLens.

    Args:
        config: ModelConfig specifying which pretrained model to load.
        device: Torch device string. Defaults to CUDA if available, else CPU.

    Returns:
        A HookedTransformer instance ready for inference or fine-tuning.

    Raises:
        ValueError: If config.source is not a known TransformerLens model name.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading pretrained model '%s' onto device '%s'", config.source, device)

    model = transformer_lens.HookedTransformer.from_pretrained(
        config.source,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )
    model = model.to(device)
    model.eval()
    logger.info(
        "Model loaded: %d layers, %d heads, d_model=%d",
        model.cfg.n_layers,
        model.cfg.n_heads,
        model.cfg.d_model,
    )
    return model


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    step: int,
    model: transformer_lens.HookedTransformer,
    optimizer: torch.optim.Optimizer,
    loss: float,
    induction_scores: torch.Tensor,
    config: TrainConfig,
    seed: int,
) -> None:
    """Save a full training checkpoint with metadata.

    The checkpoint includes model weights, optimiser state, metrics computed
    at save time, and full provenance metadata so results remain reproducible
    months later.

    Args:
        path: File path for the checkpoint (.pt).
        step: Current gradient step number.
        model: The HookedTransformer instance to save.
        optimizer: The current optimizer, including its state dict.
        loss: Most recent training loss scalar.
        induction_scores: Float tensor of shape [n_layers, n_heads] computed
            at this checkpoint.
        config: TrainConfig used for this run.
        seed: The random seed active at checkpoint time.

    Raises:
        OSError: If the checkpoint directory cannot be created or written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": float(loss),
        "induction_scores": induction_scores.cpu(),
        "config": dataclasses.asdict(config),
        "seed": seed,
        "transformer_lens_version": transformer_lens.__version__,
        "torch_version": torch.__version__,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }

    torch.save(payload, path)
    logger.info("Checkpoint saved at step %d → %s", step, path)


def load_checkpoint(
    path: Path,
    model: transformer_lens.HookedTransformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """Load a checkpoint and restore model (and optionally optimizer) state.

    Args:
        path: Path to a checkpoint file saved by save_checkpoint().
        model: The HookedTransformer instance to restore weights into.
        optimizer: If provided, the optimizer state dict is also restored.

    Returns:
        The full checkpoint dictionary including metadata fields.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    logger.info(
        "Checkpoint loaded from step %d (timestamp: %s)",
        payload["step"],
        payload.get("timestamp", "unknown"),
    )
    return payload


# ---------------------------------------------------------------------------
# Scheduler helper
# ---------------------------------------------------------------------------


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build a cosine decay scheduler with linear warmup.

    Args:
        optimizer: The optimizer whose learning rate will be scheduled.
        total_steps: Total number of gradient steps in the run.
        warmup_steps: Number of steps for the linear warmup phase.

    Returns:
        A SequentialLR that applies linear warmup then cosine decay.
    """
    warmup = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=0.0,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    model: transformer_lens.HookedTransformer,
    dataloader: DataLoader,
    config: TrainConfig,
    compute_induction_scores_fn: Optional[object] = None,
    device: Optional[str] = None,
) -> list[dict]:
    """Run the fine-tuning training loop with automatic checkpointing.

    This function is intentionally generic — it can be used for both the
    code fine-tuning run and the prose control run by swapping the dataloader.

    Args:
        model: A loaded HookedTransformer to fine-tune (in-place).
        dataloader: DataLoader yielding tokenised input_ids tensors.
        config: TrainConfig with all hyperparameters.
        compute_induction_scores_fn: Callable(model) → Tensor[layer, head].
            If None, induction scores are skipped (not recommended).
        device: Torch device string. Inferred from model parameters if None.

    Returns:
        List of checkpoint metadata dicts (one per checkpoint, no tensors) for
        downstream analysis without loading all checkpoint files.

    Raises:
        RuntimeError: If the dataloader yields batches of unexpected shape.
    """
    if device is None:
        device = next(model.parameters()).device.type

    set_global_seed(config.seed)
    model.train()
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = build_scheduler(optimizer, config.total_steps, config.warmup_steps)

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    step = 0
    tokens_seen = 0

    logger.info(
        "Starting training: %d total steps, checkpointing every %d steps",
        config.total_steps,
        config.checkpoint_every,
    )

    for batch in dataloader:
        if step >= config.total_steps:
            break

        input_ids: torch.Tensor = batch["input_ids"].to(device)

        # Forward pass — use model's built-in loss computation
        loss = model(input_ids, return_type="loss")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        tokens_seen += input_ids.numel()
        step += 1

        current_lr = scheduler.get_last_lr()[0]

        if step % 10 == 0:
            logger.info(
                "step=%d | loss=%.4f | lr=%.2e | tokens=%d",
                step,
                loss.item(),
                current_lr,
                tokens_seen,
            )

        # Checkpoint and induction score evaluation
        if step % config.checkpoint_every == 0:
            model.eval()

            if compute_induction_scores_fn is not None:
                induction_scores = compute_induction_scores_fn(model)
            else:
                induction_scores = torch.zeros(
                    model.cfg.n_layers, model.cfg.n_heads, dtype=torch.float32
                )

            ckpt_path = checkpoint_dir / f"step_{step:06d}.pt"
            save_checkpoint(
                path=ckpt_path,
                step=step,
                model=model,
                optimizer=optimizer,
                loss=loss.item(),
                induction_scores=induction_scores,
                config=config,
                seed=config.seed,
            )

            # Record lightweight metadata for callers
            history.append(
                {
                    "step": step,
                    "loss": loss.item(),
                    "lr": current_lr,
                    "tokens_seen": tokens_seen,
                    "induction_scores_mean": float(induction_scores.mean()),
                    "checkpoint_path": str(ckpt_path),
                }
            )

            model.train()

    logger.info("Training complete. %d steps, %d tokens.", step, tokens_seen)
    return history
