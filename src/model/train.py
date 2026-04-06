"""Pre-training loop and checkpointing utilities."""
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

from src.model.config import TrainConfig

logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """Set all random seeds for full determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to %d", seed)


def load_pretrained_model(
    config,
    device: Optional[str] = None,
) -> transformer_lens.HookedTransformer:
    """Load the pretrained attention-only transformer from TransformerLens."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading model '%s' on '%s'", config.source, device)
    model = transformer_lens.HookedTransformer.from_pretrained(
        config.source,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )
    model = model.to(device)
    model.eval()
    logger.info("Loaded: %dL %dH d_model=%d", model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_model)
    return model


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
    """Save a full training checkpoint with provenance metadata."""
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
    logger.info("Checkpoint saved step=%d -> %s", step, path)


def load_checkpoint(
    path: Path,
    model: transformer_lens.HookedTransformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """Load a checkpoint and restore model (and optionally optimizer) state."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    logger.info("Loaded checkpoint step=%d (%s)", payload["step"], payload.get("timestamp", "?"))
    return payload


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Cosine decay scheduler with linear warmup."""
    warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=0.0)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def train(
    model: transformer_lens.HookedTransformer,
    dataloader: DataLoader,
    config: TrainConfig,
    compute_induction_scores_fn=None,
    device: Optional[str] = None,
) -> list[dict]:
    """Run the fine-tuning training loop with automatic checkpointing.

    Returns:
        List of per-checkpoint metadata dicts for downstream analysis.
    """
    if device is None:
        device = next(model.parameters()).device.type

    set_global_seed(config.seed)
    model.train()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = build_scheduler(optimizer, config.total_steps, config.warmup_steps)
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    step = 0
    tokens_seen = 0
    logger.info("Training: %d total steps, checkpoint every %d", config.total_steps, config.checkpoint_every)

    for batch in dataloader:
        if step >= config.total_steps:
            break
        input_ids: torch.Tensor = batch["input_ids"].to(device)
        loss = model(input_ids, return_type="loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        tokens_seen += input_ids.numel()
        step += 1

        if step % 10 == 0:
            logger.info("step=%d loss=%.4f lr=%.2e tokens=%d", step, loss.item(), scheduler.get_last_lr()[0], tokens_seen)

        if step % config.checkpoint_every == 0:
            model.eval()
            if compute_induction_scores_fn is not None:
                ind_scores = compute_induction_scores_fn(model)
            else:
                ind_scores = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
            ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
            save_checkpoint(
                path=ckpt_path, step=step, model=model, optimizer=optimizer,
                loss=loss.item(), induction_scores=ind_scores, config=config, seed=config.seed,
            )
            history.append({
                "step": step, "loss": loss.item(),
                "lr": scheduler.get_last_lr()[0], "tokens_seen": tokens_seen,
                "induction_scores_mean": float(ind_scores.mean()),
                "checkpoint_path": str(ckpt_path),
            })
            model.train()

    logger.info("Training complete: %d steps, %d tokens", step, tokens_seen)
    return history
