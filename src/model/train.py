"""Pre-training loop and checkpointing utilities."""
from __future__ import annotations

import dataclasses
import datetime
import importlib.metadata
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


def _get_transformer_lens_version() -> str:
    """Return the installed transformer_lens version string.

    Some transformer_lens releases (notably recent pip-installed versions
    on Colab) do not expose a top-level ``__version__`` attribute, which
    raises AttributeError if accessed directly. This falls back to
    importlib.metadata, and finally to "unknown" if neither works.

    Returns:
        Version string, or "unknown" if it cannot be determined.
    """
    try:
        return str(transformer_lens.__version__)
    except AttributeError:
        try:
            return importlib.metadata.version("transformer_lens")
        except importlib.metadata.PackageNotFoundError:
            return "unknown"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger with the project-standard format.

    Must be called once at the entry point of any script or notebook that
    uses this package. Format matches Section 3.6 of AGENT_INSTRUCTIONS.

    Args:
        level: Logging level (default: logging.INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )


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
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading model '%s' on device '%s'", config.source, device)
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
        "Model loaded: %dL %dH d_model=%d",
        model.cfg.n_layers,
        model.cfg.n_heads,
        model.cfg.d_model,
    )
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
    """Save a full training checkpoint with provenance metadata.

    The checkpoint includes model weights, optimiser state, metrics, and full
    provenance metadata so results remain reproducible months later.

    Args:
        path: File path for the checkpoint (.pt).
        step: Current gradient step number.
        model: The HookedTransformer instance to save.
        optimizer: The current optimizer, including its state dict.
        loss: Most recent training loss scalar.
        induction_scores: Float tensor [n_layers, n_heads] computed at this step.
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
        "transformer_lens_version": _get_transformer_lens_version(),
        "torch_version": str(torch.__version__),  # cast to str; TorchVersion object not safe in PyTorch>=2.6
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    torch.save(payload, path)
    logger.info("Checkpoint saved step=%d -> %s", step, path)


def load_checkpoint(
    path: Path,
    model: transformer_lens.HookedTransformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:  # type: ignore[type-arg]
    """Load a checkpoint and restore model (and optionally optimizer) state.

    Args:
        path: Path to a checkpoint file saved by save_checkpoint().
        model: The HookedTransformer instance to restore weights into.
        optimizer: If provided, the optimizer state dict is also restored.

    Returns:
        The full checkpoint dictionary including metadata.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        # Fallback for checkpoints saved with PyTorch <2.6 that contain
        # TorchVersion objects or other non-allowlisted globals. Log a
        # warning but proceed — these checkpoints come from our own
        # training runs and are trusted.
        logger.warning(
            "torch.load(weights_only=True) failed for %s (%s); "
            "retrying with weights_only=False. Upgrade to v8 codebase "
            "to generate checkpoints safe for weights_only=True.",
            path.name, type(e).__name__,
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)  # noqa: S614
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    logger.info(
        "Loaded checkpoint step=%d (timestamp: %s)",
        payload["step"],
        payload.get("timestamp", "unknown"),
    )
    return payload  # type: ignore[return-value]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Cosine decay scheduler with linear warmup.

    Args:
        optimizer: The optimizer whose learning rate will be scheduled.
        total_steps: Total number of gradient steps in the run.
        warmup_steps: Number of steps for the linear warmup phase.

    Returns:
        A SequentialLR that applies linear warmup then cosine decay.
    """
    warmup = LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=0.0
    )
    return SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )


def train(
    model: transformer_lens.HookedTransformer,
    dataloader: DataLoader,  # type: ignore[type-arg]
    config: TrainConfig,
    compute_induction_scores_fn: Optional[object] = None,
    device: Optional[str] = None,
) -> list[dict]:  # type: ignore[type-arg]
    """Run the fine-tuning training loop with automatic checkpointing.

    Args:
        model: A loaded HookedTransformer to fine-tune (in-place).
        dataloader: DataLoader yielding dicts with input_ids tensors.
        config: TrainConfig with all hyperparameters.
        compute_induction_scores_fn: Callable(model) -> Tensor[layer, head].
            If None, induction scores are skipped.
        device: Torch device string. Inferred from model parameters if None.

    Returns:
        List of checkpoint metadata dicts for downstream analysis.
    """
    if device is None:
        device = next(model.parameters()).device.type

    set_global_seed(config.seed)
    model.train()
    model.to(device)

    optimizer = AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = build_scheduler(optimizer, config.total_steps, config.warmup_steps)
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []  # type: ignore[type-arg]
    step = 0
    tokens_seen = 0
    logger.info(
        "Training: %d total steps, checkpoint every %d",
        config.total_steps,
        config.checkpoint_every,
    )

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

        current_lr = scheduler.get_last_lr()[0]
        if step % 10 == 0:
            logger.info(
                "step=%d | loss=%.4f | lr=%.2e | tokens=%d",
                step,
                loss.item(),
                current_lr,
                tokens_seen,
            )

        if step % config.checkpoint_every == 0:
            model.eval()
            if compute_induction_scores_fn is not None:
                ind_scores = compute_induction_scores_fn(model)  # type: ignore[operator]
            else:
                ind_scores = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)

            ckpt_path = ckpt_dir / f"step_{step:06d}.pt"
            save_checkpoint(
                path=ckpt_path,
                step=step,
                model=model,
                optimizer=optimizer,
                loss=loss.item(),
                induction_scores=ind_scores,
                config=config,
                seed=config.seed,
            )
            history.append(
                {
                    "step": step,
                    "loss": loss.item(),
                    "lr": current_lr,
                    "tokens_seen": tokens_seen,
                    "induction_scores_mean": float(ind_scores.mean()),
                    "checkpoint_path": str(ckpt_path),
                }
            )
            model.train()

    logger.info("Training complete: %d steps, %d tokens", step, tokens_seen)
    return history
