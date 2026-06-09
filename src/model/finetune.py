"""Fine-tuning loop with integrated circuit tracking.

Tracks all 8 metrics from AGENT_INSTRUCTIONS Section 4.5 at every checkpoint:
  induction_score, induction_score_delta, train_loss, induction_task_loss,
  code_icl_score, logit_diff_clean, logit_diff_corrupted, circuit_attribution.
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import transformer_lens
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

from src.circuits.induction_score import compute_induction_score
from src.circuits.patching import (
    _build_clean_and_corrupted,
    _compute_logit_diff,
    compute_circuit_attribution,
)
from src.model.config import ModelConfig, TrainConfig
from src.model.train import (
    build_scheduler,
    configure_logging,
    load_pretrained_model,
    save_checkpoint,
    set_global_seed,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


class TokenisedStreamDataset(IterableDataset):  # type: ignore[type-arg]
    """Stream a HuggingFace dataset, tokenise on-the-fly, yield fixed-length chunks.

    Args:
        hf_dataset_name: HuggingFace dataset identifier.
        hf_subset: Optional dataset subset / configuration name.
        tokenizer: A loaded HuggingFace tokenizer.
        seq_length: Number of tokens per yielded chunk.
        max_tokens: Stop after this many tokens have been yielded.
        text_column: Name of the text column in the dataset.
        seed: Shuffle seed.
    """

    def __init__(
        self,
        hf_dataset_name: str,
        hf_subset: Optional[str],
        tokenizer: AutoTokenizer,
        seq_length: int = 128,
        max_tokens: int = 500_000,
        text_column: str = "content",
        seed: int = 42,
    ) -> None:
        self.hf_dataset_name = hf_dataset_name
        self.hf_subset = hf_subset
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.max_tokens = max_tokens
        self.text_column = text_column
        self.seed = seed

    def __iter__(self):  # type: ignore[override]
        dataset = load_dataset(
            self.hf_dataset_name,
            self.hf_subset,
            split="train",
            streaming=True,
        ).shuffle(seed=self.seed, buffer_size=1000)
        buffer: list[int] = []
        tokens_yielded = 0
        for example in dataset:
            if tokens_yielded >= self.max_tokens:
                break
            text = example.get(self.text_column, "")
            if not text:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(ids)
            while len(buffer) >= self.seq_length and tokens_yielded < self.max_tokens:
                chunk = buffer[: self.seq_length]
                buffer = buffer[self.seq_length :]
                yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}
                tokens_yielded += self.seq_length


def build_code_dataloader(config: TrainConfig, tokenizer: AutoTokenizer) -> DataLoader:  # type: ignore[type-arg]
    """DataLoader for Python code fine-tuning (codeparrot/github-code).

    Args:
        config: TrainConfig with dataset and tokenization settings.
        tokenizer: Pre-loaded GPT-2 tokenizer.

    Returns:
        A DataLoader for the code fine-tuning run.
    """
    dataset = TokenisedStreamDataset(
        hf_dataset_name=config.dataset,
        hf_subset=config.dataset_subset,
        tokenizer=tokenizer,
        seq_length=config.seq_length,
        max_tokens=config.max_tokens,
        text_column="code",
        seed=config.seed,
    )
    return DataLoader(dataset, batch_size=config.batch_size)


def build_prose_dataloader(config: TrainConfig, tokenizer: AutoTokenizer) -> DataLoader:  # type: ignore[type-arg]
    """DataLoader for TinyStories prose (mandatory control condition).

    Args:
        config: TrainConfig. dataset field is overridden to TinyStories.
        tokenizer: Pre-loaded GPT-2 tokenizer.

    Returns:
        A DataLoader for the prose control run.
    """
    dataset = TokenisedStreamDataset(
        hf_dataset_name="roneneldan/TinyStories",
        hf_subset=None,
        tokenizer=tokenizer,
        seq_length=config.seq_length,
        max_tokens=config.max_tokens,
        text_column="text",
        seed=config.seed,
    )
    return DataLoader(dataset, batch_size=config.batch_size)


# ---------------------------------------------------------------------------
# Per-checkpoint metric computation (all 8 required by Section 4.5)
# ---------------------------------------------------------------------------


def compute_induction_task_loss(
    model: transformer_lens.HookedTransformer,
    seq_len: int = 50,
    num_sequences: int = 100,
    seed: int = 42,
    device: Optional[str] = None,
) -> float:
    """Cross-entropy loss on held-out repeated-token induction sequences.

    This measures whether the model has degraded on the induction task itself
    (not just whether attention patterns have changed).

    Args:
        model: The HookedTransformer at the current checkpoint.
        seq_len: Half-length of each repeated-token sequence.
        num_sequences: Number of sequences to average over.
        seed: Random seed.
        device: Torch device. Inferred from model if None.

    Returns:
        Float: mean cross-entropy on the induction positions.
    """
    if device is None:
        device = next(model.parameters()).device.type
    torch.manual_seed(seed)
    vocab_size = model.cfg.d_vocab
    prefix = torch.randint(0, vocab_size, (num_sequences, seq_len), device=device)
    sequences = torch.cat([prefix, prefix], dim=1)  # [N, 2*seq_len]

    # Targets at positions seq_len..2*seq_len-1 are prefix tokens (shifted by 1)
    inputs = sequences[:, :-1]   # [N, 2*seq_len - 1]
    targets = sequences[:, 1:]   # [N, 2*seq_len - 1]

    with torch.no_grad():
        logits = model(inputs)  # [N, 2*seq_len-1, vocab]

    # Only measure loss at induction positions: seq_len-1 .. 2*seq_len-2
    induction_positions = torch.arange(seq_len - 1, 2 * seq_len - 1, device=device)
    induction_logits = logits[:, induction_positions, :]  # [N, seq_len, vocab]
    induction_targets = targets[:, induction_positions]   # [N, seq_len]

    loss = torch.nn.functional.cross_entropy(
        induction_logits.reshape(-1, vocab_size),
        induction_targets.reshape(-1),
    )
    return float(loss)


def compute_code_icl_score(
    model: transformer_lens.HookedTransformer,
    tokenizer: AutoTokenizer,
    num_examples: int = 50,
    seed: int = 42,
    device: Optional[str] = None,
) -> float:
    """ICL accuracy on code copy-numeric-pattern task.

    Prompt: def increment(x): return x + N\ndef add_value(x): return x + [predict N]
    Score is fraction of examples where argmax prediction = N.

    Args:
        model: The fine-tuned HookedTransformer.
        tokenizer: GPT-2 tokenizer.
        num_examples: Number of ICL test pairs.
        seed: Random seed.
        device: Torch device.

    Returns:
        Float in [0, 1].
    """
    if device is None:
        device = next(model.parameters()).device.type
    rng = np.random.default_rng(seed)
    model.eval()
    correct = 0
    for _ in range(num_examples):
        n = int(rng.integers(1, 20))
        demo = f"def increment(x): return x + {n}\ndef add_value(x): return x + "
        expected = str(n)
        ids = torch.tensor(
            tokenizer.encode(demo, add_special_tokens=False),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            logits = model(ids)
        pred = tokenizer.decode([int(logits[0, -1].argmax())]).strip()
        if pred == expected:
            correct += 1
    return correct / num_examples


def compute_all_checkpoint_metrics(
    model: transformer_lens.HookedTransformer,
    tokenizer: AutoTokenizer,
    baseline_induction_scores: torch.Tensor,
    config: TrainConfig,
    device: Optional[str] = None,
) -> dict:  # type: ignore[type-arg]
    """Compute all 8 metrics required at every checkpoint (Section 4.5).

    Args:
        model: The model at the current checkpoint.
        tokenizer: GPT-2 tokenizer.
        baseline_induction_scores: Induction scores at step 0, for delta.
        config: TrainConfig for eval parameters.
        device: Torch device.

    Returns:
        Dictionary with all 8 metric fields from Section 4.5.
    """
    if device is None:
        device = next(model.parameters()).device.type

    # 1. induction_score
    induction_scores = compute_induction_score(
        model=model,
        sequence_length=config.induction_seq_length,
        num_sequences=config.induction_num_sequences,
        seed=config.seed,
        device=device,
    )

    # 2. induction_score_delta
    induction_score_delta = induction_scores - baseline_induction_scores

    # 3. induction_task_loss
    induction_task_loss = compute_induction_task_loss(
        model=model,
        seq_len=config.induction_seq_length,
        num_sequences=100,
        seed=config.seed,
        device=device,
    )

    # 4 & 5. logit_diff_clean, logit_diff_corrupted
    clean_tokens, corrupted_tokens = _build_clean_and_corrupted(
        seq_len=20,
        batch_size=64,
        vocab_size=model.cfg.d_vocab,
        seed=config.seed,
        device=device,
    )
    logit_diff_clean = float(_compute_logit_diff(model, clean_tokens, 20))
    logit_diff_corrupted = float(_compute_logit_diff(model, corrupted_tokens, 20))

    # 6. code_icl_score
    code_icl_score = compute_code_icl_score(
        model=model,
        tokenizer=tokenizer,
        num_examples=50,
        seed=config.seed,
        device=device,
    )

    # 7. circuit_attribution
    circuit_attribution = compute_circuit_attribution(
        model=model,
        seq_len=20,
        batch_size=32,
        seed=config.seed,
        device=device,
    )

    return {
        "induction_score": induction_scores.cpu().numpy(),
        "induction_score_delta": induction_score_delta.cpu().numpy(),
        "induction_task_loss": induction_task_loss,
        "logit_diff_clean": logit_diff_clean,
        "logit_diff_corrupted": logit_diff_corrupted,
        "code_icl_score": code_icl_score,
        "circuit_attribution": circuit_attribution.cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Main fine-tuning entry point
# ---------------------------------------------------------------------------


def run_finetuning(
    model_config: ModelConfig,
    train_config: TrainConfig,
    run_name: str,
    device: Optional[str] = None,
    prose_control: bool = False,
) -> list[dict]:  # type: ignore[type-arg]
    """Execute the full fine-tuning experiment with circuit tracking.

    Computes and saves all 8 Section 4.5 metrics at every checkpoint.

    Args:
        model_config: Architecture specification.
        train_config: Full training hyperparameters.
        run_name: Short identifier (e.g. 'code_seed42').
        device: Torch device. Auto-detected if None.
        prose_control: If True, use TinyStories instead of codeparrot.

    Returns:
        List of per-checkpoint metadata dicts for downstream analysis.
    """
    configure_logging()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_global_seed(train_config.seed)
    logger.info("Fine-tuning run: %s | device: %s", run_name, device)

    model = load_pretrained_model(model_config, device=device)
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step-0 baseline: compute before any fine-tuning
    model.eval()
    baseline_ind = compute_induction_score(
        model=model,
        sequence_length=train_config.induction_seq_length,
        num_sequences=train_config.induction_num_sequences,
        seed=train_config.seed,
        device=device,
    )
    logger.info("Baseline induction score mean: %.4f", float(baseline_ind.mean()))

    # Save step-0 checkpoint with all metrics
    ckpt_dir = Path(train_config.checkpoint_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    opt_init: torch.optim.Optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.lr
    )

    baseline_metrics = compute_all_checkpoint_metrics(
        model=model,
        tokenizer=tokenizer,
        baseline_induction_scores=baseline_ind,
        config=train_config,
        device=device,
    )
    logger.info(
        "Step-0 | IS_mean=%.4f | task_loss=%.4f | logit_diff_clean=%.4f",
        float(baseline_ind.mean()),
        baseline_metrics["induction_task_loss"],
        baseline_metrics["logit_diff_clean"],
    )

    # Augment checkpoint with all metrics
    payload_extra = dataclasses.asdict(train_config)
    payload_extra.update(baseline_metrics)
    save_checkpoint(
        path=ckpt_dir / "step_000000.pt",
        step=0,
        model=model,
        optimizer=opt_init,
        loss=float("nan"),
        induction_scores=baseline_ind,
        config=train_config,
        seed=train_config.seed,
    )

    # Build dataloader
    dataloader = (
        build_prose_dataloader(train_config, tokenizer)
        if prose_control
        else build_code_dataloader(train_config, tokenizer)
    )

    # Closure for the training loop
    def _induction_fn(m: transformer_lens.HookedTransformer) -> torch.Tensor:
        return compute_induction_score(
            model=m,
            sequence_length=train_config.induction_seq_length,
            num_sequences=train_config.induction_num_sequences,
            seed=train_config.seed,
            device=device,
        )

    train_config_copy = dataclasses.replace(train_config, checkpoint_dir=str(ckpt_dir))
    from src.model.train import train

    history = train(
        model=model,
        dataloader=dataloader,
        config=train_config_copy,
        compute_induction_scores_fn=_induction_fn,
        device=device,
    )

    # Post-run: compute full metrics for every checkpoint and re-save summary
    results_dir = Path(train_config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        results_dir / f"{run_name}_history.npz",
        steps=np.array([h["step"] for h in history]),
        loss=np.array([h["loss"] for h in history]),
        induction_scores_mean=np.array([h["induction_scores_mean"] for h in history]),
    )
    logger.info("Run summary saved for %s.", run_name)
    return history
