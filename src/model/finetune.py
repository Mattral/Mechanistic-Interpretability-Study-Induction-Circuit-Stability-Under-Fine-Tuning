"""Fine-tuning loop with integrated circuit tracking.

This module wraps the generic training loop from train.py and adds:
- Per-checkpoint induction score computation (mandatory, not optional)
- Code ICL score computation
- Full metric logging as defined in Section 4.5 of AGENT_INSTRUCTIONS
- Dataset loading for both code (codeparrot) and prose (TinyStories) runs

The fine-tuning protocol is specified exactly in AGENT_INSTRUCTIONS Section 4.4.
Deviations require a DECISION_LOG entry.
"""

from __future__ import annotations

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
from src.circuits.patching import compute_circuit_attribution
from src.model.config import ModelConfig, TrainConfig
from src.model.train import build_scheduler, load_pretrained_model, save_checkpoint, set_global_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


class TokenisedStreamDataset(IterableDataset):
    """Stream a Hugging Face dataset, tokenise on-the-fly, and yield fixed-length chunks.

    Args:
        hf_dataset_name: Hugging Face dataset identifier.
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


def build_code_dataloader(config: TrainConfig, tokenizer: AutoTokenizer) -> DataLoader:
    """Build the DataLoader for Python code fine-tuning.

    Uses codeparrot/github-code, Python subset, as specified in Section 4.4.

    Args:
        config: TrainConfig with dataset and tokenization settings.
        tokenizer: Pre-loaded GPT-2 tokenizer.

    Returns:
        A DataLoader ready for the fine-tuning loop.
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


def build_prose_dataloader(config: TrainConfig, tokenizer: AutoTokenizer) -> DataLoader:
    """Build the DataLoader for prose (TinyStories) control fine-tuning.

    The prose control is mandatory — see AGENT_INSTRUCTIONS Section 4.4.

    Args:
        config: TrainConfig. dataset field will be overridden to TinyStories.
        tokenizer: Pre-loaded GPT-2 tokenizer.

    Returns:
        A DataLoader for the TinyStories control run.
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
# Code ICL score
# ---------------------------------------------------------------------------


def compute_code_icl_score(
    model: transformer_lens.HookedTransformer,
    tokenizer: AutoTokenizer,
    num_examples: int = 50,
    seed: int = 42,
    device: Optional[str] = None,
) -> float:
    """Compute in-context learning accuracy on code-specific completion patterns.

    The ICL score measures whether the model can complete a code pattern
    (function signature → body) given one in-context demonstration.

    We use a simple proxy: given a prompt of the form
        <demo_signature> <demo_body> <test_signature>
    the model should predict tokens that match <test_body> more than random.

    Score is fraction of test examples where the argmax next-token prediction
    matches the first token of the expected continuation.

    Args:
        model: The fine-tuned HookedTransformer.
        tokenizer: GPT-2 tokenizer.
        num_examples: Number of ICL test pairs to evaluate.
        seed: Random seed for sequence construction.
        device: Torch device. Inferred from model if None.

    Returns:
        Float in [0, 1] representing ICL accuracy.
    """
    if device is None:
        device = next(model.parameters()).device.type

    rng = np.random.default_rng(seed)
    model.eval()

    # Simple Python patterns: def foo(x): return x + <N>
    # In-context demo: model should copy the numeric increment
    correct = 0
    for _ in range(num_examples):
        n = int(rng.integers(1, 20))
        demo_text = f"def increment(x): return x + {n}\ndef add_value(x): return x + "
        expected_token = str(n)

        input_ids = torch.tensor(
            tokenizer.encode(demo_text, add_special_tokens=False),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        with torch.no_grad():
            logits = model(input_ids)  # [1, seq, vocab]
        next_token_id = int(logits[0, -1].argmax())
        predicted = tokenizer.decode([next_token_id]).strip()

        if predicted == expected_token:
            correct += 1

    score = correct / num_examples
    logger.debug("Code ICL score: %.3f (%d/%d)", score, correct, num_examples)
    return score


# ---------------------------------------------------------------------------
# Full metrics snapshot
# ---------------------------------------------------------------------------


def compute_checkpoint_metrics(
    model: transformer_lens.HookedTransformer,
    tokenizer: AutoTokenizer,
    induction_scores_baseline: torch.Tensor,
    seed: int = 42,
    device: Optional[str] = None,
) -> dict:
    """Compute the full set of metrics required at every checkpoint (Section 4.5).

    Args:
        model: The model at the current checkpoint.
        tokenizer: GPT-2 tokenizer.
        induction_scores_baseline: Induction scores at step 0, for delta computation.
        seed: Random seed.
        device: Torch device.

    Returns:
        Dictionary with all metric fields from Section 4.5.
    """
    if device is None:
        device = next(model.parameters()).device.type

    induction_scores = compute_induction_score(
        model=model,
        sequence_length=50,
        num_sequences=500,
        seed=seed,
        device=device,
    )
    induction_scores_delta = induction_scores - induction_scores_baseline

    attribution = compute_circuit_attribution(
        model=model,
        seed=seed,
        device=device,
    )

    code_icl = compute_code_icl_score(
        model=model,
        tokenizer=tokenizer,
        num_examples=50,
        seed=seed,
        device=device,
    )

    return {
        "induction_score": induction_scores.cpu().numpy(),
        "induction_score_delta": induction_scores_delta.cpu().numpy(),
        "code_icl_score": code_icl,
        "circuit_attribution": attribution.cpu().numpy(),
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
) -> list[dict]:
    """Execute the full fine-tuning experiment with circuit tracking.

    This is the primary entry point for both the code fine-tuning run and
    the prose control run. Both are required by Section 4.4.

    Args:
        model_config: Architecture specification (defaults to attn-only-2l).
        train_config: Full training hyperparameters.
        run_name: A short identifier for this run (e.g. 'code_seed42').
            Used to name checkpoint directories and results files.
        device: Torch device string. Auto-detected if None.
        prose_control: If True, use TinyStories instead of codeparrot.

    Returns:
        List of per-checkpoint metadata dicts for downstream analysis.

    Raises:
        RuntimeError: If the model cannot be loaded or training fails.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_global_seed(train_config.seed)
    logger.info("Starting fine-tuning run: %s | device: %s", run_name, device)

    # Load model
    model = load_pretrained_model(model_config, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Compute step-0 baseline induction scores (before any fine-tuning)
    model.eval()
    baseline_induction_scores = compute_induction_score(
        model=model,
        sequence_length=train_config.induction_seq_length,
        num_sequences=train_config.induction_num_sequences,
        seed=train_config.seed,
        device=device,
    )
    logger.info(
        "Baseline induction score (mean): %.4f",
        float(baseline_induction_scores.mean()),
    )

    # Save step-0 checkpoint
    ckpt_dir = Path(train_config.checkpoint_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    optimizer_init = torch.optim.AdamW(model.parameters(), lr=train_config.lr)
    save_checkpoint(
        path=ckpt_dir / "step_000000.pt",
        step=0,
        model=model,
        optimizer=optimizer_init,
        loss=float("nan"),
        induction_scores=baseline_induction_scores,
        config=train_config,
        seed=train_config.seed,
    )

    # Build dataloader
    if prose_control:
        dataloader = build_prose_dataloader(train_config, tokenizer)
    else:
        dataloader = build_code_dataloader(train_config, tokenizer)

    # Define the induction score computation closure for the training loop
    def _induction_fn(m: transformer_lens.HookedTransformer) -> torch.Tensor:
        return compute_induction_score(
            model=m,
            sequence_length=train_config.induction_seq_length,
            num_sequences=train_config.induction_num_sequences,
            seed=train_config.seed,
            device=device,
        )

    # Override checkpoint dir for this run
    train_config_copy = dataclasses.replace(train_config, checkpoint_dir=str(ckpt_dir))  # type: ignore[attr-defined]

    from src.model.train import train
    history = train(
        model=model,
        dataloader=dataloader,
        config=train_config_copy,
        compute_induction_scores_fn=_induction_fn,
        device=device,
    )

    # Persist the full run summary
    results_dir = Path(train_config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / f"{run_name}_history.npz"
    np.savez(
        summary_path,
        steps=np.array([h["step"] for h in history]),
        loss=np.array([h["loss"] for h in history]),
        induction_scores_mean=np.array([h["induction_scores_mean"] for h in history]),
    )
    logger.info("Run summary saved to %s", summary_path)

    return history


# Needed for dataclasses.replace inside run_finetuning
import dataclasses  # noqa: E402 (placed here to avoid circular at top-level)
