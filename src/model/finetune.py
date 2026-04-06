"""Fine-tuning loop with integrated circuit tracking."""
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
from src.circuits.patching import compute_circuit_attribution
from src.model.config import ModelConfig, TrainConfig
from src.model.train import build_scheduler, load_pretrained_model, save_checkpoint, set_global_seed

logger = logging.getLogger(__name__)


class TokenisedStreamDataset(IterableDataset):
    """Stream a HuggingFace dataset, tokenise on-the-fly, yield fixed-length chunks."""

    def __init__(self, hf_dataset_name, hf_subset, tokenizer, seq_length=128,
                 max_tokens=500_000, text_column="content", seed=42):
        self.hf_dataset_name = hf_dataset_name
        self.hf_subset = hf_subset
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.max_tokens = max_tokens
        self.text_column = text_column
        self.seed = seed

    def __iter__(self):
        dataset = load_dataset(
            self.hf_dataset_name, self.hf_subset, split="train", streaming=True,
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
                chunk = buffer[:self.seq_length]
                buffer = buffer[self.seq_length:]
                yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}
                tokens_yielded += self.seq_length


def build_code_dataloader(config: TrainConfig, tokenizer) -> DataLoader:
    """DataLoader for Python code fine-tuning (codeparrot/github-code)."""
    dataset = TokenisedStreamDataset(
        hf_dataset_name=config.dataset, hf_subset=config.dataset_subset,
        tokenizer=tokenizer, seq_length=config.seq_length,
        max_tokens=config.max_tokens, text_column="code", seed=config.seed,
    )
    return DataLoader(dataset, batch_size=config.batch_size)


def build_prose_dataloader(config: TrainConfig, tokenizer) -> DataLoader:
    """DataLoader for TinyStories prose control run."""
    dataset = TokenisedStreamDataset(
        hf_dataset_name="roneneldan/TinyStories", hf_subset=None,
        tokenizer=tokenizer, seq_length=config.seq_length,
        max_tokens=config.max_tokens, text_column="text", seed=config.seed,
    )
    return DataLoader(dataset, batch_size=config.batch_size)


def compute_code_icl_score(model, tokenizer, num_examples=50, seed=42, device=None) -> float:
    """ICL accuracy on code copy-numeric-pattern task.

    Prompt: def increment(x): return x + N\ndef add_value(x): return x + 
    Model should predict N. Returns fraction correct.
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
            tokenizer.encode(demo, add_special_tokens=False), dtype=torch.long, device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            logits = model(ids)
        pred = tokenizer.decode([int(logits[0, -1].argmax())]).strip()
        if pred == expected:
            correct += 1
    return correct / num_examples


def run_finetuning(
    model_config: ModelConfig,
    train_config: TrainConfig,
    run_name: str,
    device: Optional[str] = None,
    prose_control: bool = False,
) -> list[dict]:
    """Execute the full fine-tuning experiment with circuit tracking.

    Args:
        model_config: Architecture specification.
        train_config: Training hyperparameters.
        run_name: Short identifier for this run.
        device: Torch device. Auto-detected if None.
        prose_control: If True, use TinyStories instead of codeparrot.

    Returns:
        List of per-checkpoint metadata dicts.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_global_seed(train_config.seed)
    logger.info("Fine-tuning run: %s | device: %s", run_name, device)

    model = load_pretrained_model(model_config, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step-0 baseline
    model.eval()
    baseline_ind = compute_induction_score(
        model=model, sequence_length=train_config.induction_seq_length,
        num_sequences=train_config.induction_num_sequences,
        seed=train_config.seed, device=device,
    )
    logger.info("Baseline induction score mean: %.4f", float(baseline_ind.mean()))

    ckpt_dir = Path(train_config.checkpoint_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    opt_init = torch.optim.AdamW(model.parameters(), lr=train_config.lr)
    save_checkpoint(
        path=ckpt_dir / "step_000000.pt", step=0, model=model, optimizer=opt_init,
        loss=float("nan"), induction_scores=baseline_ind, config=train_config, seed=train_config.seed,
    )

    dataloader = build_prose_dataloader(train_config, tokenizer) if prose_control else build_code_dataloader(train_config, tokenizer)

    def _induction_fn(m):
        return compute_induction_score(
            model=m, sequence_length=train_config.induction_seq_length,
            num_sequences=train_config.induction_num_sequences,
            seed=train_config.seed, device=device,
        )

    train_config_copy = dataclasses.replace(train_config, checkpoint_dir=str(ckpt_dir))
    from src.model.train import train
    history = train(
        model=model, dataloader=dataloader, config=train_config_copy,
        compute_induction_scores_fn=_induction_fn, device=device,
    )

    results_dir = Path(train_config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        results_dir / f"{run_name}_history.npz",
        steps=np.array([h["step"] for h in history]),
        loss=np.array([h["loss"] for h in history]),
        induction_scores_mean=np.array([h["induction_scores_mean"] for h in history]),
    )
    logger.info("Run summary saved.")
    return history
