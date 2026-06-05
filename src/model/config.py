"""Configuration dataclasses for model architecture and training.

All hyperparameters are centralised here. No magic numbers in training loops.
"""

from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class ModelConfig:
    """Specification for the attention-only transformer baseline.

    Default values match the attn-only-2l TransformerLens pretrained model.
    """

    architecture: str = "attention-only"
    num_layers: int = 2
    num_heads: int = 8
    d_model: int = 256
    d_head: int = 32
    source: str = "attn-only-2l"
    tokenizer: str = "gpt2"
    act_fn: Optional[str] = None  # None → no MLP; attention-only

    def __post_init__(self) -> None:
        if self.d_model != self.num_heads * self.d_head:
            raise ValueError(
                f"d_model ({self.d_model}) must equal num_heads * d_head "
                f"({self.num_heads} * {self.d_head} = {self.num_heads * self.d_head})"
            )


@dataclasses.dataclass
class TrainConfig:
    """Hyperparameters for the fine-tuning training loop."""

    # Data
    dataset: str = "codeparrot/github-code"
    dataset_subset: str = "Python"
    max_tokens: int = 500_000
    seq_length: int = 128
    batch_size: int = 16

    # Optimisation
    optimizer: str = "AdamW"
    lr: float = 2e-5
    lr_schedule: str = "cosine"
    warmup_fraction: float = 0.05
    weight_decay: float = 0.01

    # Checkpointing and evaluation
    checkpoint_every: int = 100
    induction_eval_every: int = 100
    induction_num_sequences: int = 500
    induction_seq_length: int = 50

    # Reproducibility
    seed: int = 42

    # Output
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "experiments/results"

    @property
    def total_steps(self) -> int:
        """Approximate total gradient steps given max_tokens budget."""
        tokens_per_batch = self.batch_size * self.seq_length
        return self.max_tokens // tokens_per_batch

    @property
    def warmup_steps(self) -> int:
        return int(self.total_steps * self.warmup_fraction)


@dataclasses.dataclass
class EvalConfig:
    """Configuration for evaluation and analysis tasks."""

    # Induction score computation
    induction_seq_length: int = 50
    induction_num_sequences: int = 500
    induction_circuit_threshold: float = 0.5  # attribution score to be in circuit

    # Phase detection
    phase_window: int = 5  # smoothing window for phase transition detection
    phase_threshold: float = 0.1  # min change in induction score to flag transition

    # Adversarial probes
    adversarial_num_prompts: int = 20

    seed: int = 42
