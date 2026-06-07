"""Tests for configuration dataclasses."""
from __future__ import annotations
import pytest
from src.model.config import EvalConfig, ModelConfig, TrainConfig


def test_model_config_d_model_consistency():
    config = ModelConfig(num_heads=8, d_head=32, d_model=256)
    assert config.d_model == config.num_heads * config.d_head


def test_model_config_raises_on_bad_d_model():
    with pytest.raises(ValueError, match="d_model"):
        ModelConfig(num_heads=8, d_head=32, d_model=128)


def test_train_config_total_steps():
    config = TrainConfig(max_tokens=50_000, batch_size=16, seq_length=128)
    assert config.total_steps == 50_000 // (16 * 128)


def test_train_config_warmup_steps():
    config = TrainConfig(max_tokens=50_000, batch_size=16, seq_length=128, warmup_fraction=0.1)
    assert config.warmup_steps == int(config.total_steps * 0.1)


def test_eval_config_defaults():
    config = EvalConfig()
    assert config.induction_seq_length > 0
    assert config.induction_num_sequences > 0
    assert 0 < config.induction_circuit_threshold < 1
