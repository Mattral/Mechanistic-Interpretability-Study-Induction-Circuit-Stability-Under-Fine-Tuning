"""Tests for the adversarial prompt suite."""
from __future__ import annotations
import pytest
import torch
from src.analysis.adversarial import MIN_PROMPT_PATTERNS, build_adversarial_suite


def test_suite_minimum_size():
    suite = build_adversarial_suite(vocab_size=1000, seq_len=10, batch_size=4, seed=42, device="cpu")
    assert len(suite) >= MIN_PROMPT_PATTERNS


def test_suite_has_required_keys():
    suite = build_adversarial_suite(vocab_size=1000, seq_len=10, batch_size=4, seed=42, device="cpu")
    assert "clean_repeated" in suite
    assert "fully_random" in suite


def test_suite_shapes():
    seq_len, batch = 10, 8
    suite = build_adversarial_suite(vocab_size=500, seq_len=seq_len, batch_size=batch, seed=42, device="cpu")
    for name, tokens in suite.items():
        assert tokens.shape == (batch, 2 * seq_len), f"{name}: expected {(batch, 2*seq_len)}, got {tokens.shape}"


def test_clean_repeated_structure():
    seq_len = 12
    suite = build_adversarial_suite(vocab_size=1000, seq_len=seq_len, batch_size=16, seed=42, device="cpu")
    clean = suite["clean_repeated"]
    assert torch.all(clean[:, :seq_len] == clean[:, seq_len:])


def test_sub_differs_at_position():
    seq_len = 12
    suite = build_adversarial_suite(vocab_size=1000, seq_len=seq_len, batch_size=32, seed=42, device="cpu")
    assert (suite["clean_repeated"][:, seq_len] != suite["sub_pos_0"][:, seq_len]).sum() > 0


def test_reproducibility():
    a = build_adversarial_suite(vocab_size=500, seq_len=10, batch_size=8, seed=7, device="cpu")
    b = build_adversarial_suite(vocab_size=500, seq_len=10, batch_size=8, seed=7, device="cpu")
    for name in a:
        assert torch.all(a[name] == b[name]), f"Not reproducible: {name}"
