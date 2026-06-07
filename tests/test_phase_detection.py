"""Tests for phase transition detection utilities."""
from __future__ import annotations
import numpy as np
import pytest
from src.analysis.phase_detection import (
    detect_circuit_dissolution_step,
    detect_phase_transitions,
    smooth_curve,
    summarise_transitions,
)


def test_smooth_curve_same_length():
    x = np.random.rand(30)
    assert len(smooth_curve(x, window=5)) == len(x)


def test_smooth_constant():
    x = np.ones(20) * 0.5
    np.testing.assert_allclose(smooth_curve(x, window=5), x, atol=1e-6)


def test_detects_step_change():
    steps = np.arange(0, 2000, 100)
    scores = np.zeros((20, 1, 1))
    scores[10:, 0, 0] = -0.5
    transitions = detect_phase_transitions(steps, scores, threshold=0.1, window=3)
    assert len(transitions) >= 1
    assert any(t["layer"] == 0 and t["head"] == 0 for t in transitions)


def test_no_transition_flat_curve():
    steps = np.arange(0, 1000, 100)
    scores = np.ones((10, 2, 4)) * 0.7
    assert detect_phase_transitions(steps, scores, threshold=0.1, window=3) == []


def test_summarise_empty():
    s = summarise_transitions([])
    assert s["n_degradations"] == 0 and s["earliest_step"] is None


def test_summarise_counts():
    ts = [
        {"layer": 0, "head": 0, "step": 50, "step_fraction": 0.05,
         "score_before": 0.9, "score_after": 0.7, "delta": -0.2},
        {"layer": 1, "head": 1, "step": 200, "step_fraction": 0.2,
         "score_before": 0.3, "score_after": 0.5, "delta": 0.2},
    ]
    s = summarise_transitions(sorted(ts, key=lambda x: x["step"]))
    assert s["n_degradations"] == 1 and s["n_improvements"] == 1
    assert s["earliest_step"] == 50


def test_dissolution_none_if_stable():
    steps = np.arange(0, 1000, 100)
    attr = np.ones((10, 2, 4)) * 0.8
    assert detect_circuit_dissolution_step(steps, attr, threshold=0.5) is None


def test_dissolution_finds_step():
    steps = np.arange(0, 1000, 100)
    attr = np.ones((10, 2, 4)) * 0.8
    attr[5:, 1, 2] = 0.3
    assert detect_circuit_dissolution_step(steps, attr, threshold=0.5) == 500
