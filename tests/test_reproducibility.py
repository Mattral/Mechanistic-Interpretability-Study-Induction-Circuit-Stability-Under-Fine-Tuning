"""Reproducibility: determinism across runs and checkpoint round-trips."""
from __future__ import annotations
import tempfile
from pathlib import Path
import pytest
import torch
import transformer_lens


@pytest.fixture(scope="module")
def small_model():
    model = transformer_lens.HookedTransformer.from_pretrained(
        "attn-only-2l", center_unembed=True, center_writing_weights=True,
        fold_ln=True, refactor_factored_attn_matrices=True,
    )
    model.eval()
    return model


def test_set_global_seed_idempotent():
    from src.model.train import set_global_seed
    set_global_seed(42); a = torch.rand(10)
    set_global_seed(42); b = torch.rand(10)
    assert torch.allclose(a, b)


def test_checkpoint_roundtrip(small_model):
    from src.model.train import load_checkpoint, save_checkpoint, set_global_seed
    from src.model.config import TrainConfig
    set_global_seed(42)
    dummy_input = torch.randint(0, 100, (1, 16))
    with torch.no_grad():
        logits_before = small_model(dummy_input).clone()
    opt = torch.optim.AdamW(small_model.parameters(), lr=1e-4)
    dummy_scores = torch.zeros(small_model.cfg.n_layers, small_model.cfg.n_heads)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "ckpt.pt"
        save_checkpoint(ckpt, 0, small_model, opt, 0.0, dummy_scores, TrainConfig(), 42)
        with torch.no_grad():
            for p in small_model.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        load_checkpoint(ckpt, small_model)
        with torch.no_grad():
            logits_after = small_model(dummy_input)
    assert torch.allclose(logits_before, logits_after, atol=1e-5)


def test_checkpoint_required_metadata(small_model):
    from src.model.train import save_checkpoint
    from src.model.config import TrainConfig
    opt = torch.optim.AdamW(small_model.parameters(), lr=1e-4)
    dummy_scores = torch.zeros(small_model.cfg.n_layers, small_model.cfg.n_heads)
    required = {"step","model_state_dict","optimizer_state_dict","loss","induction_scores",
                "config","seed","transformer_lens_version","torch_version","timestamp"}
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "meta.pt"
        save_checkpoint(ckpt, 100, small_model, opt, 1.23, dummy_scores, TrainConfig(), 42)
        payload = torch.load(ckpt, map_location="cpu")
    assert not (required - set(payload.keys()))
