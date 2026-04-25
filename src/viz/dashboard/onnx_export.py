"""ONNX export for CPU-based dashboard inference.

Usage:
    python src/viz/dashboard/onnx_export.py \
        --pre checkpoints/code_seed42/step_000000.pt \
        --post checkpoints/code_seed42/step_005000.pt
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import transformer_lens

from src.circuits.induction_score import compute_induction_score
from src.model.config import ModelConfig
# Note: ModelConfig.tokenizer defaults to NeelNanda/gpt-neox-tokenizer-digits (DECISION-006)
from src.model.train import load_checkpoint, load_pretrained_model, set_global_seed

logger = logging.getLogger(__name__)


class InductionCircuitONNX(nn.Module):
    """Wrapper for ONNX export: outputs attention patterns + pre-computed induction scores."""

    def __init__(self, model: transformer_lens.HookedTransformer, induction_scores: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("induction_scores_const", induction_scores)
        self.n_layers = model.cfg.n_layers

    def forward(self, input_ids: torch.Tensor):
        _, cache = self.model.run_with_cache(
            input_ids,
            names_filter=lambda n: "hook_attn" in n or "hook_pattern" in n,
            return_type=None,
        )
        attn_outputs = []
        for layer in range(self.n_layers):
            key = f"blocks.{layer}.attn.hook_attn"
            if key not in cache:
                key = f"blocks.{layer}.attn.hook_pattern"
            attn_outputs.append(cache[key])  # [batch, n_heads, seq, seq]
        return tuple(attn_outputs) + (self.induction_scores_const,)


def export_model_to_onnx(
    model: transformer_lens.HookedTransformer,
    output_path: Path,
    seq_len: int = 64,
    seed: int = 42,
    device: Optional[str] = None,
) -> None:
    """Export a HookedTransformer to ONNX with attention and induction score outputs."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    set_global_seed(seed)
    model = model.to(device)
    model.eval()

    induction_scores = compute_induction_score(
        model=model, sequence_length=50, num_sequences=200, seed=seed, device=device,
    )
    wrapper = InductionCircuitONNX(model=model, induction_scores=induction_scores)
    wrapper.eval()
    dummy = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    n_layers = model.cfg.n_layers
    out_names = [f"attn_layer_{l}" for l in range(n_layers)] + ["induction_scores"]

    logger.info("Exporting ONNX to %s ...", output_path)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (dummy,), str(output_path),
            input_names=["input_ids"], output_names=out_names,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                **{f"attn_layer_{l}": {0: "batch", 2: "seq_len", 3: "seq_len"} for l in range(n_layers)},
            },
            opset_version=17, do_constant_folding=True,
        )
    logger.info("ONNX export complete: %s", output_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser(description="Export models to ONNX for dashboard.")
    parser.add_argument("--pre", type=str, default=None)
    parser.add_argument("--post", type=str, default=None)
    parser.add_argument("--output", type=str, default="src/viz/dashboard/onnx_models/")
    parser.add_argument("--seq-len", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.output)
    mc = ModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.pre:
        m = load_pretrained_model(mc, device=device)
        load_checkpoint(Path(args.pre), m)
        export_model_to_onnx(m, out_dir / "model_pre.onnx", seq_len=args.seq_len, device=device)
    if args.post:
        m = load_pretrained_model(mc, device=device)
        load_checkpoint(Path(args.post), m)
        export_model_to_onnx(m, out_dir / "model_post.onnx", seq_len=args.seq_len, device=device)
    if not args.pre and not args.post:
        logger.warning("No checkpoint paths provided. Nothing exported.")


if __name__ == "__main__":
    main()
