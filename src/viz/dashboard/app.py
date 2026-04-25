"""Interactive Gradio dashboard for induction circuit visualisation.

Displays attention heatmaps, per-head induction score grids, and circuit
diagrams before and after fine-tuning. Runs on CPU via ONNX Runtime.

Run locally:
    python src/viz/dashboard/app.py
"""
from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

ONNX_DIR = Path("src/viz/dashboard/onnx_models")
PRE_PATH = ONNX_DIR / "model_pre.onnx"
POST_PATH = ONNX_DIR / "model_post.onnx"

_sessions: dict[str, Optional[ort.InferenceSession]] = {"pre": None, "post": None}
_tokenizer = None

EXPLAINER = """
**What am I seeing?**

Induction heads are attention heads that implement the copy-and-complete pattern:
if the model has seen A->B before, an induction head at the second A will attend
strongly to B, predicting B will come next (Olsson et al., 2022).

- **Attention heatmap**: which tokens each head attends to for your input text.
- **Induction score grid**: 0 = no induction, 1 = perfect induction per head.
- **Circuit diagram**: orange heads are causally verified circuit members (attribution >= 0.5).

Toggle Pre / Post fine-tuning to compare circuit state before and after code training.
""".strip()


def _load_session(state: str) -> Optional[ort.InferenceSession]:
    global _sessions
    if _sessions[state]:
        return _sessions[state]
    path = PRE_PATH if state == "pre" else POST_PATH
    if not path.exists():
        logger.warning("ONNX model missing at %s. Run onnx_export.py first.", path)
        return None
    t0 = time.perf_counter()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    logger.info("ONNX session loaded (%s) in %.2fs", state, time.perf_counter() - t0)
    _sessions[state] = session
    return session


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained("NeelNanda/gpt-neox-tokenizer-digits")
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


def analyse_text(input_text: str, model_state: str, selected_layer: int, selected_head: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state_key = "pre" if "Pre" in model_state else "post"
    if not input_text.strip():
        return None, None, None, "Please enter some text."

    tok = _get_tokenizer()
    ids = tok.encode(input_text, add_special_tokens=True)[:64]
    seq_len = len(ids)
    ids_padded = ids + [tok.pad_token_id] * (64 - seq_len)
    token_ids = np.array([ids_padded], dtype=np.int64)
    token_labels = [tok.decode([t]).replace("\n", "\u21b5") for t in ids]

    session = _load_session(state_key)
    if session is None:
        return None, None, None, f"ONNX model not available. Run onnx_export.py first."

    outputs = session.run(None, {"input_ids": token_ids})
    attn_patterns = [out[0] for out in outputs[:-1]]  # list of [n_heads, seq, seq]
    induction_scores = outputs[-1]  # [n_layers, n_heads]

    # Attention heatmap
    attn = attn_patterns[selected_layer][selected_head, :seq_len, :seq_len]
    fig_attn, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(attn, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(seq_len)); ax.set_xticklabels(token_labels, rotation=90, fontsize=8)
    ax.set_yticks(range(seq_len)); ax.set_yticklabels(token_labels, fontsize=8)
    ax.set_title(f"Attention: L{selected_layer}H{selected_head}", fontsize=11)
    plt.tight_layout()

    # Induction score grid
    n_layers, n_heads = induction_scores.shape
    fig_ind, ax2 = plt.subplots(figsize=(max(6, n_heads * 0.7), max(2, n_layers * 0.9)))
    im2 = ax2.imshow(induction_scores, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax2)
    ax2.set_xticks(range(n_heads)); ax2.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=9)
    ax2.set_yticks(range(n_layers)); ax2.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=9)
    for l in range(n_layers):
        for h in range(n_heads):
            ax2.text(h, l, f"{induction_scores[l,h]:.2f}", ha="center", va="center", fontsize=7)
    ax2.set_title("Per-head induction scores", fontsize=11)
    plt.tight_layout()

    # Circuit diagram
    circuit_heads = [(l, h) for l in range(n_layers) for h in range(n_heads) if induction_scores[l, h] >= 0.5]
    from src.viz.circuit_diagram import plot_circuit_diagram
    fig_circuit = plot_circuit_diagram(
        circuit_heads=circuit_heads, n_layers=n_layers, n_heads=n_heads,
        title=f"Active circuit heads ({model_state})",
    )

    status = (f"{model_state} | tokens={seq_len} | "
              f"circuit heads={len(circuit_heads)} | mean IS={induction_scores.mean():.3f}")
    return fig_attn, fig_ind, fig_circuit, status


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Induction Circuit Visualiser") as demo:
        gr.Markdown("# Induction Circuit Stability Under Fine-Tuning")
        gr.Markdown(EXPLAINER)
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(label="Input text", placeholder="Type or paste text here...", lines=3)
                model_state = gr.Radio(
                    choices=["Pre-fine-tuning", "Post-fine-tuning"],
                    value="Pre-fine-tuning", label="Model state",
                )
                with gr.Row():
                    sel_layer = gr.Slider(minimum=0, maximum=1, step=1, value=1, label="Layer")
                    sel_head = gr.Slider(minimum=0, maximum=7, step=1, value=0, label="Head")
                run_btn = gr.Button("Analyse", variant="primary")
                status_text = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            attn_plot = gr.Plot(label="Attention pattern")
            ind_plot = gr.Plot(label="Induction scores")
        with gr.Row():
            circuit_plot = gr.Plot(label="Circuit diagram")
        run_btn.click(
            fn=analyse_text,
            inputs=[input_text, model_state, sel_layer, sel_head],
            outputs=[attn_plot, ind_plot, circuit_plot, status_text],
        )
    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    build_interface().launch(server_name="0.0.0.0", server_port=7860, share=False)
