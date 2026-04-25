"""
Induction Circuit Stability Under Fine-Tuning -- Hugging Face Spaces app.
Self-contained Gradio dashboard. Uses Plotly for interactive attention heatmaps.
All other panels use Matplotlib. No external src/ package required.
Performance targets: cold-start < 3s on HF CPU Basic, inference < 500ms.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import plotly.graph_objects as go

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

ONNX_DIR = Path("onnx_models")
PRE_PATH = ONNX_DIR / "model_pre.onnx"
POST_PATH = ONNX_DIR / "model_post.onnx"
CIRCUIT_THRESHOLD = 0.5
MAX_SEQ_LEN = 64
LABEL_FONTSIZE = 9
TITLE_FONTSIZE = 11

_sessions: dict[str, Optional[ort.InferenceSession]] = {"pre": None, "post": None}
_tokenizer = None
_load_times: dict[str, float] = {}


def _load_session(state: str) -> tuple[Optional[ort.InferenceSession], str]:
    """Load or return cached ONNX InferenceSession."""
    global _sessions, _load_times
    if _sessions[state] is not None:
        return _sessions[state], ""
    path = PRE_PATH if state == "pre" else POST_PATH
    if not path.exists():
        msg = (
            f"ONNX model not found: {path}. "
            "Upload model_pre.onnx and model_post.onnx to the onnx_models/ directory. "
            "See HUGGINGFACE_SETUP.md in the GitHub repo."
        )
        logger.error(msg)
        return None, msg
    t0 = time.perf_counter()
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(path), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    elapsed = time.perf_counter() - t0
    _load_times[state] = elapsed
    logger.info("ONNX session loaded (%s) in %.3f s", state, elapsed)
    _sessions[state] = session
    return session, ""


def _get_tokenizer():
    """Lazy-load the correct tokenizer for attn-only-2l (DECISION-006).

    attn-only-2l uses NeelNanda/gpt-neox-tokenizer-digits, NOT gpt2.
    Using gpt2 feeds wrong token IDs to the ONNX model.
    """
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            "NeelNanda/gpt-neox-tokenizer-digits"
        )
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        logger.info("Tokenizer loaded: NeelNanda/gpt-neox-tokenizer-digits")
    return _tokenizer


def tokenise(text: str) -> tuple[np.ndarray, int, list[str]]:
    """Tokenise text, pad to MAX_SEQ_LEN. Returns (ids, seq_len, token_labels)."""
    tok = _get_tokenizer()
    ids = tok.encode(text, add_special_tokens=True)[:MAX_SEQ_LEN]
    seq_len = len(ids)
    ids_padded = ids + [tok.pad_token_id] * (MAX_SEQ_LEN - seq_len)
    labels = [tok.decode([t]).replace("\n", "\u21b5").replace(" ", "\u00b7") for t in ids]
    return np.array([ids_padded], dtype=np.int64), seq_len, labels


def run_inference(
    token_ids: np.ndarray, state: str
) -> tuple[Optional[list[np.ndarray]], Optional[np.ndarray], str]:
    """Run ONNX inference. Returns (attn_patterns, induction_scores, error_msg)."""
    session, err = _load_session(state)
    if session is None:
        return None, None, err
    t0 = time.perf_counter()
    outputs = session.run(None, {"input_ids": token_ids})
    logger.debug("Inference (%s): %.3f s", state, time.perf_counter() - t0)
    attn_patterns = [out[0] for out in outputs[:-1]]
    induction_scores: np.ndarray = outputs[-1]
    return attn_patterns, induction_scores, ""


# ---------------------------------------------------------------------------
# Plotly interactive attention heatmap (spec Section 7: "interactive, zoomable")
# ---------------------------------------------------------------------------

def build_attention_heatmap_plotly(
    attn: np.ndarray,
    labels: list[str],
    layer: int,
    head: int,
    model_state: str,
) -> go.Figure:
    """Interactive, zoomable attention heatmap (scroll to zoom, drag to pan).

    Satisfies spec Section 7: "Attention pattern heatmap for each head
    (interactive, zoomable)."
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=attn,
            x=labels,
            y=labels,
            colorscale="Viridis",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Attention weight", thickness=15),
            hovertemplate=(
                "Query: %{y}<br>Key: %{x}<br>Weight: %{z:.4f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Attention: L{layer}H{head} -- {model_state}",
            font=dict(size=13),
        ),
        xaxis=dict(
            title="Key position", tickfont=dict(size=9),
            tickangle=-45, automargin=True,
        ),
        yaxis=dict(
            title="Query position", tickfont=dict(size=9),
            autorange="reversed",
        ),
        height=460,
        margin=dict(l=80, r=40, t=60, b=100),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Matplotlib induction score grid
# ---------------------------------------------------------------------------

def build_induction_score_grid(
    induction_scores: np.ndarray,
    circuit_heads: list[tuple[int, int]],
    model_state: str,
) -> plt.Figure:
    """Per-head induction score heatmap with circuit heads outlined in yellow."""
    n_layers, n_heads = induction_scores.shape
    fig, ax = plt.subplots(figsize=(max(5, n_heads * 0.65), max(2.5, n_layers * 0.85)))
    im = ax.imshow(induction_scores, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Induction score", fraction=0.06)
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=LABEL_FONTSIZE)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{ll}" for ll in range(n_layers)], fontsize=LABEL_FONTSIZE)
    for ll in range(n_layers):
        for h in range(n_heads):
            v = induction_scores[ll, h]
            ax.text(h, ll, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if v < 0.5 else "black")
            if (ll, h) in circuit_heads:
                ax.add_patch(mpatches.Rectangle(
                    (h - 0.5, ll - 0.5), 1, 1,
                    linewidth=2.5, edgecolor="yellow", facecolor="none", zorder=5,
                ))
    ax.set_title(
        f"Per-head induction scores ({model_state})\n"
        f"Yellow = circuit head (IS >= {CIRCUIT_THRESHOLD})",
        fontsize=TITLE_FONTSIZE,
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Matplotlib circuit diagram
# ---------------------------------------------------------------------------

def build_circuit_diagram(
    circuit_heads: list[tuple[int, int]],
    n_layers: int,
    n_heads: int,
    model_state: str,
) -> plt.Figure:
    """Schematic circuit diagram showing previous-token -> induction head flow."""
    fig, ax = plt.subplots(figsize=(max(9, n_heads * 1.1), 5))
    ax.set_xlim(-0.7, n_heads - 0.3)
    ax.set_ylim(-0.8, n_layers + 0.8)
    ax.axis("off")
    ax.set_title(
        f"Induction circuit -- {model_state}\n"
        f"Orange = circuit member (attribution >= {CIRCUIT_THRESHOLD})",
        fontsize=TITLE_FONTSIZE, pad=12,
    )
    circuit_set = set(circuit_heads)
    layer_y = {ll: float(ll) for ll in range(n_layers)}

    for h in range(n_heads):
        ax.plot([float(h), float(h)],
                [layer_y[0] + 0.32, layer_y[n_layers - 1] - 0.32],
                color="#aaaaaa", linewidth=0.9, linestyle="--", zorder=1)

    for ll in range(n_layers):
        for h in range(n_heads):
            in_c = (ll, h) in circuit_set
            box = mpatches.FancyBboxPatch(
                (float(h) - 0.28, layer_y[ll] - 0.28), 0.56, 0.56,
                boxstyle="round,pad=0.05",
                facecolor="#FF8C00" if in_c else "#B0C4DE",
                edgecolor="#CC5500" if in_c else "#4682B4",
                linewidth=2.5 if in_c else 1.0, zorder=3,
            )
            ax.add_patch(box)
            ax.text(float(h), layer_y[ll], f"L{ll}H{h}",
                    ha="center", va="center", fontsize=7,
                    fontweight="bold" if in_c else "normal", zorder=4)

    for l0, h0 in circuit_set:
        for l1, h1 in circuit_set:
            if l1 <= l0:
                continue
            ax.annotate("", xy=(float(h1), layer_y[l1] - 0.32),
                        xytext=(float(h0), layer_y[l0] + 0.32),
                        arrowprops=dict(arrowstyle="->", color="#CC5500", lw=2.0),
                        zorder=2)

    for ll in range(n_layers):
        role = "Previous-token heads" if ll == 0 else "Induction heads"
        ax.text(-0.55, layer_y[ll], f"Layer {ll}\n({role})",
                ha="right", va="center", fontsize=LABEL_FONTSIZE - 1, fontweight="bold")

    ax.legend(handles=[
        mpatches.Patch(facecolor="#FF8C00", edgecolor="#CC5500",
                       label=f"Circuit head (attr >= {CIRCUIT_THRESHOLD})"),
        mpatches.Patch(facecolor="#B0C4DE", edgecolor="#4682B4",
                       label="Non-circuit head"),
    ], loc="upper right", fontsize=LABEL_FONTSIZE)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main inference pipeline
# ---------------------------------------------------------------------------

def analyse_text(
    input_text: str,
    model_state: str,
    selected_layer: int,
    selected_head: int,
) -> tuple:
    """Main Gradio callback: tokenise -> infer -> build 3 figures + status."""
    state_key = "pre" if "Pre" in model_state else "post"
    if not input_text.strip():
        return None, None, None, "Please enter some text."

    t0 = time.perf_counter()
    token_ids, seq_len, token_labels = tokenise(input_text)
    attn_patterns, induction_scores, err = run_inference(token_ids, state_key)
    if err:
        return None, None, None, f"Error: {err}"

    head_attn = attn_patterns[selected_layer][selected_head, :seq_len, :seq_len]
    circuit_heads = [
        (ll, h)
        for ll in range(induction_scores.shape[0])
        for h in range(induction_scores.shape[1])
        if induction_scores[ll, h] >= CIRCUIT_THRESHOLD
    ]
    n_layers = induction_scores.shape[0]
    n_heads = induction_scores.shape[1]

    fig_attn = build_attention_heatmap_plotly(
        head_attn, token_labels, selected_layer, selected_head, model_state
    )
    fig_ind = build_induction_score_grid(induction_scores, circuit_heads, model_state)
    fig_circ = build_circuit_diagram(circuit_heads, n_layers, n_heads, model_state)

    elapsed = time.perf_counter() - t0
    load_note = (f" (model load: {_load_times[state_key]:.2f}s)"
                 if state_key in _load_times else "")
    status = (
        f"{model_state} | tokens={seq_len} | circuit heads={len(circuit_heads)} | "
        f"mean IS={induction_scores.mean():.3f} | time: {elapsed:.2f}s{load_note}"
    )
    return fig_attn, fig_ind, fig_circ, status


# ---------------------------------------------------------------------------
# "What am I seeing?" -- exactly 3 sentences (spec Section 7)
# ---------------------------------------------------------------------------

WHAT_AM_I_SEEING = (
    "**What am I seeing?** "
    "Induction heads implement copy-and-complete: if the model has seen "
    "A->B before, an induction head at the second A attends to B and "
    "predicts it next -- this is the primary in-context learning mechanism "
    "in small transformers (Olsson et al., 2022). "
    "The **attention heatmap** (Plotly: scroll to zoom, drag to pan) shows which "
    "tokens each head attends to; the **induction score grid** rates each head "
    "0-1 with yellow borders on circuit members (attribution >= 0.5); and the "
    "**circuit diagram** shows information flow from previous-token heads "
    "(layer 0) to induction heads (layer 1) via the residual stream. "
    "Toggle Pre- vs Post-fine-tuning to see whether the circuit changes "
    "after Python code training, and check the status bar for circuit "
    "membership count and mean induction score."
)

EXAMPLE_INPUTS = [
    ["def fibonacci(n): return fibonacci(n-1) + fibonacci(n-2)", "Pre-fine-tuning", 1, 0],
    ["The cat sat on the mat. The cat", "Post-fine-tuning", 1, 0],
    ["import numpy as np\nx = np.array([1, 2, 3])\ny = np", "Post-fine-tuning", 1, 3],
    ["a b c d e a b c d e a", "Pre-fine-tuning", 1, 0],
]


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_interface() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""
    with gr.Blocks(
        title="Induction Circuit Visualiser",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
        css=".gradio-container { max-width: 1200px !important }",
    ) as demo:
        gr.Markdown("# Induction Circuit Stability Under Fine-Tuning")
        gr.Markdown(WHAT_AM_I_SEEING)
        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                input_text = gr.Textbox(
                    label="Input text",
                    placeholder="Type or paste text here...",
                    lines=4,
                    info="Up to 64 tokens. Try code, prose, or repetition (a b c a b c).",
                )
                model_state = gr.Radio(
                    choices=["Pre-fine-tuning", "Post-fine-tuning"],
                    value="Pre-fine-tuning",
                    label="Model state",
                    info="Pre = pretrained weights. Post = after Python code fine-tuning.",
                )
                with gr.Row():
                    sel_layer = gr.Slider(
                        minimum=0, maximum=1, step=1, value=1,
                        label="Layer (attention heatmap)",
                        info="0 = previous-token heads | 1 = induction heads",
                    )
                    sel_head = gr.Slider(
                        minimum=0, maximum=7, step=1, value=0,
                        label="Head",
                    )
                run_btn = gr.Button("Analyse", variant="primary")
                status_box = gr.Textbox(
                    label="Status", interactive=False, lines=2,
                )

            with gr.Column(scale=2):
                with gr.Tab("Attention Pattern (interactive)"):
                    attn_plot = gr.Plot(
                        label="Attention heatmap -- scroll to zoom, drag to pan"
                    )
                with gr.Tab("Induction Score Grid"):
                    ind_plot = gr.Plot(label="Per-head induction scores")
                with gr.Tab("Circuit Diagram"):
                    circuit_plot = gr.Plot(label="Circuit structure")

        gr.Examples(
            examples=EXAMPLE_INPUTS,
            inputs=[input_text, model_state, sel_layer, sel_head],
            label="Quick examples (click to run)",
            cache_examples=False,
        )

        gr.Markdown(
            "_Source: "
            "[Mattral/Mechanistic-Interpretability-Study-Induction-Circuit-Stability-Under-Fine-Tuning]"
            "(https://github.com/Mattral/Mechanistic-Interpretability-Study-Induction-Circuit-Stability-Under-Fine-Tuning)"
            "  |  Paper: [ArXiv TBD]_"
        )

        run_btn.click(
            fn=analyse_text,
            inputs=[input_text, model_state, sel_layer, sel_head],
            outputs=[attn_plot, ind_plot, circuit_plot, status_box],
        )
        input_text.submit(
            fn=analyse_text,
            inputs=[input_text, model_state, sel_layer, sel_head],
            outputs=[attn_plot, ind_plot, circuit_plot, status_box],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_error=True,
    )
