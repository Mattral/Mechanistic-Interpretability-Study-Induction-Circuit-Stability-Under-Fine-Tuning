"""
Induction Circuit Stability Under Fine-Tuning — Hugging Face Spaces app.

Self-contained Gradio dashboard. No external src/ package required.
All logic is in this single file for Space compatibility.

Performance targets (Section 7 of AGENT_INSTRUCTIONS):
  - Cold-start load: < 3 s on HF CPU Basic
  - Inference per input: < 500 ms
  - No GPU required (ONNX CPU Runtime)
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

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Logging (project-standard format)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ONNX_DIR = Path("onnx_models")
PRE_PATH = ONNX_DIR / "model_pre.onnx"
POST_PATH = ONNX_DIR / "model_post.onnx"
CIRCUIT_THRESHOLD = 0.5
MAX_SEQ_LEN = 64
LABEL_FONTSIZE = 9
TITLE_FONTSIZE = 11

# ---------------------------------------------------------------------------
# Session cache (avoid re-loading ONNX on each call)
# ---------------------------------------------------------------------------
_sessions: dict[str, Optional[ort.InferenceSession]] = {"pre": None, "post": None}
_tokenizer = None
_load_times: dict[str, float] = {}


def _load_session(state: str) -> tuple[Optional[ort.InferenceSession], str]:
    """Load (or return cached) ONNX session. Returns (session, status_message)."""
    global _sessions, _load_times
    if _sessions[state] is not None:
        return _sessions[state], ""

    path = PRE_PATH if state == "pre" else POST_PATH
    if not path.exists():
        msg = (
            f"ONNX model not found: {path}. "
            "Upload model_pre.onnx and model_post.onnx to the onnx_models/ directory "
            "of this Space (see HUGGINGFACE_SETUP.md in the GitHub repo)."
        )
        logger.error(msg)
        return None, msg

    t0 = time.perf_counter()
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    elapsed = time.perf_counter() - t0
    _load_times[state] = elapsed
    logger.info("ONNX session loaded (%s) in %.3f s", state, elapsed)
    _sessions[state] = session
    return session, ""


def _get_tokenizer():
    """Lazy-load GPT-2 tokenizer (cached after first call)."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        logger.info("GPT-2 tokenizer loaded.")
    return _tokenizer


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
def tokenise(text: str) -> tuple[np.ndarray, int, list[str]]:
    """Tokenise text, pad to MAX_SEQ_LEN, return (ids_padded, seq_len, labels)."""
    tok = _get_tokenizer()
    ids = tok.encode(text, add_special_tokens=True)[:MAX_SEQ_LEN]
    seq_len = len(ids)
    ids_padded = ids + [tok.pad_token_id] * (MAX_SEQ_LEN - seq_len)
    labels = [tok.decode([t]).replace("\n", "↵").replace(" ", "·") for t in ids]
    return np.array([ids_padded], dtype=np.int64), seq_len, labels


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(
    token_ids: np.ndarray, state: str
) -> tuple[Optional[list[np.ndarray]], Optional[np.ndarray], str]:
    """Run ONNX inference; returns (attn_patterns, induction_scores, error_msg)."""
    session, err = _load_session(state)
    if session is None:
        return None, None, err
    t0 = time.perf_counter()
    outputs = session.run(None, {"input_ids": token_ids})
    elapsed = time.perf_counter() - t0
    logger.debug("Inference (%s): %.3f s", state, elapsed)
    # outputs: [attn_L0, attn_L1, ..., induction_scores]
    attn_patterns = [out[0] for out in outputs[:-1]]  # each: [n_heads, seq, seq]
    induction_scores = outputs[-1]                      # [n_layers, n_heads]
    return attn_patterns, induction_scores, ""


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
def _save_and_return(fig: plt.Figure) -> plt.Figure:
    fig.patch.set_facecolor("none")
    plt.tight_layout()
    return fig


def build_attention_heatmap(
    attn: np.ndarray,
    labels: list[str],
    layer: int,
    head: int,
) -> plt.Figure:
    """Attention heatmap [seq_len x seq_len] with token labels."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(5, n * 0.38), max(4, n * 0.38)))
    im = ax.imshow(attn, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Attention weight", fraction=0.04)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=75, fontsize=LABEL_FONTSIZE - 1, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=LABEL_FONTSIZE - 1)
    ax.set_xlabel("Key position", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Query position", fontsize=LABEL_FONTSIZE)
    ax.set_title(f"Attention: Layer {layer}, Head {head}", fontsize=TITLE_FONTSIZE)
    return _save_and_return(fig)


def build_induction_score_grid(
    induction_scores: np.ndarray,
    circuit_heads: list[tuple[int, int]],
    model_state: str,
) -> plt.Figure:
    """Per-head induction score heatmap with circuit heads outlined."""
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
        f"Yellow border = circuit head (IS ≥ {CIRCUIT_THRESHOLD})",
        fontsize=TITLE_FONTSIZE,
    )
    return _save_and_return(fig)


def build_circuit_diagram(
    circuit_heads: list[tuple[int, int]],
    n_layers: int,
    n_heads: int,
    model_state: str,
) -> plt.Figure:
    """Schematic circuit diagram with previous-token and induction heads."""
    fig, ax = plt.subplots(figsize=(max(9, n_heads * 1.1), 5))
    ax.set_xlim(-0.7, n_heads - 0.3)
    ax.set_ylim(-0.8, n_layers + 0.8)
    ax.axis("off")
    ax.set_title(
        f"Induction circuit — {model_state}\n"
        f"Orange = circuit member (attribution ≥ {CIRCUIT_THRESHOLD})",
        fontsize=TITLE_FONTSIZE, pad=12,
    )

    circuit_set = set(circuit_heads)
    layer_y = {ll: float(ll) for ll in range(n_layers)}

    # Residual stream lines
    for h in range(n_heads):
        ax.plot(
            [float(h), float(h)],
            [layer_y[0] + 0.32, layer_y[n_layers - 1] - 0.32],
            color="#aaaaaa", linewidth=0.9, linestyle="--", zorder=1,
        )

    # Head circles
    for ll in range(n_layers):
        for h in range(n_heads):
            in_c = (ll, h) in circuit_set
            fc = "#FF8C00" if in_c else "#B0C4DE"
            ec = "#CC5500" if in_c else "#4682B4"
            circle = mpatches.FancyBboxPatch(
                (float(h) - 0.28, layer_y[ll] - 0.28), 0.56, 0.56,
                boxstyle="round,pad=0.05",
                facecolor=fc, edgecolor=ec,
                linewidth=2.5 if in_c else 1.0, zorder=3,
            )
            ax.add_patch(circle)
            ax.text(
                float(h), layer_y[ll], f"L{ll}H{h}",
                ha="center", va="center", fontsize=7,
                fontweight="bold" if in_c else "normal", zorder=4,
            )

    # Arrows between circuit heads (layer 0 -> layer 1+)
    for l0, h0 in circuit_set:
        for l1, h1 in circuit_set:
            if l1 <= l0:
                continue
            ax.annotate(
                "", xy=(float(h1), layer_y[l1] - 0.32),
                xytext=(float(h0), layer_y[l0] + 0.32),
                arrowprops=dict(arrowstyle="->", color="#CC5500", lw=2.0),
                zorder=2,
            )

    # Layer labels
    for ll in range(n_layers):
        role = "Previous-token heads" if ll == 0 else "Induction heads"
        ax.text(
            -0.55, layer_y[ll], f"Layer {ll}\n({role})",
            ha="right", va="center", fontsize=LABEL_FONTSIZE - 1,
            fontweight="bold",
        )

    ax.legend(
        handles=[
            mpatches.Patch(facecolor="#FF8C00", edgecolor="#CC5500",
                           label=f"Circuit head (attr ≥ {CIRCUIT_THRESHOLD})"),
            mpatches.Patch(facecolor="#B0C4DE", edgecolor="#4682B4",
                           label="Non-circuit head"),
        ],
        loc="upper right", fontsize=LABEL_FONTSIZE,
    )
    return _save_and_return(fig)


# ---------------------------------------------------------------------------
# Main inference + figure pipeline
# ---------------------------------------------------------------------------
def analyse_text(
    input_text: str,
    model_state: str,
    selected_layer: int,
    selected_head: int,
) -> tuple:
    """Main Gradio callback: tokenise → infer → build 3 figures + status."""
    state_key = "pre" if "Pre" in model_state else "post"

    if not input_text.strip():
        return None, None, None, "⚠️  Please enter some text."

    t0 = time.perf_counter()

    # Tokenise
    token_ids, seq_len, token_labels = tokenise(input_text)

    # Inference
    attn_patterns, induction_scores, err = run_inference(token_ids, state_key)
    if err:
        return None, None, None, f"❌ {err}"

    # Clip to actual sequence length
    head_attn = attn_patterns[selected_layer][selected_head, :seq_len, :seq_len]
    circuit_heads = [
        (ll, h)
        for ll in range(induction_scores.shape[0])
        for h in range(induction_scores.shape[1])
        if induction_scores[ll, h] >= CIRCUIT_THRESHOLD
    ]
    n_layers = induction_scores.shape[0]
    n_heads = induction_scores.shape[1]

    # Build figures
    fig_attn = build_attention_heatmap(head_attn, token_labels, selected_layer, selected_head)
    fig_ind = build_induction_score_grid(induction_scores, circuit_heads, model_state)
    fig_circ = build_circuit_diagram(circuit_heads, n_layers, n_heads, model_state)

    elapsed = time.perf_counter() - t0
    load_note = ""
    if state_key in _load_times:
        load_note = f" (model load: {_load_times[state_key]:.2f}s)"

    status = (
        f"✅ {model_state} | tokens={seq_len} | "
        f"circuit heads={len(circuit_heads)} | "
        f"mean IS={induction_scores.mean():.3f} | "
        f"inference: {elapsed:.2f}s{load_note}"
    )
    return fig_attn, fig_ind, fig_circ, status


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
EXPLAINER = """
### What am I seeing?

**Induction heads** implement the copy-and-complete pattern: if the model has seen
the sequence A → B earlier, an induction head at the second occurrence of A attends
back to B and predicts it will follow. This is the primary in-context learning
mechanism in small transformers (Olsson et al., 2022).

This dashboard visualises how the induction circuit changes after fine-tuning on
Python code.

| Panel | What it shows |
|-------|---------------|
| **Attention heatmap** | Attention weights for the selected head on your input text |
| **Induction score grid** | Per-head IS: 0 = no induction, 1 = perfect induction. Yellow border = circuit member |
| **Circuit diagram** | Orange nodes = causally verified circuit members (attribution ≥ 0.5) |

**Toggle Pre / Post fine-tuning** to compare circuit state before and after code training.
""".strip()

EXAMPLE_INPUTS = [
    ["def fibonacci(n): return fibonacci(n-1) + fibonacci(n-2)", "Pre-fine-tuning", 1, 0],
    ["The cat sat on the mat. The cat", "Post-fine-tuning", 1, 0],
    ["import numpy as np\nx = np.array([1, 2, 3])\ny = np", "Post-fine-tuning", 1, 3],
]


def build_interface() -> gr.Blocks:
    """Construct the Gradio Blocks interface."""
    with gr.Blocks(
        title="Induction Circuit Visualiser",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
        css=".gradio-container { max-width: 1200px !important }",
    ) as demo:
        gr.Markdown("# 🔬 Induction Circuit Stability Under Fine-Tuning")
        gr.Markdown(EXPLAINER)

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                input_text = gr.Textbox(
                    label="Input text",
                    placeholder="Type or paste text here…",
                    lines=4,
                    info="Any text up to 64 tokens. Try code, prose, or repeated patterns.",
                )
                model_state = gr.Radio(
                    choices=["Pre-fine-tuning", "Post-fine-tuning"],
                    value="Pre-fine-tuning",
                    label="Model state",
                    info="Compare circuit structure before vs. after Python code fine-tuning.",
                )
                with gr.Row():
                    sel_layer = gr.Slider(
                        minimum=0, maximum=1, step=1, value=1,
                        label="Layer (attention heatmap)",
                        info="Layer 0 = previous-token heads; Layer 1 = induction heads",
                    )
                    sel_head = gr.Slider(
                        minimum=0, maximum=7, step=1, value=0,
                        label="Head (attention heatmap)",
                    )
                run_btn = gr.Button("🔍 Analyse", variant="primary", scale=1)
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2,
                )

            with gr.Column(scale=2):
                with gr.Tab("Attention Pattern"):
                    attn_plot = gr.Plot(label="Attention heatmap")
                with gr.Tab("Induction Scores"):
                    ind_plot = gr.Plot(label="Per-head induction scores")
                with gr.Tab("Circuit Diagram"):
                    circuit_plot = gr.Plot(label="Circuit structure")

        gr.Examples(
            examples=EXAMPLE_INPUTS,
            inputs=[input_text, model_state, sel_layer, sel_head],
            label="Quick examples",
        )

        gr.Markdown(
            "_Source code: [github.com/your-org/mech-interp-induction](https://github.com/your-org/mech-interp-induction) | "
            "Paper: [ArXiv TBD]_"
        )

        run_btn.click(
            fn=analyse_text,
            inputs=[input_text, model_state, sel_layer, sel_head],
            outputs=[attn_plot, ind_plot, circuit_plot, status_text],
        )
        # Also trigger on Enter in text box
        input_text.submit(
            fn=analyse_text,
            inputs=[input_text, model_state, sel_layer, sel_head],
            outputs=[attn_plot, ind_plot, circuit_plot, status_text],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_error=True,
    )
