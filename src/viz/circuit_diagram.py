"""Circuit flow diagram generation (Matplotlib, no external graph layout)."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
LABEL_FONTSIZE = 10
TITLE_FONTSIZE = 12


def _save(fig, path, dpi=300):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
    logger.info("Saved: %s", path)


def _arrow(ax, x0, y0, x1, y1, **kw):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                 arrowprops=dict(arrowstyle="->", lw=1.5, **kw))


def plot_circuit_diagram(
    circuit_heads: list[tuple[int, int]],
    path_scores: Optional[np.ndarray] = None,
    n_layers: int = 2,
    n_heads: int = 8,
    title: str = "Induction circuit",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Draw induction circuit schematic (Figure 1).

    Circuit heads shown in orange; connections scaled by path score if provided.
    """
    fig, ax = plt.subplots(figsize=(max(10, n_heads * 1.2), 5))
    ax.set_xlim(-0.5, n_heads - 0.5)
    ax.set_ylim(-0.5, n_layers + 0.5)
    ax.axis("off")
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=16)

    circuit_set = set(circuit_heads)
    layer_y = {l: float(l) for l in range(n_layers)}

    for layer in range(n_layers):
        y = layer_y[layer]
        for head in range(n_heads):
            in_c = (layer, head) in circuit_set
            circle = mpatches.Circle(
                (float(head), y), radius=0.3,
                facecolor="orange" if in_c else "lightsteelblue",
                edgecolor="darkorange" if in_c else "steelblue",
                linewidth=2.5 if in_c else 1.0, zorder=3,
            )
            ax.add_patch(circle)
            ax.text(float(head), y, f"L{layer}H{head}",
                    ha="center", va="center", fontsize=7, zorder=4)

    # Residual stream lines
    for head in range(n_heads):
        ax.plot([float(head), float(head)],
                [layer_y[0] + 0.3, layer_y[n_layers - 1] - 0.3],
                color="grey", linewidth=0.8, linestyle="--", zorder=1)

    # Arrows between circuit heads
    for l0, h0 in circuit_set:
        for l1, h1 in circuit_set:
            if l1 <= l0:
                continue
            weight = 1.0
            if path_scores is not None and hasattr(path_scores, "ndim") and path_scores.ndim == 4:
                weight = float(np.clip(path_scores[l0, h0, l1, h1], 0, 1))
            if weight < 0.1:
                continue
            _arrow(ax, float(h0), layer_y[l0] + 0.3, float(h1), layer_y[l1] - 0.3,
                   color="darkorange", lw=0.8 + weight * 2.0)

    for layer in range(n_layers):
        ax.text(-0.4, layer_y[layer], f"Layer {layer}",
                ha="right", va="center", fontsize=LABEL_FONTSIZE, fontweight="bold")

    ax.legend(handles=[
        mpatches.Patch(facecolor="orange", edgecolor="darkorange", label="Circuit head (attr >= 0.5)"),
        mpatches.Patch(facecolor="lightsteelblue", edgecolor="steelblue", label="Non-circuit head"),
    ], loc="upper right", fontsize=LABEL_FONTSIZE)
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_attribution_heatmap(
    attribution_scores: np.ndarray,
    threshold: float = 0.5,
    title: str = "Activation patching attribution scores",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Per-head attribution score heatmap (Figure 4). Circuit heads outlined with thick border."""
    n_layers, n_heads = attribution_scores.shape
    fig, ax = plt.subplots(figsize=(max(6, n_heads * 0.7), max(3, n_layers * 0.9)))
    im = ax.imshow(attribution_scores, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Attribution score")
    ax.set_xticks(range(n_heads)); ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=LABEL_FONTSIZE)
    ax.set_yticks(range(n_layers)); ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=LABEL_FONTSIZE)
    ax.set_title(f"{title}\n(circuit threshold = {threshold})", fontsize=TITLE_FONTSIZE)
    for l in range(n_layers):
        for h in range(n_heads):
            v = attribution_scores[l, h]
            ax.text(h, l, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if 0.2 < v < 0.8 else "white")
            if v >= threshold:
                ax.add_patch(mpatches.Rectangle(
                    (h - 0.5, l - 0.5), 1, 1,
                    linewidth=2.5, edgecolor="black", facecolor="none", zorder=5,
                ))
    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig
