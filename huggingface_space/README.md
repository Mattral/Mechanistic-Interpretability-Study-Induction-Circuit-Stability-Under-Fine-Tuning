---
title: Induction Circuit Stability Under Fine-Tuning
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.37.2
app_file: app.py
pinned: true
license: mit
short_description: Visualise how induction head circuits change under fine-tuning.
tags:
  - mechanistic-interpretability
  - transformers
  - circuits
  - induction-heads
  - fine-tuning
  - safety
---

# Induction Circuit Stability Under Fine-Tuning

Interactive dashboard accompanying the paper *Induction Circuit Stability Under
Fine-Tuning: A Mechanistic Interpretability Study*.

## What this does

Induction heads are attention heads that implement the copy-and-complete pattern:
if the model has seen A→B before, an induction head at the second A attends back
to B and predicts B will follow. This is the primary in-context learning mechanism
in small transformers (Olsson et al., 2022).

This dashboard lets you inspect:
- **Attention heatmap** — which tokens each head attends to on your input text.
- **Induction score grid** — per-head induction strength (0 = none, 1 = perfect).
- **Circuit diagram** — which heads are causally verified circuit members.

Toggle between **Pre-fine-tuning** and **Post-fine-tuning** states to see how the
circuit changes after training on Python code.

## Load time

The dashboard loads ONNX models on first run (pre-cached in this Space).
Typical cold-start: **< 3 seconds** on Hugging Face CPU Basic hardware.
Inference per input: **< 500 ms**.

## Links

- GitHub: [your-org/mech-interp-induction](https://github.com/your-org/mech-interp-induction)
- Paper: [ArXiv link TBD]
