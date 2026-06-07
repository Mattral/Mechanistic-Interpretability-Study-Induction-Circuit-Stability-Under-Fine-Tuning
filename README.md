# Mechanistic Interpretability Study: Induction Circuit Stability Under Fine-Tuning

[![CI](https://github.com/Mattral/Mechanistic-Interpretability-Study-Induction-Circuit-Stability-Under-Fine-Tuning/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattral/Mechanistic-Interpretability-Study-Induction-Circuit-Stability-Under-Fine-Tuning/actions)

A rigorous mechanistic interpretability study tracking whether induction head
circuits in a 2-layer attention-only transformer survive fine-tuning on Python
code, with measurement of the exact training steps at which structural changes occur.

---

## Quick Start

```bash
# 1. Set up environment
conda env create -f environment.yml
conda activate mech-interp-induction

# 2. Run pipeline (in order)
jupyter notebook notebooks/01_replication.ipynb    # ~5 min CPU
jupyter notebook notebooks/02_patching.ipynb        # ~15 min CPU
jupyter notebook notebooks/03_finetune.ipynb        # ~80 min GPU
jupyter notebook notebooks/04_analysis.ipynb        # ~30 min CPU
jupyter notebook notebooks/05_figures.ipynb         # ~5 min CPU

# 3. Launch dashboard
python src/viz/dashboard/onnx_export.py \
    --pre checkpoints/code_seed42/step_000000.pt \
    --post checkpoints/code_seed42/step_005000.pt
python src/viz/dashboard/app.py
```

---

## Repository Structure

```
mech-interp-induction/
├── src/
│   ├── model/
│   │   ├── config.py           # ModelConfig, TrainConfig, EvalConfig
│   │   ├── train.py            # Generic training loop + checkpoint utils
│   │   └── finetune.py         # Fine-tuning entry point (code + prose)
│   ├── circuits/
│   │   ├── induction_score.py  # Canonical induction score (Olsson et al.)
│   │   ├── patching.py         # Activation patching + circuit attribution
│   │   ├── path_patching.py    # Path patching for indirect effects
│   │   └── attribution.py      # Direct logit attribution (DLA)
│   ├── analysis/
│   │   ├── checkpoint_sweep.py # Sweep checkpoints, compute all metrics
│   │   ├── phase_detection.py  # Phase transition detection
│   │   └── adversarial.py      # 20+ adversarial probe types
│   └── viz/
│       ├── attention_vis.py    # Publication-quality static figures
│       ├── circuit_diagram.py  # Circuit flow diagrams
│       └── dashboard/
│           ├── app.py          # Gradio interactive dashboard
│           └── onnx_export.py  # Export models to ONNX for CPU inference
├── notebooks/
│   ├── 01_replication.ipynb   # Phase 1: replicate induction circuit
│   ├── 02_patching.ipynb      # Phase 2: causal verification
│   ├── 03_finetune.ipynb      # Phase 3: fine-tuning (code + prose)
│   ├── 04_analysis.ipynb      # Phase 4: sweep + phase detection + adversarial
│   └── 05_figures.ipynb       # Single source of truth for all paper figures
├── experiments/configs/        # YAML experiment configs
├── paper/                      # LaTeX paper
├── decisions/DECISION_LOG.md   # Design decision log
└── tests/                      # Full pytest test suite
```

---

## Experimental Design

| Hypothesis | Description | Test |
|---|---|---|
| H1 | Circuit degrades in first 20% of steps | Step where IS < baseline − 0.1 |
| H2 | Code-specific variant forms later | IS increases on code patterns |
| H3 | Prose causes less degradation than code | IS(code) vs IS(prose) |

**Induction score** (Olsson et al. 2022): For repeated-token sequence [t₁…tₙ t₁…tₙ]:
IS(l,h) = (1/n) Σᵢ A^(l,h)[n+i, i]. Change ≥ 0.1 = practically meaningful (a priori).

**Attribution threshold**: ≥ 0.5 for circuit membership (stated a priori, raw scores in all figures).

**Seeds**: 42, 123, 7. Results = mean ± 1 SD.

---

## Running Tests

```bash
pytest tests/ -v
```

Tests run on CPU, no GPU needed. Downloads ~25 MB pretrained model on first run.

---

## References

- Elhage et al. (2021). *A Mathematical Framework for Transformer Circuits.*
- Olsson et al. (2022). *In-Context Learning and Induction Heads.*
- Wang et al. (2022). *Interpretability in the Wild.*
- Conmy et al. (2023). *Towards Automated Circuit Discovery.*
- Nanda & Bloom (2022). *TransformerLens.*
