# Mechanistic Interpretability Study: Induction Circuit Stability Under Fine-Tuning


[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/Mattral/induction-circuit-stability)


A rigorous mechanistic interpretability study tracking whether induction head circuits
in a 2-layer attention-only transformer survive fine-tuning on Python code, measuring
per-head induction scores and activation-patching attribution at every 100-step checkpoint
across three random seeds.

<div align="center">
  <a href="https://huggingface.co/spaces/Mattral/induction-circuit-stability" target="_blank">
    <img src="ICSUFT.png" alt="Induction Circuit Stability Dashboard" width="800">
  </a>
  <p>
    <b>Interactive Dashboard:</b> <a href="https://huggingface.co/spaces/Mattral/induction-circuit-stability">Try it live on Hugging Face Spaces</a> 
  </p>
</div>

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Mattral/Mechanistic-Interpretability-Study-Induction-Circuit-Stability-Under-Fine-Tuning
cd mech-interp-induction

# 2. Set up environment (Python 3.10)
conda env create -f environment.yml
conda activate mech-interp-induction

# 3. Run the full experiment pipeline (in order)
jupyter notebook notebooks/01_replication.ipynb    # ~5 min CPU  — baseline IS scores
jupyter notebook notebooks/02_patching.ipynb        # ~15 min CPU — causal verification
jupyter notebook notebooks/03_finetuning.ipynb      # ~80 min GPU — code + prose fine-tuning
jupyter notebook notebooks/04_adversarial.ipynb     # ~30 min CPU — sweep + phase + probes
jupyter notebook notebooks/05_figures.ipynb         # ~5 min CPU  — all paper figures

# 4. Run tests
pytest tests/ -v

# 5. Launch dashboard locally (after completing notebook 03)
python src/viz/dashboard/onnx_export.py \
    --pre  checkpoints/code_seed42/step_000000.pt \
    --post checkpoints/code_seed42/step_005000.pt
python src/viz/dashboard/app.py
# Open http://localhost:7860
```

---

## Repository Structure

```
mech-interp-induction/
├── PAPER_CHECKLIST.md          ← Track completion of all deliverables
├── src/
│   ├── model/
│   │   ├── config.py           ModelConfig, TrainConfig, EvalConfig (all hyperparams)
│   │   ├── train.py            Generic training loop + checkpoint utilities
│   │   └── finetune.py         Fine-tuning entry point (code + prose) — all 8 metrics
│   ├── circuits/
│   │   ├── induction_score.py  Canonical induction score (Olsson et al. 2022)
│   │   ├── patching.py         Activation patching + circuit attribution
│   │   ├── path_patching.py    Path patching for indirect effect decomposition
│   │   └── attribution.py      Direct logit attribution (DLA)
│   ├── analysis/
│   │   ├── checkpoint_sweep.py All 8 metrics across every checkpoint
│   │   ├── phase_detection.py  Savitzky-Golay smoothing + transition detection
│   │   └── adversarial.py      20+ adversarial probe types
│   └── viz/
│       ├── attention_vis.py    Publication figures (viridis/tab10, PDF+PNG)
│       ├── circuit_diagram.py  Circuit flow diagrams
│       └── dashboard/
│           ├── app.py          Local Gradio dashboard
│           └── onnx_export.py  Export to ONNX for CPU inference
├── huggingface_space/          ← Self-contained HF Spaces deployment
│   ├── app.py                  Standalone dashboard (no src/ imports)
│   ├── requirements.txt        Space-specific dependencies
│   ├── README.md               HF Space card
│   └── HUGGINGFACE_SETUP.md    Step-by-step deployment guide
├── notebooks/
│   ├── 01_replication.ipynb   Phase 1: replicate induction circuit
│   ├── 02_patching.ipynb      Phase 2: causal verification
│   ├── 03_finetuning.ipynb    Phase 3: fine-tuning (code + prose, 3 seeds each)
│   ├── 04_adversarial.ipynb   Phase 4: checkpoint sweep + phase detection + probes
│   └── 05_figures.ipynb       ★ Single source of truth for all paper figures
├── experiments/configs/        YAML configs (baseline, code, prose)
├── paper/                      LaTeX paper (NeurIPS single-column)
├── decisions/DECISION_LOG.md   Every non-obvious design decision
└── tests/                      Full pytest test suite (CPU, no GPU needed)
```

---

## Regenerating Excluded Files

The following directories are excluded from git (see `.gitignore`).
To regenerate them from scratch:

| Directory | How to regenerate |
|---|---|
| `checkpoints/` | Run `notebooks/03_finetuning.ipynb` (~80 min/run on GPU) |
| `experiments/results/` | Run notebooks 01–04 in order |
| `paper/figures/` | Run `notebooks/05_figures.ipynb` after 01–04 complete |
| `src/viz/dashboard/onnx_models/` | Run `src/viz/dashboard/onnx_export.py` |

Future: `dvc pull` will fetch all outputs once a DVC remote is configured.

---

## Experimental Design & Core Questions

Do structural circuits learned during foundational pre-training survive domain adaptation, or does the network overwrite them to accommodate highly structured syntax? We target three definitive research avenues:

1. **Circuit Degradation vs. Adaptation:** Do pre-existing induction heads degrade entirely, or do they transition into code-specific induction variants?
2. **Phase Change Dynamics:** At what precise training step do structural phase changes materialize within attention layers?
3. **Causal Mapping:** Can we mathematically map intermediate structural states during the fine-tuning trajectory?


### Stated-A-Priori Hypotheses

| ID | Hypothesis | Metric | Threshold |
|----|-----------|--------|-----------|
| H1 | Induction circuit degrades in first 20% of fine-tuning steps | IS drop | ≥ 0.1 |
| H2 | Code-specific induction variant forms in later steps | IS rise on code patterns | ≥ 0.1 |
| H3 | Prose fine-tuning causes less degradation than code | ΔIS(code) vs ΔIS(prose) | p < 0.05 |

### Induction Score Definition

For repeated-token sequence $[t_1,...,t_n, t_1,...,t_n]$ (Olsson et al. 2022):

$$\text{IS}(l,h) = \frac{1}{n} \sum_{i=1}^{n} A^{(l,h)}[n+i,\; i]$$

Averaged over 500 sequences, half-length 50. **Change ≥ 0.1 = practically meaningful (a priori).**

### All 8 Checkpoint Metrics (every 100 steps)

| # | Metric | Type | Description |
|---|--------|------|-------------|
| 1 | `induction_score` | [L, H] | Per-head IS (canonical definition) |
| 2 | `induction_score_delta` | [L, H] | Change from step-0 baseline |
| 3 | `train_loss` | scalar | Cross-entropy on training batch |
| 4 | `induction_task_loss` | scalar | CE on held-out induction sequences |
| 5 | `logit_diff_clean` | scalar | Logit diff on clean induction task |
| 6 | `logit_diff_corrupted` | scalar | Logit diff on corrupted baseline |
| 7 | `code_icl_score` | scalar | ICL accuracy on code copy patterns |
| 8 | `circuit_attribution` | [L, H] | Activation-patching attribution |

### Stated-A-Priori Thresholds

| Threshold | Value | Decision |
|---|---|---|
| Meaningful IS change | ≥ 0.1 | DECISION-004 |
| Circuit membership | ≥ 0.5 | DECISION-002 |
| Dashboard load time | < 3 s | Phase 5 checklist |

---

## Running Tests

```bash
pytest tests/ -v
# All tests run on CPU. Downloads ~25 MB pretrained model on first run.
```

Test coverage:

| File | What is tested |
|------|---------------|
| `test_config.py` | ModelConfig validation, TrainConfig properties |
| `test_induction_score.py` | Shape, range, known IS heads, reproducibility, non-IS heads |
| `test_patching.py` | Sequence construction, attribution shape, clean/corrupted recovery |
| `test_reproducibility.py` | Seed determinism, checkpoint round-trip, training determinism |
| `test_phase_detection.py` | Smoothing, transition detection, dissolution detection |
| `test_adversarial.py` | Suite size, shapes, structure, reproducibility |

---

## Code Standards

- **Formatting:** black (line-length=88), ruff, isort
- **Types:** mypy --strict on all src/ modules
- **Logging:** `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s` (ISO 8601 timestamps)
- **Docs:** All public functions have Google-style docstrings

---

## References

- Elhage et al. (2021). *A Mathematical Framework for Transformer Circuits.*
- Olsson et al. (2022). *In-Context Learning and Induction Heads.*
- Wang et al. (2022). *Interpretability in the Wild.*
- Conmy et al. (2023). *Towards Automated Circuit Discovery.*
- Marks et al. (2024). *Sparse Feature Circuits.*
- Nanda & Bloom (2022). *TransformerLens.*

---

### Critical Bugs Fixed During Development

Two implementation bugs were discovered and fixed through live experimentation:

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| [DECISION-005](decisions/DECISION_LOG.md) | TransformerLens prepends BOS token by default, offsetting all attention positions by +1 | Induction scores near zero (~0.04) despite correct circuit attribution | `prepend_bos=False` in all `run_with_cache()` calls |
| [DECISION-006](decisions/DECISION_LOG.md) | `attn-only-2l` uses `NeelNanda/gpt-neox-tokenizer-digits`, not `gpt2` | Garbled attention patterns in dashboard | All tokenizer loads updated to `NeelNanda/gpt-neox-tokenizer-digits` |

Both bugs were identified from live experimental output (notebook runs + HF Space testing).
See `decisions/DECISION_LOG.md` for full evidence trail.

---

## Commit Standards (Section 10)

Format: `<type>(<scope>): <subject>` — no exceptions.

| Field | Values |
|-------|--------|
| `type` | `feat` \| `fix` \| `exp` \| `docs` \| `test` \| `refactor` \| `deps` \| `chore` |
| `scope` | `model` \| `circuits` \| `analysis` \| `viz` \| `paper` \| `notebooks` \| `ci` |
| `subject` | Imperative mood, ≤ 72 chars, no full stop |

**Examples:**
```
feat(circuits): add path patching for indirect effect decomposition
exp(model): add checkpoint sweep across all fine-tuning runs
fix(induction_score): correct off-by-one in repeated sequence construction
docs(paper): draft results section with Fig 5 and Fig 6 references
test(patching): add clean-run full-recovery assertion
deps: update gradio to 4.37.2
```

**Rules:**
- Every commit must leave the codebase in a runnable state.
- No broken imports, no half-finished refactors in `src/`.
- Scratch work goes in `notebooks/`, never in `src/`.
- No direct commits to `main`. All changes via branch + self-review.

