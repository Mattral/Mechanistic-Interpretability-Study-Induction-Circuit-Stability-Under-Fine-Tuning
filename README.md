# Mechanistic Interpretability Study: Induction Circuit Stability Under Fine-Tuning


[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/YOUR_USERNAME/induction-circuit-stability)


> **TL;DR:** We are tracking exactly what happens to the internal "circuitry" (induction heads) of a 2-layer attention-only Transformer when forced to undergo domain adaptation from prose to structured Python code.

A rigorous mechanistic interpretability study tracking whether induction head circuits
in a 2-layer attention-only transformer survive fine-tuning on Python code, measuring
per-head induction scores and activation-patching attribution at every 100-step checkpoint
across three random seeds.

*If you are interested in AI safety, mechanistic alignment, or tracking phase changes in neural networks under the hood, this project is for you.*

<div align="center">
  <a href="https://huggingface.co/spaces/Mattral/induction-circuit-stability" target="_blank">
    <img src="ICSUFT.png" alt="Induction Circuit Stability Dashboard" width="800">
  </a>
  <p>
    <b>Interactive Dashboard:</b> <a href="https://huggingface.co/spaces/Mattral/induction-circuit-stability">Try it live on Hugging Face Spaces</a> 
  </p>
</div>

---


## 🚀 Quickstart

Ensure you have a local environment running Python 3.10.

### 1. Clone & Environment Set Up

```bash
# Clone the repository
git clone https://github.com/Mattral/Mechanistic-Interpretability-Study-Induction-Circuit-Stability-Under-Fine-Tuning.git
cd Mechanistic-Interpretability-Study-Induction-Circuit-Stability-Under-Fine-Tuning

# Provision the environment
conda env create -f environment.yml
conda activate mech-interp-induction

# Install package locally in editable mode
pip install -e .

```

### 2. Execute the Pipeline

Run the implementation files in order to generate results from scratch:

```bash
jupyter notebook notebooks/01_replication.ipynb    # ~5 min CPU  — Establish baseline metrics
jupyter notebook notebooks/02_patching.ipynb       # ~15 min CPU — Causal verification 
jupyter notebook notebooks/03_finetuning.ipynb     # ~80 min GPU — Multi-seed training execution
jupyter notebook notebooks/04_adversarial.ipynb    # ~30 min CPU — Checkpoint sweep & detection
jupyter notebook notebooks/05_figures.ipynb        # ~5 min CPU  — Render publication figures

```

### 3. Spin Up the Local Dashboard

Once step `03` saves checkpoints locally, convert weights and launch your interactive visualization engine:

```bash
python src/viz/dashboard/onnx_export.py \
    --pre checkpoints/code_seed42/step_000000.pt \
    --post checkpoints/code_seed42/step_005000.pt

python src/viz/dashboard/app.py
# View the local dashboard interface at http://localhost:7860

```

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


## 📖 Experimental Design & Core Questions

Do structural circuits learned during foundational pre-training survive domain adaptation, or does the network overwrite them to accommodate highly structured syntax? We target three definitive research avenues:

1. **Circuit Degradation vs. Adaptation:** Do pre-existing induction heads degrade entirely, or do they transition into code-specific induction variants?
2. **Phase Change Dynamics:** At what precise training step do structural phase changes materialize within attention layers?
3. **Causal Mapping:** Can we mathematically map intermediate structural states during the fine-tuning trajectory?

### Stated-A-Priori Hypotheses

| ID | Hypothesis | Operational Metric | Quantitative Threshold |
| --- | --- | --- | --- |
| **H1** | Induction circuit degrades rapidly in the first 20% of fine-tuning steps. | IS Drop | $\Delta \text{IS} \ge 0.1$ |
| **H2** | A code-specific induction variant forms during late-stage alignment. | IS Rise on Code Patterns | $\Delta \text{IS} \ge 0.1$ |
| **H3** | Prose fine-tuning causes significantly less structural degradation than code. | $\Delta\text{IS}_{\text{code}}$ vs $\Delta\text{IS}_{\text{prose}}$ | $p < 0.05$ |

### Mathematical Definition of Induction Score (IS)

Following Olsson et al. (2022), for a repeated token sequence $[t_1, \dots, t_n, t_1, \dots, t_n]$, the canonical induction score for layer $l$ and head $h$ is evaluated as:

$$\text{IS}(l,h) = \frac{1}{n} \sum_{i=1}^{n} A^{(l,h)}[n+i,\; i]$$

Scores are averaged over 500 generated sequences with a sequence half-length of 50. An absolute delta $\ge 0.1$ denotes a practically meaningful a-priori circuit shift.

## All 8 Checkpoint Metrics (every 100 steps)

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

---

## 🗺️ Repository Structure

We design our research codebase to read like a structured document. The codebase is broken down into modular operational segments:

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

## ⚠️ Transparent Limitations

* **Model Scope:** This study is strictly bounded within a **2-layer attention-only transformer architecture**. These findings offer clean, isolated mathematical clarity, but they *do not* automatically scale linearly to heavy Multi-Layer Perceptrons (MLPs) or massive frontier architectures (e.g., Llama 3 or GPT-4).
* **Domain Bounding:** Fine-tuning datasets focus entirely on Python code and standard linguistic prose benchmarks.

---


---

## 🧪 Comprehensive Test Suite

We maintain strict test integrity across all core metrics. All unit and integration tests run entirely on a standard CPU node (automatically provisions a ~25MB toy model on initialization).

```bash
# Execute full testing suite with verbose readout
pytest tests/ -v

```

| Target File | Test Coverage Mapping |
| --- | --- |
| `test_config.py` | Config validations, structural parameter typing, initialization limits |
| `test_induction_score.py` | Dimensionality checks, boundary ranges, isolated tracking of known vs non-IS heads |
| `test_patching.py` | Counterfactual prompt generation, activation tracking, clean vs corrupted logit states |
| `test_reproducibility.py` | Global seed determinism, model check-pointing round-trips, identical training loss profiles |
| `test_phase_detection.py` | Savitzky-Golay filtering stability and mathematical edge cases in transition alarms |

---

## 🛠️ Code Quality Standards

* **Formatting:** `black` (line-length=88), `ruff`, and `isort` configurations strictly enforced via pre-commit hooks.
* **Type Safety:** `mypy --strict` passing across all modular modules in `src/`.
* **Standard Logging:** Structured via unified formatting: `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s` utilizing strict ISO 8601 timestamps.

---

## 🤝 References

* Elhage et al. (2021). *A Mathematical Framework for Transformer Circuits.*
* Olsson et al. (2022). *In-Context Learning and Induction Heads.*
* Wang et al. (2022). *Interpretability in the Wild: Localization of an Indirect Object Identification Circuit.*
* Conmy et al. (2023). *Towards Automated Circuit Discovery for Language Models.*

---

**Authors:** [Mattral](https://github.com/Mattral) | **License:** Apache 2.0 License (See [LICENSE](https://www.google.com/search?q=LICENSE))
