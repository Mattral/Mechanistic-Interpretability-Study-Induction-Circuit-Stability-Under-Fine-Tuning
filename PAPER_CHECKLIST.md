# Paper Checklist

Track completion of every deliverable in AGENT_INSTRUCTIONS v1.0.0.
Update this file as each item is completed. Use [x] for done, [ ] for pending.

---

## Phase 1 — Environment & Replication

- [ ] `conda env create -f environment.yml` runs without errors
- [ ] `pytest tests/ -q` passes all tests (CI green)
- [ ] `notebooks/01_replication.ipynb` runs top-to-bottom without errors
- [ ] `experiments/results/baseline_induction_scores.npz` saved
- [ ] Layer-1 induction head confirmed (IS ≥ 0.5, consistent with Olsson et al. 2022)
- [ ] `paper/figures/fig3_induction_scores_baseline.{pdf,png}` generated

## Phase 2 — Causal Verification

- [ ] `notebooks/02_patching.ipynb` runs without errors
- [ ] `experiments/results/baseline_attribution_scores.npz` saved
- [ ] Circuit heads identified (attribution ≥ 0.5); list recorded in DECISION_LOG.md
- [ ] `paper/figures/fig1_circuit_diagram.{pdf,png}` generated
- [ ] `paper/figures/fig2_attention_heatmap.{pdf,png}` generated
- [ ] `paper/figures/fig4_attribution_baseline.{pdf,png}` generated

## Phase 3 — Fine-Tuning (Code + Prose Control)

- [ ] Code run seed=42 completed; checkpoints in `checkpoints/code_seed42/`
- [ ] Code run seed=123 completed
- [ ] Code run seed=7 completed
- [ ] Prose control seed=42 completed; checkpoints in `checkpoints/prose_seed42/`
- [ ] Prose control seed=123 completed
- [ ] Prose control seed=7 completed
- [ ] All 8 checkpoint metrics verified in saved .pt files

## Phase 4 — Analysis

- [ ] `notebooks/04_adversarial.ipynb` runs without errors
- [ ] `experiments/results/sweep_code_seed42.npz` saved (all 3 seeds)
- [ ] `experiments/results/sweep_prose_seed42.npz` saved (all 3 seeds)
- [ ] `experiments/results/phase_transitions.json` saved
- [ ] `experiments/results/adversarial_pre_post.npz` saved
- [ ] Phase transition detected / not detected (record result in DECISION_LOG.md)
- [ ] Circuit dissolution step identified / confirmed stable

## Phase 5 — Dashboard

- [ ] ONNX models exported: `src/viz/dashboard/onnx_models/model_pre.onnx`
- [ ] ONNX models exported: `src/viz/dashboard/onnx_models/model_post.onnx`
- [ ] Dashboard loads in < 3 seconds on standard laptop CPU (measured, record time)
- [ ] Dashboard deployed to Hugging Face Spaces
- [ ] HF Spaces URL added to README.md
- [ ] Dashboard smoke-tested: attention heatmap, induction grid, circuit diagram all render

## Phase 6 — Paper Figures

- [ ] `notebooks/05_figures.ipynb` runs top-to-bottom without errors
- [ ] All 8 figures generated in `paper/figures/`
- [ ] Figures are colourblind-safe (viridis/tab10 throughout)
- [ ] All figures have self-contained captions (not just "see text")
- [ ] PDF (vector) + PNG (300 dpi) generated for each figure

## Phase 7 — Paper Write-Up

- [ ] Abstract: exactly 4 sentences (question / what-we-did / finding / implication)
- [ ] Finding sentence is quantitative (e.g. "IS decreases from X to Y ± Z")
- [ ] Contribution sentence uses prescribed form (Section 1.3 of AGENT_INSTRUCTIONS)
- [ ] Methods section matches actual experimental protocol (no divergence)
- [ ] Results section populated with actual numbers (no [PENDING] placeholders)
- [ ] Safety discussion grounded in actual findings
- [ ] Limitations section covers all 6 required points
- [ ] All 7 references present in references.bib
- [ ] LaTeX compiles without errors (`pdflatex main.tex`)
- [ ] Paper archived on ArXiv (or internal pre-print server); link added to README.md

---

## Quick-Reference: Stated-A-Priori Thresholds

| Threshold | Value | Location |
|---|---|---|
| Induction score meaningful change | ≥ 0.1 | Section 4.5, DECISION-004 |
| Circuit membership (attribution) | ≥ 0.5 | Section 4.3, DECISION-002 |
| Non-induction head upper bound (layer 0) | < 0.3 | test_score_known_non_induction_head |
| Dashboard load time | < 3 s | Phase 5 checklist |

---

_Last updated: 2026-06 (scaffold complete; experiments pending)_
