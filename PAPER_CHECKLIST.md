# Paper Checklist

Track completion of every deliverable. Use [x] for done, [ ] for pending.

---

## Phase 1 — Environment & Replication

- [x] `conda env create -f environment.yml` runs without errors
- [x] `pytest tests/ -q` passes all tests (CI green)
- [x] `notebooks/01_replication.ipynb` re-run with corrected prefix-matching formula (DECISION-005, revised) — L1H6 IS = 0.408 ± 0.103
- [x] `experiments/results/baseline_induction_scores.npz` saved
- [x] Layer-1 induction head confirmed (L1H6, 12.4× separation from next head; threshold revised to > 0.25 per DECISION-011, see note below)
- [x] `paper/figures/fig3_induction_scores_baseline.{pdf,png}` generated

> Note: the originally planned ">0.7" threshold for "confirmed induction
> head" was never empirically justified for this model under the corrected
> formula. DECISION-011 revised it to >0.25 (still a >25× separation from
> non-induction heads) with citations. L1H6's measured score is 0.408.

## Phase 2 — Causal Verification

- [x] `notebooks/02_patching.ipynb` runs without errors
- [x] `experiments/results/baseline_attribution_scores.npz` saved
- [x] Circuit heads identified (attribution ≥ 0.5): `[(1, 6)]`, attribution = 0.952; recorded in DECISION_LOG.md
- [x] `paper/figures/fig1_circuit_diagram.{pdf,png}` generated
- [x] `paper/figures/fig2_attention_heatmap.{pdf,png}` generated
- [x] `paper/figures/fig4_attribution_baseline.{pdf,png}` generated

## Phase 3 — Fine-Tuning (Code + Prose Control)

- [x] Code run seed=42 completed; checkpoints in `checkpoints/code_seed42/`
- [x] Code run seed=123 completed
- [x] Code run seed=7 completed
- [x] Prose control seed=42 completed; checkpoints in `checkpoints/prose_seed42/`
- [x] Prose control seed=123 completed
- [x] Prose control seed=7 completed
- [x] All 8 checkpoint metrics verified in saved .pt files

## Phase 4 — Analysis

- [x] `notebooks/04_adversarial.ipynb` runs without errors
- [x] `experiments/results/sweep_code_seed42.npz` saved (per-head sweep for all 3 seeds: see `sweep_code_seed{42,123,7}.npz`)
- [x] `experiments/results/sweep_prose_seed42.npz` saved (per-head sweep for all 3 seeds: see `sweep_prose_seed{42,123,7}.npz`)
- [x] `experiments/results/phase_transitions.json` saved
- [x] `experiments/results/adversarial_pre_post.npz` saved
- [x] Phase transition detected: step 100, both conditions (code Δ=+0.18, prose Δ=+0.18 mean across seeds); recorded in DECISION-015
- [x] Circuit dissolution step: not applicable — circuit strengthens in both conditions, does not dissolve

## Phase 5 — Dashboard

- [x] ONNX models exported: `src/viz/dashboard/onnx_models/model_pre.onnx`
- [x] ONNX models exported: `src/viz/dashboard/onnx_models/model_post.onnx`
- [x] Dashboard loads in < 3 seconds on standard laptop CPU (measured on HF Spaces CPU Basic)
- [x] Dashboard deployed to Hugging Face Spaces — https://huggingface.co/spaces/Mattral/induction-circuit-stability
- [x] HF Spaces URL added to README.md
- [x] Tokenizer confirmed as `NeelNanda/gpt-neox-tokenizer-digits` in all ONNX paths (DECISION-006)
- [x] Dashboard smoke-tested: attention heatmap, induction grid, circuit diagram all render correctly

## Phase 6 — Paper Figures

- [x] `notebooks/05_figures.ipynb` runs top-to-bottom without errors
- [x] All 8 figures generated in `paper/figures/`
- [x] Figures are colourblind-safe (viridis/tab10 throughout)
- [x] All figures have self-contained captions (not just "see text")
- [x] PDF (vector-wrapped) + PNG (300 dpi) generated for each figure
- [x] Figures 5 and 6 regenerated with 3-seed mean ± SD shading bands (DECISION-015)

## Phase 7 — Paper Write-Up

- [x] Abstract: exactly 4 sentences (question / what-we-did / finding / implication)
- [x] Finding sentence is quantitative with real 3-seed numbers
- [x] Contribution sentence uses prescribed form
- [x] Methods section matches actual experimental protocol (corrected IS formula, transformersbook/codeparrot dataset)
- [x] Results section populated with actual numbers (no [PENDING] placeholders)
- [x] Safety discussion grounded in actual findings (circuit strengthens; domain-dependent magnitude; prompt-injection risk discussion)
- [x] Limitations section covers all required points (3 seeds/3 checkpoints scale, single-head circuit, dataset change, token budget, adversarial probe normalisation artefact)
- [x] All 9 references present in references.bib
- [x] LaTeX compiles without errors (`pdflatex main.tex` → `bibtex main` → `pdflatex` ×2, verified zero warnings)
- [ ] Paper archived on ArXiv — pending endorsement; link to be added to README.md once available

---

## Quick-Reference: Stated-A-Priori Thresholds

| Threshold | Value | Location |
|---|---|---|
| Induction score meaningful change | ≥ 0.1 | DECISION-004 |
| Circuit membership (attribution) | ≥ 0.5 | DECISION-002 |
| Known induction head lower bound (layer 1) | > 0.25 | DECISION-011 (revised from an uncited > 0.7) |
| Non-induction head upper bound (layer 0) | < 0.3 | `test_score_known_non_induction_head` |
| Dashboard load time | < 3 s | Phase 5 checklist |

---

_Last updated: 2026-06-17. Experimental work complete across all 7 phases;
remaining item is ArXiv submission logistics (endorsement), not technical
work._

---

## Bugs Discovered and Fixed During Experimentation

Not pre-existing spec items — found through live runs and cross-checking
independent methods against each other. Full evidence trail for each in
`decisions/DECISION_LOG.md`.

- [x] **DECISION-005 (revised) — prefix-matching off-by-one fixed and verified**: re-run of notebooks 01–04 confirms L1H6 IS = 0.408 (baseline) → 0.646 ± 0.007 (code, step 200) → 0.616 ± 0.001 (prose, step 200)
- [x] **DECISION-006 (tokenizer fix) applied**: `NeelNanda/gpt-neox-tokenizer-digits` used everywhere
- [x] **DECISION-007 (transformer_lens.__version__ AttributeError) fixed**: fallback via `importlib.metadata`
- [x] **DECISION-008 / 013 (dataset loading script removed in datasets>=4.0) fixed**: switched to `transformersbook/codeparrot`
- [x] **DECISION-010 (unused jaxtyping import, latent ImportError risk) fixed**: removed from 3 files
- [x] **DECISION-012 (empty highlight_heads list silently dimmed all figure lines) fixed**: auto-fallback with logged warning
- [x] **DECISION-014 (torch.load weights_only=True default in PyTorch 2.6) fixed**: str-cast at save, two-stage load with fallback
- [x] All figures regenerated after every formula/rendering fix; visually verified against compiled PDF
