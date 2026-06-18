<div align="center">

# Mechanistic Interpretability Study: Induction Circuit Stability Under Fine-Tuning

[![Preprint](https://img.shields.io/badge/Preprint-ResearchSquare-3b82f6?style=for-the-badge)](https://www.researchsquare.com/article/rs-10067094/v1)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/Mattral/induction-circuit-stability)

</div>

A mechanistic interpretability study asking a simple question: does the
induction head circuit in a 2-layer attention-only transformer survive
fine-tuning on a narrow distribution, or does it change — and if it
changes, in which direction? We measure per-head induction scores and
activation-patching attribution at every checkpoint, across three random
seeds, for both a Python code fine-tuning run and a TinyStories prose
control.

**Headline result:** the circuit does not dissolve under fine-tuning.
It strengthens, under both conditions, and significantly more so under
code than under prose. Full numbers in [Results](#results-summary) below.

<div align="center">
  <a href="https://huggingface.co/spaces/Mattral/induction-circuit-stability" target="_blank">
    <img src="ICSUFT.png" alt="Induction Circuit Stability Dashboard" width="800">
  </a>
  <p>
    <b>Interactive Dashboard:</b> <a href="https://huggingface.co/spaces/Mattral/induction-circuit-stability">Try it live on Hugging Face Spaces</a>
    — toggle pre/post fine-tuning and inspect the attention pattern on your own input text.
  </p>
</div>

---
## Overview

This repository contains a mechanistic interpretability study examining the stability of **induction heads** and related circuits in a small (2-layer attention-only) Transformer during domain adaptation.

We fine-tune the model from prose to structured Python code and track circuit behavior across training using:
- Induction scores
- Activation patching for causal attribution
- Multi-seed experiments with checkpoint analysis

The goal is to understand whether induction circuits persist, adapt, or undergo phase changes when the model shifts domains.

**Note**: This study is conducted on a minimal 2-layer attention-only model. Findings may not directly generalize to larger MLPs or frontier-scale Transformers.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Mattral/Mechanistic-Interpretability-Study-Induction-Circuit-Stability-Under-Fine-Tuning
cd mech-interp-induction

# 2. Set up environment (Python 3.10)
conda env create -f environment.yml
conda activate mech-interp-induction

# 3. Run the full experiment pipeline (in order), or use the combined notebook
jupyter notebook notebooks/01_replication.ipynb    # ~5 min CPU  — baseline IS scores
jupyter notebook notebooks/02_patching.ipynb        # ~15 min CPU — causal verification
jupyter notebook notebooks/03_finetuning.ipynb      # ~80 min GPU (T4) — code + prose, 3 seeds each
jupyter notebook notebooks/04_adversarial.ipynb     # ~30 min CPU — checkpoint sweep + phase detection + probes
jupyter notebook notebooks/05_figures.ipynb         # ~5 min CPU  — all paper figures
# Or: notebooks/01_to_05_all_in_one.ipynb combines all five for Colab convenience.

# 4. Run tests
pytest tests/ -v

# 5. Launch dashboard locally (after completing notebook 03)
python src/viz/dashboard/onnx_export.py \
    --pre  checkpoints/code_seed42/step_000000.pt \
    --post checkpoints/code_seed42/step_000200.pt
python src/viz/dashboard/app.py
# Open http://localhost:7860
```

> All three seeds (42, 123, 7) for both conditions are required to
> reproduce the confidence intervals reported in the paper. Notebook 03
> trains all three; notebook 04 sweeps the resulting checkpoints into
> per-checkpoint metrics.

---

## Repository Structure

```
mech-interp-induction/
├── PAPER_CHECKLIST.md          ← Deliverable completion tracker
├── src/
│   ├── model/
│   │   ├── config.py           ModelConfig, TrainConfig, EvalConfig (all hyperparams)
│   │   ├── train.py            Generic training loop + checkpoint utilities
│   │   └── finetune.py         Fine-tuning entry point (code + prose) — all 8 metrics
│   ├── circuits/
│   │   ├── induction_score.py  Canonical prefix-matching score (Olsson et al. 2022)
│   │   ├── patching.py         Activation patching + circuit attribution
│   │   ├── path_patching.py    Path patching for indirect effect decomposition
│   │   └── attribution.py      Direct logit attribution (DLA)
│   ├── analysis/
│   │   ├── checkpoint_sweep.py All 8 metrics across every checkpoint
│   │   ├── phase_detection.py  Savitzky-Golay smoothing + transition detection
│   │   └── adversarial.py      22 adversarial probe types
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
│   ├── 01_replication.ipynb    Phase 1: replicate induction circuit
│   ├── 02_patching.ipynb       Phase 2: causal verification
│   ├── 03_finetuning.ipynb     Phase 3: fine-tuning (code + prose, 3 seeds each)
│   ├── 04_adversarial.ipynb    Phase 4: checkpoint sweep + phase detection + probes
│   ├── 05_figures.ipynb        ★ Single source of truth for all paper figures
│   └── 01_to_05_all_in_one.ipynb  All five phases combined, for Colab convenience
├── experiments/
│   ├── configs/                YAML configs (baseline, code, prose)
│   └── results/                Tracked experimental output (.npz, .json) — see below
├── paper/                      LaTeX paper (single-column) + tracked figures/
├── decisions/DECISION_LOG.md   Every non-obvious design decision and bug fix (15 entries)
└── tests/                      Full pytest test suite (CPU, no GPU needed)
```

**Note on tracked data:** unlike many research repos, `experiments/results/`
and `paper/figures/` are committed to this repository rather than
gitignored. The experiments are small enough (a few hundred KB total) that
committing the real output is more useful for reproducibility than
regenerating it — anyone cloning the repo can inspect the exact numbers
behind every figure without re-running anything.

---

## Experimental Design & Core Questions

This study is structured as a controlled comparison, not an open-ended
exploration. Before any fine-tuning run, we fixed the model, the metric,
the comparison condition, and the threshold for "meaningful change" —
all logged below — so that the analysis afterward is a check against a
pre-registered prediction rather than a search for a story in the data.

**The model.** `attn-only-2l` (Neel Nanda's pretrained 2-layer,
attention-only transformer, 8 heads per layer, `d_model=512`) was chosen
because its induction circuit is already well characterised in the
mechanistic interpretability literature — a single documented head, L1H6,
implements the canonical "prefix-matching" behaviour described by
Olsson et al. (2022). Starting from a circuit whose existence and identity
are independently confirmed in prior work means any change we observe
after fine-tuning is attributable to the fine-tuning, not to uncertainty
about what the circuit was in the first place.

**The intervention.** We fine-tune this model on 500,000 tokens of two
very different distributions: Python source code
(`transformersbook/codeparrot`) and short children's stories
(`roneneldan/TinyStories`). Code is the primary condition; prose is the
control. The two corpora were chosen specifically because they differ
enormously in surface structure (syntax-heavy and repetitive vs.
free-form natural language) while both plausibly engage the same
"copy a token you've seen before" mechanism that induction heads
implement — code repeats variable names and function calls; prose repeats
character names and recurring phrases. If fine-tuning affects the circuit
at all, comparing these two conditions tells us whether the effect is
domain-general or domain-specific.

**The question, precisely.** Not "is the circuit still there after
fine-tuning" as a yes/no, but: at every checkpoint during training, what
is L1H6's prefix-matching score, and how does its trajectory differ
between the code and prose conditions? This is answered with a number at
every 100-step checkpoint, not a single before/after snapshot, so that we
can also see *when* any change happens, not just *whether* it happens.

**Why this matters for safety.** Capability benchmarks measure whether a
fine-tuned model can still do a task. They do not measure whether the
internal mechanism it uses to do that task is the same mechanism, a
different one, or a degraded one. A model that scores identically on a
benchmark before and after fine-tuning could have lost a circuit and
compensated some other way, or could have a circuit that works
identically but is now invoked under different conditions. Tracking
the circuit directly, rather than inferring its state from downstream
task performance, is the entire methodological point of this study.

### Hypotheses (stated before the fine-tuning runs)

| ID | Hypothesis | Metric | Threshold |
|----|-----------|--------|-----------|
| H1 | Induction circuit degrades (IS decreases) during fine-tuning | IS drop from step-0 baseline | ≥ 0.1 |
| H2 | If the circuit changes, the magnitude differs between code and prose | \|ΔIS(code) − ΔIS(prose)\| | ≥ 1× pooled SD (suggestive), ≥ 2× (significant) |
| H3 | Any detected change is concentrated in a specific training-step window, not spread evenly | Single-step \|Δ\| ≥ meaningful-change threshold | ≥ 0.1 |

**What we actually found, for direct comparison:** H1 was *not* confirmed
— the circuit strengthened (IS increased) under both conditions, the
opposite of the stated hypothesis. H2 *was* confirmed, in a more specific
form than originally stated: code fine-tuning produced a significantly
larger IS increase than prose by step 200 (4.1× pooled SD). H3 was
confirmed: nearly all of the change occurs in the first 100 of 244 steps,
with a partial plateau afterward. See [Results](#results-summary) below
for the full numbers, and `paper/main.pdf` for the complete writeup
including the safety implications of a circuit that strengthens rather
than dissolves.

### Induction Score Definition

For a repeated-token sequence $[t_1,\ldots,t_n,\,t_1,\ldots,t_n]$
(half-length $n$, no BOS token), the induction (prefix-matching) score
for head $(l,h)$ is:

$$\text{IS}(l,h) = \frac{1}{n-1}\sum_{j=0}^{n-2} A^{(l,h)}[n+j,\;j+1]$$

This is the attention from a token's second occurrence back to the
position immediately *following* its first occurrence — the value an
induction head must copy to predict correctly — following the
prefix-matching score formula in Olsson et al. (2022) and Nanda & Bloom's
TransformerLens reference implementation. An earlier version of this
codebase used the formula $A^{(l,h)}[n+i,\,i]$ (attention back to the
token's own first occurrence, not the following position), which produced
near-zero scores for every head including the genuine induction head; see
`decisions/DECISION_LOG.md`, DECISION-005 (REVISED), for the full
derivation and the independent evidence (activation-patching attribution)
that exposed the bug.

Baseline measured over 100 sequences, half-length 30, seed 42.
**Change ≥ 0.1 = practically meaningful (stated a priori, DECISION-004).**

### All 8 Checkpoint Metrics (every 100 steps)

| # | Metric | Type | Description |
|---|--------|------|--------------|
| 1 | `induction_score` | [L, H] | Per-head IS (canonical definition above) |
| 2 | `induction_score_delta` | [L, H] | Change from step-0 baseline |
| 3 | `train_loss` | scalar | Cross-entropy on training batch |
| 4 | `induction_task_loss` | scalar | CE on held-out induction sequences |
| 5 | `logit_diff_clean` | scalar | Logit diff on clean induction task |
| 6 | `logit_diff_corrupted` | scalar | Logit diff on corrupted baseline |
| 7 | `code_icl_score` | scalar | ICL accuracy on code copy patterns (requires `tokenizer=`; see note below) |
| 8 | `circuit_attribution` | [L, H] | Activation-patching attribution |

> `code_icl_score` is skipped (logged as a warning, not an error) unless a
> tokenizer is explicitly passed to `sweep_checkpoints()`. This is by
> design — the metric needs detokenised text to check for genuine code
> copy-patterns, which most sweep calls don't need.

### Stated-A-Priori Thresholds

| Threshold | Value | Decision |
|---|---|---|
| Meaningful IS change | ≥ 0.1 | DECISION-004 |
| Circuit membership (attribution) | ≥ 0.5 | DECISION-002 |
| Known induction head lower bound (layer 1) | > 0.25 | DECISION-011 |
| Dashboard load time | < 3 s | Phase 5 checklist |

---

## Results Summary

Full derivations, all figures, and the safety discussion are in
`paper/main.pdf`. The numbers below are the same ones reported there,
confirmed across three random seeds (42, 123, 7).

**Baseline (pretrained model).** Head L1H6 is the circuit: induction
score $0.408 \pm 0.103$ (12.4× the next-highest head), activation-patching
attribution $0.952$ (the only head exceeding the 0.5 circuit-membership
threshold).

**Fine-tuning (244 steps, checkpoints at 0/100/200, mean ± SD over 3 seeds).**

| Condition | Step 100 | Step 200 |
|---|---|---|
| Code (`transformersbook/codeparrot`) | $0.591 \pm 0.005$ | $0.646 \pm 0.007$ |
| Prose (`roneneldan/TinyStories`, control) | $0.583 \pm 0.002$ | $0.616 \pm 0.001$ |

Both conditions **increase** L1H6's induction score well above the 0.1
meaningful-change threshold — the circuit strengthens, it does not
dissolve. The code/prose gap at step 200 ($+0.030$) is 4.1× the pooled
standard deviation, a statistically meaningful separation given the tight
inter-seed variance observed in both conditions.

**Adversarial probes (post fine-tuning, code, step 200, 22 probe types).**
Structure-preserving perturbations retain $0.944 \pm 0.043$ of the clean
score; structure-removing perturbations drop to $0.132 \pm 0.030$,
confirming the strengthened head remains selective to genuine
prefix-matching structure rather than becoming a generic high-attention
artefact.

---

## Regenerating Tracked Files

Most experimental output is committed to this repo (see note above), but
can be regenerated from scratch:

| Directory | How to regenerate |
|---|---|
| `checkpoints/` | Run `notebooks/03_finetuning.ipynb` (~80 min/seed on a T4 GPU; gitignored — large) |
| `experiments/results/` | Run notebooks 01–04 in order (already tracked; re-run to verify) |
| `paper/figures/` | Run `notebooks/05_figures.ipynb` after 01–04 complete (already tracked; re-run to verify) |
| `src/viz/dashboard/onnx_models/` | Run `src/viz/dashboard/onnx_export.py` (gitignored — large binary) |

---

## Running Tests

```bash
pytest tests/ -v
# All tests run on CPU. Downloads ~25 MB pretrained model on first run.
```

| File | What is tested |
|------|-----------------|
| `test_config.py` | ModelConfig validation, TrainConfig properties |
| `test_induction_score.py` | Shape, range, known IS heads (>0.25), reproducibility, non-IS heads (<0.3) |
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
- Vaswani et al. (2017). *Attention Is All You Need.*
- Nanda & Bloom (2022). *TransformerLens.*
- Wolf et al. (2020). *Transformers: State-of-the-Art Natural Language Processing.*
- Eldan & Li (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*

---

## Bugs Found and Fixed During Development

This project went through fifteen logged decisions
(`decisions/DECISION_LOG.md`), several of which were genuine bugs caught
through cross-checking two independent methods against each other, or
through breaking changes in upstream libraries. The most consequential:

| Decision | Bug | Symptom | Fix |
|---|-----|---------|-----|
| DECISION-005 (revised) | Induction score formula read attention to a token's *own* first occurrence (`A[n+i,i]`) instead of the position *following* it (`A[n+j,j+1]`) | L1H6 scored 0.035 despite an independent activation-patching measurement giving the same head 0.952 attribution | Corrected formula per Olsson et al. (2022); L1H6 now scores 0.408, a >10× increase |
| DECISION-006 | `attn-only-2l` uses `NeelNanda/gpt-neox-tokenizer-digits`, not `gpt2` | Garbled (but error-free) attention patterns in the dashboard | All tokenizer loads updated to the correct neox tokenizer |
| DECISION-008 / 013 | `codeparrot/github-code` uses a loading script, unsupported in `datasets>=4.0` | `RuntimeError: Dataset scripts are no longer supported` | Switched to `transformersbook/codeparrot` (same source, Parquet format) |
| DECISION-014 | PyTorch 2.6 changed `torch.load`'s `weights_only` default to `True`, which rejects the `TorchVersion` object stored in checkpoints | `UnpicklingError` when sweeping checkpoints | Cast version string to `str()` at save time; two-stage load with a logged fallback for older checkpoints |
| DECISION-012 | An empty (not `None`) `highlight_heads` list silently dimmed every line in the induction-score-over-training figure | Headline result (L1H6's trajectory) was nearly invisible in Figures 5–6 | Auto-fallback to the highest-scoring head with a logged warning when the list is empty |

The original DECISION-005 entry (retained in the log for the debugging
trail) misdiagnosed the near-zero score as a BOS-token offset; that fix
(`prepend_bos=False`) had no actual effect, since test sequences are
constructed as integer tensors and TransformerLens only applies
BOS-prepending to string input. The real fix is the index correction
described above. See the full log for all fifteen decisions, each with
the evidence that motivated it.

---

## Commit Standards

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
fix(induction_score): correct off-by-one in prefix-matching formula
docs(paper): draft results section with Fig 5 and Fig 6 references
test(patching): add clean-run full-recovery assertion
deps: update gradio to 4.37.2
```

**Rules:**
- Every commit must leave the codebase in a runnable state.
- No broken imports, no half-finished refactors in `src/`.
- Scratch work goes in `notebooks/`, never in `src/`.
- No direct commits to `main`. All changes via branch + self-review.

---

## Publication

**Preprint**  
Myet. Min Htet (2026). *Induction Circuit Stability Under Fine-Tuning: A Mechanistic Interpretability Study*.  
ResearchSquare. https://www.researchsquare.com/article/rs-10067094/v1


