# Decision Log

Every non-obvious design decision is recorded here.
Format per AGENT_INSTRUCTIONS Section 11: date, status, context, options, decision, rationale, consequences.

---

## DECISION-001: Mean ablation (shuffled second half) over zero ablation

**Date:** 2026-06  
**Status:** Decided

**Context:** Activation patching requires choosing a baseline for the "corrupted" run.
Two common choices are zero ablation (replace activations with zeros) and mean ablation
(replace with average activation over a reference set).

**Options considered:**
1. Zero ablation — simple, no reference set needed, but zero is out-of-distribution
   for attention outputs and can introduce spurious effects in downstream heads.
2. Mean ablation via second-half shuffling — the second half of each repeated-token
   sequence is independently shuffled per batch element, preserving the marginal
   token distribution while destroying the repetition structure.

**Decision:** Mean ablation via second-half shuffling.

**Rationale:** Preserves the marginal distribution of tokens, making the corrupted run
a valid counterfactual. Follows Wang et al. (2022) protocol. Zero ablation would shift
the distribution of inputs to downstream components, confounding attribution.

**Consequences:** Attribution scores are defined relative to this specific corruption.
A head that recovers induction performance under shuffled inputs may not recover it
under a different corruption (e.g., random token replacement). This is acknowledged
in the Limitations section.

---

## DECISION-002: Attribution threshold 0.5 for circuit membership

**Date:** 2026-06  
**Status:** Decided

**Context:** A threshold is required to discretise the continuous attribution score
into binary circuit membership.

**Options considered:**
1. 0.3 — more inclusive, captures heads with minor contributions.
2. 0.5 — symmetric: the head recovers at least 50% of the corrupted logit gap.
3. 0.7 — stricter; only strongly contributing heads.

**Decision:** 0.5. Stated a priori. Raw attribution scores reported in all figures
to allow readers to apply alternative thresholds.

**Rationale:** 0.5 is the natural symmetry point, easy to interpret, and consistent
with the implicit standard in the mechanistic interpretability literature.

**Consequences:** The set of circuit heads at this threshold may differ from papers
using 0.3 or 0.7. All comparisons with prior work explicitly note this threshold.

---

## DECISION-003: Streamed datasets (no local download)

**Date:** 2026-06  
**Status:** Decided

**Context:** The fine-tuning datasets must be accessible on Colab T4 free tier
without exceeding the ~78 GB disk quota.

**Options considered:**
1. Download and cache — faster per-batch access, but uses significant disk space
   for codeparrot/github-code (hundreds of GB for the full dataset).
2. Stream via `streaming=True` — no disk usage, slightly slower per-batch.

**Decision:** Streaming for both codeparrot and TinyStories.

**Rationale:** Colab T4 disk constraints make local download impractical.
Streaming is fully supported by the Hugging Face datasets API and imposes no
correctness penalty for our 500K-token budget.

**Consequences:** Cannot shuffle the full dataset; uses a buffer_size=1000 local
shuffle. For 500K tokens this is acceptable — the buffer is large enough relative
to typical document sizes to avoid significant ordering bias.

---

## DECISION-004: Meaningful-effect threshold of 0.1 for induction score changes

**Date:** 2026-06  
**Status:** Decided

**Context:** A threshold is needed to determine when a change in induction score
constitutes a practically meaningful difference, not merely noise.

**Options considered:**
1. 0.05 — sensitive but may flag noise in the smoothed curve.
2. 0.1 — corresponds to 10 percentage points on a [0,1] scale.
3. 0.2 — conservative; may miss genuine structural changes.

**Decision:** 0.1. Stated a priori. Not adjusted after seeing results.

**Rationale:** 0.1 is a round number in the natural scale of the metric.
Per-head induction scores in the pretrained model range from ~0.0 to ~0.8,
so a 0.1 change represents a meaningful fraction (~12%) of the dynamic range.

**Consequences:** Phase transitions flagged at this threshold are reported in
the paper. Any transition below 0.1 is not reported as structurally significant.

---

---

## DECISION-005: prepend_bos=False in all run_with_cache calls

**Date:** 2026-06
**Status:** Decided — confirmed by experimental data (Notebooks 01 & 02)

**Context:** TransformerLens prepends a BOS token by default
(`default_prepend_bos=True`). This shifts all attention position indices
by +1, corrupting the induction score formula IS[l,h] = mean_i(A[n+i, i]).

**Evidence from experiments:** The pretrained attn-only-2l model produced
induction scores near zero (max=0.28 at L0H7) while activation patching
correctly identified L1H6 as the induction head with attribution=0.95.
Since attribution uses logit differences (not attention positions), it was
unaffected by the BOS offset. The discrepancy proved the IS formula was
reading the wrong positions.

**Decision:** Pass `prepend_bos=False` to every `run_with_cache()` call in
`src/circuits/` so the attention matrix dimensions are exactly [batch, n_heads,
2n, 2n] and position n+i correctly refers to the second occurrence of token i.

**Alternative considered:** Adjust indices to n+1+i (key) and i+1 (query) to
account for BOS. Rejected because it makes the code harder to verify against
the formula in the paper and introduces asymmetry between sequence construction
and index formula.

**Consequences:** All IS scores will now be recomputed with the correct formula.
Expected result: L1H6 IS ~ 0.9+ (consistent with Olsson et al. 2022 Fig. 4).
All three circuit analysis notebooks (01, 02, 04) must be re-run from scratch.

---

## DECISION-006: attn-only-2l uses NeelNanda/gpt-neox-tokenizer-digits, NOT gpt2

**Date:** 2026-06
**Status:** Decided — confirmed by live HF Space error

**Context:** The ONNX export and dashboard were loading `gpt2` as the tokenizer.
The attn-only-2l pretrained model in TransformerLens uses
`NeelNanda/gpt-neox-tokenizer-digits` (vocab_size ≈ 48262), not the GPT-2
tokenizer (vocab_size = 50257). Feeding GPT-2 token IDs to a neox-tokenizer
ONNX model produces completely wrong token mappings, leading to garbled
attention patterns and meaningless induction scores in the dashboard.

**Evidence:** The HF Space produced incorrect outputs with gpt2. Switching to
`NeelNanda/gpt-neox-tokenizer-digits` fixed the dashboard (confirmed by
@Mattral in live testing).

**Decision:** All tokenizer loads across the codebase now use
`NeelNanda/gpt-neox-tokenizer-digits`:
  - `src/model/config.py`: ModelConfig.tokenizer default
  - `src/model/finetune.py`: via model_config.tokenizer
  - `src/viz/dashboard/app.py`: local dashboard
  - `src/viz/dashboard/onnx_export.py`: ONNX export
  - `huggingface_space/app.py`: HF Space dashboard

**Consequences:** The ONNX export must be re-run after any change to tokenizer
configuration. Notebooks 01-04 that use the tokenizer must be re-run with this
fix. The HF Space ONNX models in `onnx_models/` remain valid (they were
exported with the model itself, not with gpt2).


---

## DECISION-007: Safe transformer_lens version retrieval

**Date:** 2026-06
**Status:** Decided — confirmed by Colab T4 run (notebook 03)

**Context:** `save_checkpoint()` accessed `transformer_lens.__version__`
directly. On Colab, the pip-installed `transformer_lens` package does not
expose a top-level `__version__` attribute, raising:
```
AttributeError: module 'transformer_lens' has no attribute '__version__'
```
This crashed `run_finetuning()` at the very first checkpoint (step 0),
after baseline metrics were already computed and logged.

**Decision:** Added `_get_transformer_lens_version()` helper in
`src/model/train.py` that tries `transformer_lens.__version__` first, falls
back to `importlib.metadata.version("transformer_lens")`, and returns
`"unknown"` if neither succeeds. `save_checkpoint()` now calls this helper
instead of accessing the attribute directly.

**Consequences:** Checkpoints saved on environments without
`transformer_lens.__version__` will record the version from package
metadata (or "unknown") instead of crashing. No change to checkpoint
structure or downstream analysis code.

**Note on the parallel BOS issue:** The Colab run's `IS_mean=0.0336` at
step 0 (matching the pre-fix `fig3_induction_scores_baseline.png`, where
L1H6=0.04) confirms the BOS fix (DECISION-005) has **not yet propagated to
the GitHub repository**. Notebook 03 clones from GitHub
(`Mattral/Mechanistic-Interpretability-Study-...`), which still contains the
pre-fix `induction_score.py`. The BOS and tokenizer fixes (DECISION-005,
DECISION-006) exist only in the delivered zip archives. **Action required:**
push the corrected `src/` tree (from `mech-interp-induction-final-v3.zip`) to
the GitHub repository before re-running notebooks 01–04 on Colab.

---

## DECISION-008: codeparrot/github-code subset name and trust_remote_code

**Date:** 2026-06
**Status:** Decided — confirmed by Colab T4 run (notebook 03, code fine-tuning)

**Context:** `run_finetuning()` crashed on the first call to
`build_code_dataloader()` with:
```
ValueError: BuilderConfig 'Python' not found.
Available: ['all-all', 'all-mit', ..., 'Python-all', 'Python-mit', ...]
```
`codeparrot/github-code` does not expose a bare `"Python"` builder config —
subset names are `"<Language>-<license>"`, e.g. `"Python-all"` (all
licenses) or `"Python-mit"` (MIT-licensed only).

A second, related issue: loading this dataset triggers an interactive
`Do you wish to run the custom code? [y/N]` prompt (the dataset ships a
custom loading script). In a non-interactive Colab cell this prompt would
hang indefinitely; the user's run only proceeded because they answered
manually before the ValueError occurred downstream.

**Decision:**
1. `TrainConfig.dataset_subset` default changed from `"Python"` to
   `"Python-all"` (all licenses, broadest sample — matches the "Python code"
   framing in the hypothesis with no license-based filtering bias).
2. `experiments/configs/finetune_code.yaml` updated to match.
3. `load_dataset(...)` in `TokenisedStreamDataset.__iter__` now passes
   `trust_remote_code=True` so the prompt never blocks execution.

**Consequences:** No change to the prose control (`roneneldan/TinyStories`
has no custom loading script and a single default config, so it is
unaffected). All three code fine-tuning seeds (42, 123, 7) must be re-run
from notebook 03 with this fix.

---

## DECISION-009: d_model / d_head corrected to match attn-only-2l (512 / 64)

**Date:** 2026-06
**Status:** Decided — confirmed by Colab T4 run (notebook 03 log)

**Context:** `ModelConfig` and all three experiment YAMLs stated
`d_model=256, d_head=32`. The actual loaded model reports:
```
Model loaded: 2L 8H d_model=512
```
With `n_heads=8`, this implies `d_head = 512/8 = 64`, not 32. The stated
values (256/32) were incorrect from the start — likely transcribed from a
different TransformerLens toy model rather than `attn-only-2l` itself.

**Decision:** Updated `d_model: int = 512` and `d_head: int = 64` in
`src/model/config.py`, all three `experiments/configs/*.yaml` files, and
`paper/sections/methods.tex`. The `ModelConfig.__post_init__` validator
(`d_model == num_heads * d_head`) passes for 512 == 8*64.

**Consequences:** Purely a metadata/documentation correction — no code
path hardcoded the old 256/32 values (all dimensions are read dynamically
from `model.cfg` at runtime), so no functional behaviour changes. The paper
methods section now correctly states the model's actual architecture.
