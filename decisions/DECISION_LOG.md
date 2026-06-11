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

