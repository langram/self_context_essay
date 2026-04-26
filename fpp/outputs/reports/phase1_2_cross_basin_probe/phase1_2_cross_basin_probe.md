# Phase 1.2 — Experiment E: Cross-basin Probe

- source data: Phase 1.1 Experiment B 30-input nested zoo (`data/raw/phase1_1_html_outlier/`)
- feature: h_10 mean-pooled to 768 (the same step Phase 1.1 A measured at)
- basin labels assigned from the **final** state's cos to a Phase 1 lowercase converger
- thresholds: cos > 0.999 → lowercase, cos < 0.7 → capital, else hybrid
- CV: leave-one-out (per plan §6.2 — small per-basin samples)
- model: sklearn LogisticRegression (C = 1, lbfgs, max_iter = 4000), standard-scaled features

## Basin distribution under the chosen thresholds

| basin | n | composition |
| --- | ---: | --- |
| **capital** (cos < 0.7) | 10 | html × 5, xml × 4, latex × 1 |
| **lowercase** (cos > 0.999) | 8 | xml × 1, latex × 4, markdown_list × 4 (note: markdown_list#25–27,29 also have cos = 0.9995 / 0.9996 just above the threshold; #28 sits at exactly 1.0) |
| **hybrid** (0.7 ≤ cos ≤ 0.999) | 12 | json × 5, pure_brackets × 5, latex × 2 |

Every nested category lands in at least one basin, but most concentrate in
one (html/xml → capital, json/pure_brackets → hybrid, markdown_list → lowercase).
Latex spreads across all three. This already weakens any "category = basin"
identity claim: the same surface category goes to different basins depending
on its content.

## Probe results

| probe | n samples | n classes | majority chance | LOO accuracy ± std | accuracy / chance |
| --- | ---: | ---: | ---: | ---: | ---: |
| **probe-basin** (3-way) | 30 | 3 | 0.400 | **0.900 ± 0.300** | 2.25× |
| probe-fine-global (6-way) | 30 | 6 | 0.167 | 0.833 ± 0.373 | 4.99× |
| probe-fine within capital | 10 | 3 (html/xml/latex) | 0.500 | 0.800 ± 0.400 | 1.60× |
| probe-fine within hybrid | 12 | 3 (json/pure_brackets/latex) | 0.417 | **1.000 ± 0.000** | 2.40× |
| probe-fine within lowercase | 8 | 3 (xml/latex/markdown_list) | 0.625 | 0.750 ± 0.433 | 1.20× |

(LOO accuracy on N samples reports per-fold 0/1 outcome, so std reflects the
binomial structure rather than continuous variance — a 0.4 std with mean 0.8
on N = 10 means 2 of 10 LOO folds misclassified.)

## Reading

**The step-10 hidden state encodes basin label, and within at least one basin
it also encodes sub-basin information.**

Three observations:

1. **Probe-basin at 0.90 vs chance 0.40 confirms the master-report hypothesis**
   that the Phase 1.1 transient probe's residual 0.40 step-10 accuracy is
   *partly* basin label. By step 10, an input has effectively committed to
   a basin (capital / lowercase / hybrid) — and the basin assignment is
   already linearly readable from the (mean-pooled) hidden state.

2. **Within the hybrid basin, sub-basin information is fully readable.**
   The within-hybrid probe achieves 1.000 LOO accuracy with N = 12 across
   3 categories (json / pure_brackets / latex). With perfect LOO accuracy
   on a 12-sample 3-class problem the binomial p-value is `(1/3)^12 ≈ 1.9e-6`
   — this is unlikely to be chance even with the small sample.

3. **Within the capital and lowercase basins, sub-basin information is
   weak.** Within-capital 0.80 vs chance 0.50 (just 1.6× chance, std 0.40);
   within-lowercase 0.75 vs chance 0.625 (only 1.2× chance, std 0.43).
   These two basins are dominated by their register pull and the categories
   inside them converge to *very* similar fixed points — the within-basin
   signal at h_10 is barely there.

**Synthesis: the 0.40 step-10 residue from Phase 1.1 A is not "purely
basin label" nor "rich sub-basin information". It is a basin-label-dominated
signal with sub-basin structure that varies across basins.** Hybrid-basin
inputs retain readable per-category structure; capital and lowercase
inputs do not.

This is the cleanest version of the Phase 1.1 master-report Q1 we can
get from existing data. It does not strongly confirm or refute the §4
hypothesis — it converts the binary question into a structured one.

## What this changes for the §4 hypothesis

The §4 hypothesis asked: do trained transformers reach stable
*per-input semantic interpretations*? The data now says:

- **Per-input variation persists at step 10** (probe-fine-global 0.83 vs
  chance 0.17), but is absorbed into a small set of register-typed basins
  by the time the iteration converges.
- **Within a basin, per-input structure is preserved only in the
  intermediate-rank hybrid basin** (eff_rank ≈ 1.5, ‖C\*‖ ≈ 2400 from
  Experiment B). Within the rank-≈ 1 capital basin (‖C\*‖ ≈ 1042) and
  the rank-≈ 1 lowercase basin (‖C\*‖ ≈ 2564), the contraction
  collapses category structure as well as register structure.
- **The §4 hypothesis is supported in *measure*, not in *kind*.** The
  trained model does retain input-discriminating information at the
  fixed point, but only in the basins where the contraction is incomplete.
  The strong-rank-1 basins are still effectively semantic dead zones.

## Caveats

- **Sample sizes are small.** Per-basin N is 8–12. LOO is the right CV but
  even a single misclassified fold moves accuracy by ~10%.
- **The "hybrid basin = 100%" result is anchored on JSON-vs-brackets-vs-latex
  surface differences.** JSON has heavy `{` / `}` / `"` tokens, brackets
  have heavy `(` / `)` tokens, latex has heavy `\` / `_` / `^` tokens.
  These are *visually* distinct categories at step 10 and the probe can
  read them. This does not prove that the probe is reading *semantic*
  content — it may be reading surface-token-distribution memory that
  hasn't been washed out by step 10 yet.
- **The basin thresholds are not sharp.** Several markdown inputs have
  cos = 0.9995–0.9996 (just below the 0.999 lowercase threshold) and
  flip into hybrid. A small threshold change reshuffles the within-basin
  composition; the probe results inside each basin may not be robust to
  ±0.001 threshold drift.
- **All inputs are nested-structure inputs by design.** The Phase 1
  natural-text 60-input zoo would give different within-basin probe
  accuracy because the basin compositions would be different. The Q1
  answer here is conditional on the Phase 1.1 B input distribution.
- **One basin (lowercase) has a within-basin probe accuracy that is
  effectively at chance.** With chance 0.625 and accuracy 0.75 ± 0.43,
  the data does not support a claim that the lowercase basin retains
  any per-category information at step 10.

## Pointers

- results JSON: `data/processed/phase1_2_cross_basin_probe/results.json`
- figure: `outputs/figures/phase1_2_cross_basin_probe/cross_basin_probe.png`
- source: `src/cross_basin_probe.py`

## Conclusion for the master report

**Q1 answered (conditional): the step-10 residual probe accuracy is
basin-label-dominated, with sub-basin structure that varies sharply by
basin.** Probe-basin alone reaches 0.90 (chance 0.40); within-basin
probes vary from at-chance (lowercase) to perfect (hybrid). The 0.40
step-10 residue from Phase 1.1 A is therefore a basin-label signal
plus a per-basin amount of additional structure.

This rules out the strict "fixed point is informationless" reading
(Phase 1's claim) and rules out the strict "fixed point preserves
per-input semantics" reading (a maximalist §4). The truth is a
graded compromise that depends on which basin the input falls into,
which itself is determined by the surface lexical features Experiment H
will analyse.
