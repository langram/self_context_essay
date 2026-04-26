# Phase 1.1 — Experiment A: Transient Linear Probe

- timestamp: 20260426_185237
- git commit: `2c0759b`
- source traces: `data/raw/phase1_gpt2_small/{trained,random}/trace_*.pt`
- probe steps: [0, 1, 2, 3, 5, 7, 10]
- pool: mean over seq_len → 768-dim vector
- model: sklearn LogisticRegression (lbfgs, C=1.0, max_iter=2000) on standard-scaled features
- CV: 5-fold stratified, seed = 42
- chance = 0.167 (6 categories × 10 inputs each)

## Accuracy table

| step | trained mean ± std | random mean ± std |
| ---: | --- | --- |
| 0 | **0.817 ± 0.062** | 0.383 ± 0.067 |
| 1 | 0.800 ± 0.085 | 0.367 ± 0.085 |
| 2 | 0.617 ± 0.180 | 0.383 ± 0.067 |
| 3 | 0.533 ± 0.113 | 0.350 ± 0.062 |
| 5 | 0.500 ± 0.118 | 0.367 ± 0.085 |
| 7 | 0.450 ± 0.155 | 0.350 ± 0.097 |
| 10 | 0.400 ± 0.111 | 0.317 ± 0.122 |

## Reading: partial support for Explanation 4, with a non-trivial residue

The trained-model probe accuracy is **5× chance at step 0** (0.817 vs 0.167)
and decays monotonically to **2.4× chance at step 10** (0.400). Random init
sits flat at ~2× chance the whole time (0.32–0.38). Three observations:

1. **The transient does carry input-discriminating information**, and that
   information is *added* by training: step-0 trained accuracy (0.817) is
   well above step-0 random accuracy (0.383). The first forward pass of
   the trained model surfaces semantic structure that the random transformer
   does not.
2. **That information decays substantially during iteration** — by step 10
   the trained probe has lost ≈ 0.42 of its accuracy (relative loss ~50%).
   The shape of this decay (steep 0→2, flatter 2→10) tracks the Phase 1
   per-token-norm collapse (input differences compressed 100× in the first
   five steps). This part is exactly what Explanation 4 predicts.
3. **But the decay does not reach chance**, nor even reach the random-init
   baseline. At step 10, the trained model's universal attractor still
   holds 0.40 accuracy (vs random's 0.32, vs chance 0.167) — the
   "universal" attractor has *small structured residual variation* that
   a linear probe can read.

This last point is genuinely surprising. Phase 1 reported pairwise cosine
similarity ≥ 0.88 (mean 0.996) between all 60 trained C\*'s, and effective
rank ≈ 1.02. Yet a 6-way linear classifier achieves 0.40 on those same
states. The two findings are consistent — high cosine similarity does
not preclude linearly separable variation in the orthogonal complement.
The 0.4% off-cosine residue carries class structure.

So the cleanest single-sentence reading is: **mode-A iteration on
pretrained GPT-2 small attenuates input-specific information with each
step but does not erase it; the converged "universal attractor" still
discriminates input categories, just much less than the initial state
does**. This is partial support for Explanation 4 (semantics live in
transient) layered with partial pushback (the fixed point isn't *quite*
trivial after all).

## What this changes for the next experiments

- **Experiment C (position-embedding ablation) becomes more pointed.**
  If removing repeated `wpe` injection lets even *more* input-specific
  information survive to step 10, that is direct evidence that
  position-embedding is the dominant homogeniser. If it doesn't help,
  LayerNorm is the more likely culprit.
- **Experiment D (mode C token interface) gains a clean prediction.**
  If hidden-level decay is real but not total, token-level iteration —
  which discretises through argmax — should either preserve much more
  input variation (because argmax is non-contractive in the discrete
  sense) or collapse to a different universal pattern. Either result is
  informative.
- **Experiment B's HTML outlier reading sharpens.** If the universal
  attractor already carries 0.40 of input information, the HTML outlier
  may not be "different" so much as "decayed less". Worth tracking
  per-step probe accuracy on the new HTML/XML/JSON inputs as a
  by-product of B.

## Caveats and methodological holes

- **Sample size is tiny.** N = 60 with 6 classes means each fold sees
  12 examples per class for training and 12 for test. Per-fold std is
  ~0.10 in places (step 2 trained: 0.180). Treat the reported means
  as ±2σ uncertain; treat the *trend* (monotonic decay) as more robust
  than any individual point.
- **Mean-pool over seq_len throws away most of the structure.** The
  signal that survives the pool is the only signal we are measuring.
  A higher step-10 accuracy under last-token or attention-weighted
  pooling would not contradict this finding; it would show the universal
  attractor preserves *more* information than the mean does.
- **Standard-scaling is per-fold on 48 examples in 768 dims.** The
  scaler itself is noisy; this is one reason the trained step-2 std
  (0.18) is much wider than step-0 (0.06).
- **Random-init step-0 at 0.38 is also not trivial.** The first forward
  pass through a random transformer still surfaces some token-distribution
  signal because the random embedding matrix is not degenerate. This
  caps how much "training added" can be claimed: the gap is
  0.817 − 0.383 = 0.43 at step 0, not the full 0.65 above chance.
- **Phase 1's `effective_rank ≈ 1.02` and this finding are not
  contradictory.** Effective rank measures dimensions in `[seq_len,
  hidden_dim]` matrix; we are asking a different question — whether the
  *mean-pooled* 768-vector preserves class structure. Two states can be
  rank-1 in the matrix sense and still differ in their (class-aligned)
  scalar coefficient on the dominant singular vector.

## Pointers

- figure: `outputs/figures/phase1_1_transient_probe/transient_probe_accuracy.png`
- raw fold accuracies: `data/processed/phase1_1_transient_probe/transient_probe_results.json`
- source: `src/transient_probe.py`

## Conclusion for the master report

**Q1 answered: yes, transient hidden states carry input-category information, and
yes, that information decays during iteration — but not to zero, even at the
universal attractor.** Explanation 4 ("semantics live in transient, not at the
fixed point") gets partial support: the *direction* is right, the *magnitude*
is incomplete. Explanation 1 ("FPP hypothesis is just wrong on GPT-2") is
weakened: there *is* per-input structure even at the converged state.

Proceed to Experiments B / C / D — the transient track is now a live
candidate, not the only candidate.
