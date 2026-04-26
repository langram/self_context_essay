# Phase 1.1 — Experiment C: Position-embedding ablation

- model: `gpt2`, fp32, CUDA
- input zoo: same Phase 1 60-input set (6 categories × 10), seq_len = 32
- variants: C1 (cancel-pos: subtract `wpe[:seq_len]` before each iteration) and
  C2 (posfree: manually run blocks + `ln_f`, skipping all embeddings)
- both run on trained checkpoint and random-init twin
- max_iter = 100, threshold = 1e-3, save_every = 1
- elapsed: 117 s

## TL;DR

**Position embedding is not the cause of the trained-model universal-attractor
collapse.** With repeated `wpe` injection cancelled, the trained model's 60
final states are still mutually cos-similar at 0.991 (mode-A baseline: 0.996),
still effective-rank 1.002 (baseline: 1.02), and individually cos-similar to
their mode-A counterparts at mean **0.9985**. The Phase 1 finding is robust
to position-embedding ablation.

**C1 and C2 agree to 7 decimals**, validating the implementation: mean
per-input `cos(C1, C2) = 1.0000`, min = 0.9999998. The two ablation paths
compute the same fixed-point equation `C* = blocks(C*)` from different
engineering directions and reach the same result.

## C1 vs C2 cross-check (engineering sanity)

| pair | mean per-input cos | min per-input cos |
| --- | ---: | ---: |
| trained C1 vs C2 | 1.0000 | 0.9999998 |
| random C1 vs C2  | 1.0000 | 0.9999999 |

C1 and C2 are mathematically equivalent (both implement
`C* = blocks(C*)`) and behave identically. Per the plan §6.2 belt-and-suspenders
rationale, this rules out engineering bugs in either variant.

## Trained model — what changes vs Phase 1 mode A

| metric | Phase 1 mode A | C1 cancelpos | C2 posfree |
| --- | ---: | ---: | ---: |
| n converged / 60 | 59 | **60** | **60** |
| mean steps to convergence | 10.0 | 10.27 | 10.27 |
| effective rank | 1.02 | 1.002 | 1.002 |
| ‖C\*‖ | 1812 | 1795 | 1795 |
| mean off-diag pairwise cos | 0.996 | 0.991 | 0.991 |
| min off-diag pairwise cos | 0.880 | 0.720 | 0.720 |
| mean cos(ablation C\*, mode-A C\*) | — | **0.9985** | **0.9985** |
| min cos(ablation C\*, mode-A C\*) | — | 0.908 | 0.908 |

Reading: removing `wpe` injection has measurable but small effects on the
trained model's mode-A fixed point.

- The fixed point itself is *almost the same point* (0.9985 cos to baseline).
- Convergence is slightly *easier* (60/60 instead of 59/60) — without `wpe`
  pulling on the state each step, the structured-text outlier that took 100+
  steps in Phase 1 also reaches the threshold within 100 steps.
- Effective rank drops further (1.02 → 1.002) — a slightly *cleaner* rank-1
  collapse.
- Pairwise cosine drops from 0.996 to 0.991, with `min` dropping from 0.88
  to 0.72. Position embedding was contributing a small homogenising
  component; without it, a few inputs deviate slightly more from the others.
  But the structure is still essentially "single universal attractor".

So Explanation 3 from plan §1.1 — "engineering artefact: repeated wpe
injection causes the collapse" — is **substantially weakened**. The contraction
to a near-rank-1 attractor must come from elsewhere — most plausibly the
LayerNorm at every block input, which clamps `‖h‖` independently of any
position structure. The Phase 1.1 plan §6.2 explicitly anticipated this
possibility; this experiment is the data that supports it.

## Random init — surprising asymmetry

| metric | Phase 1 mode A | C1 cancelpos | C2 posfree |
| --- | ---: | ---: | ---: |
| n converged / 60 | 0 | 0 | 0 |
| effective rank | ~7 | 1.12 | 1.12 |
| ‖C\*‖ | 157 | 156.8 | 156.8 |
| mean off-diag pairwise cos | 0.72 | **0.54** | **0.54** |
| min off-diag pairwise cos | 0.23 | 0.07 | 0.07 |

Removing `wpe` from the random transformer produces a different surprise:
states become *less* uniform (off-diag 0.72 → 0.54) but *more* concentrated
within each input (eff rank 7 → 1.12). The wpe injection in Phase 1 was
*holding the random states in a 7-dim subspace*; without it, each input
falls into a rank-1 contractive direction selected by its own initial
state, and those directions disagree across inputs.

This is interesting in itself but tangential to the main question.

## Implication for Experiment B's basin structure

Experiment B revealed at least two strong basins (Capital vs lowercase) on
the trained model. Experiment C now adds: **those basins persist when wpe
is ablated**. Position embedding is not the discriminator that selects which
basin an input falls into. The basin structure is a property of the trained
blocks, not of the position embedding.

(This experiment ran on Phase 1's 60-input zoo — not the Experiment B
nested zoo. Per the hard ordering, we keep variant tests within their
designated input scope. A future probe could re-run the nested zoo through
C1 to confirm the Capital-vs-lowercase basin survives wpe ablation. Cost:
~30 s. Recommended as a small Experiment C+B follow-up but explicitly
deferred to Phase 1.2.)

## Caveats

- Mode A's `wpe` injection happens *inside* `model.transformer.forward`.
  Cancelling it via subtraction (C1) and skipping it via manual block
  iteration (C2) are equivalent only at numerical precision. Their
  agreement to 7 decimals confirms this.
- The Phase 1 mode-A baseline used GPT-2's standard forward, which also
  applies `ln_f` once at the end. C1 routes through the same forward; C2
  applies `ln_f` manually inside the loop. The fact that they still agree
  rules out a `ln_f` placement bug.
- The 0.908 minimum cos to baseline (one input's ablation C\* vs its
  mode-A C\*) is worth a follow-up look but doesn't materially affect
  the conclusion. Likely the structured outlier that took >100 steps to
  converge under mode A.
- Random-init twin's effective-rank-collapse-without-wpe finding is a
  side effect, not a Q2 answer. Worth recording as background for the
  master report's caveats section.

## Pointers

- raw traces: `data/raw/phase1_1_position_ablation/{trained,random}/{C1_cancelpos,C2_posfree}/trace_*.pt`
- per-trace records: `data/processed/phase1_1_position_ablation/records.json`
- per-variant similarity matrices: `data/processed/phase1_1_position_ablation/similarity_*.npz`
- aggregate summary: `data/processed/phase1_1_position_ablation/summary.json`
- figures: `outputs/figures/phase1_1_position_ablation/`
- source: `src/posfree_iterate.py`, `src/run_position_ablation.py`

## Conclusion for the master report

**Q2 answered: position embedding is NOT the cause of the trained-model
universal-attractor collapse.** Cancelling repeated `wpe` injection (via
either C1 subtraction or C2 manual block-only forward) leaves the
trained-model fixed point essentially unchanged: cos to mode-A baseline
0.9985 (mean), 0.908 (min); effective rank 1.002 (down from 1.02);
pairwise cos within the ablation set 0.991 (down from 0.996).

Combined with Experiment B's basin-structure finding: the multiple
attractors that the trained model has are real properties of its
blocks, not position-embedding artefacts. Removing wpe doesn't merge
the basins.

This pushes the load-bearing engineering culprit toward **LayerNorm**.
Plan §3.6 lists "LayerNorm Lipschitz/Jacobian diagnostic" as a deferred
follow-up; Experiment C's negative result on wpe is what makes that
diagnostic now the natural next step *after* Experiment D.
