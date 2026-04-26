# Phase 1.2 — Experiment G: B × C joint (basin × position ablation)

- model: `gpt2`, fp32, CUDA
- input zoo: Phase 1.1 Experiment B's 30 nested-structure inputs (seq_len = 64)
- iteration: Variant **C1 cancel-pos** (subtract `wpe[:seq_len]` before each forward)
- max_iter = 1000, threshold = 1e-3, save_every = 10
- elapsed: 4.6 s

## Headline result — capital basin disappears under cancel-pos

| basin | B (mode-A, with wpe) | G (cancel-pos, no wpe net) | transition counts B → G |
| --- | ---: | ---: | --- |
| **capital** (cos < 0.7) | **10** | **0** | capital → lowercase: 6, capital → hybrid: 4, capital → capital: **0** |
| lowercase (cos > 0.999) | 8 | 25 | lowercase → lowercase: 8 |
| hybrid (0.7 ≤ cos ≤ 0.999) | 12 | 5 | hybrid → lowercase: 11, hybrid → hybrid: 1 |

Cancelling the repeated `wpe` injection **completely eliminates the capital
basin**. Every one of the 10 inputs that landed in the capital attractor
under mode A is pulled to either the lowercase basin (6) or to a residual
weak hybrid attractor at cos ≈ 0.72 (4). No input remains at the original
capital attractor (cos ≈ 0.49 from Phase 1.1 B).

Lowercase-basin inputs are unaffected (all 8 stay). Hybrid-basin inputs
mostly collapse into lowercase (11/12); one stays in hybrid.

**Mean cos to B baseline = 0.881; min = 0.502.** The mean is dragged down
by the capital-basin inputs whose entire C\* changed substantially.
Lowercase- and most hybrid-basin inputs have cos ≈ 0.97–1.00 to their B
baseline (no real change), exactly mirroring Phase 1.1 C's finding on the
natural-text zoo.

## Per-input table

| idx | category | depth | n_steps | eff_rank | ‖C*‖ | cos_to_B | cos_to_phase1u | basin under G |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | html | 1 | 13 | 1.22 | 2537 | 0.502 | 0.9998 | lowercase |
| 1 | html | 3 | 12 | 1.22 | 2537 | 0.502 | 0.9998 | lowercase |
| 2 | html | 4 | 12 | 1.22 | 2537 | 0.502 | 0.9998 | lowercase |
| 3 | html | 4 | 40 | 1.00 | 1222 | 0.951 | **0.7203** | hybrid |
| 4 | html | 5 | 13 | 1.22 | 2537 | 0.502 | 0.9998 | lowercase |
| 5 | xml | 2 | 24 | 1.00 | 1221 | 0.951 | **0.7201** | hybrid |
| 6 | xml | 4 | 15 | 1.22 | 2537 | 0.502 | 0.9998 | lowercase |
| 7 | xml | 4 | 24 | 1.00 | 1221 | 0.951 | **0.7201** | hybrid |
| 8 | xml | 6 | 16 | 1.00 | 1230 | 0.948 | **0.7263** | hybrid |
| 9 | xml | 4 | 10 | 1.00 | 2561 | 1.000 | 1.0000 | lowercase |
| 10–14 | json | — | 13–22 | 1.22–1.40 | 2491–2537 | 0.965–0.977 | 0.999 | lowercase |
| 15 | latex | 1 | 11 | 1.22 | 2538 | 0.996 | 0.9998 | lowercase |
| 16 | latex | 3 | 11 | 1.22 | 2539 | 0.996 | 0.9998 | lowercase |
| 17 | latex | 2 | 17 | 1.22 | 2536 | 0.929 | 0.9998 | lowercase |
| **18** | latex | 3 | 16 | 1.22 | 2536 | **0.580** | 0.9998 | **lowercase** (was capital in B) |
| 19 | latex | 3 | 16 | 1.22 | 2536 | 0.922 | 0.9998 | lowercase |
| 20–24 | pure_brackets | 1–5 | 14 | 1.22 | 2536 | 0.974 | 0.9998 | lowercase |
| 25–27, 29 | markdown_list | 1–6 | 22 | 1.22 | 2536 | 1.000 | 0.9998 | lowercase |
| 28 | markdown_list | 3 | 9 | 1.00 | 2560 | 1.000 | 1.0000 | lowercase |

Key observations:
- The original "capital basin attractor" at cos = 0.49, ‖C\*‖ = 1042 (Phase 1.1 B)
  no longer exists. The closest replacement is at cos = 0.72, ‖C\*‖ = 1221, which
  4 of the original capital inputs reach.
- Even within the original capital category, some inputs (0, 1, 2, 4 — html;
  6 — xml; 18 — latex) jump straight to lowercase basin after wpe cancellation.
- Effective rank under cancel-pos is uniformly 1.0 or 1.22 — same rank-1-ish
  attractor structure, just at slightly different points.

## Reading

**Q4 answered: the capital basin is a position-embedding artefact.** When
repeated `wpe` injection is cancelled, the trained model loses the capital
attractor entirely. Position embedding is required to maintain the
markup/sentence-start register pull.

**This is a major reversal of Phase 1.1 Experiment C's headline.** Phase
1.1 C concluded "position embedding is not the homogeniser" based on the
Phase 1 natural-text 60-input zoo, where cos to baseline was 0.998 after
ablation. The conclusion was correct *for that zoo* — those inputs were
all in the lowercase basin, and the lowercase basin is wpe-invariant.
Phase 1.2 G shows the picture is asymmetric across basins:

| basin | wpe-dependent? |
| --- | --- |
| lowercase (cos ≈ 1 to phase1 universal) | **no** — survives cancel-pos at cos ≈ 0.97–1.00 to baseline |
| capital (cos < 0.7) | **yes** — completely disappears under cancel-pos |
| hybrid (intermediate) | mixed — most collapse to lowercase, some stay |

So the multi-basin finding from Phase 1.1 B is *partially* an architectural
property of the trained blocks (lowercase is real) and *partially* a wpe
interaction (capital exists only with continuous wpe injection).

## Interaction with Experiment F (LayerNorm contraction)

Experiment F showed the trained model's contraction comes from LayerNorm
(σ_max ≈ 0.12 at fixed point). Experiment G shows that without wpe
injection, the system collapses to the same LayerNorm-defined attractor
that Phase 1's lowercase basin sits at. Together:

- The **single architectural attractor** the trained blocks define is the
  lowercase / unigram-prior fixed point.
- Position embedding **provides a counter-pressure** that pulls some inputs
  (those whose token sequences interact strongly with `wpe`) into a
  secondary "capital" attractor.
- The capital attractor exists only because wpe re-injection fights the
  LayerNorm contraction at every step.

This is a coherent mechanistic picture but it **shrinks the multi-basin
finding from Phase 1.1 B to one architectural attractor + one
position-driven secondary attractor**, rather than a small manifold of
genuine attractors.

## Implications for the Phase 1.1 master report's claims

The Phase 1.1 master report wrote:

> "The §4 hypothesis is neither cleanly confirmed nor refuted: training
> does produce attractor structure, but the structure corresponds to
> vocabulary register, not to per-input semantic interpretation."

Phase 1.2 G refines this: the *register-typed attractor structure* is
itself partly position-driven, not entirely a learned-blocks property.
The trained-blocks architectural attractor is the lowercase basin alone.
The capital basin is a learned-but-position-dependent secondary
attractor.

For the §4 essay revision: this **further weakens** the Phase 1.1 reading
that "training produces a small manifold of genuine attractors". The
manifold now has one main basin and a wpe-dependent satellite.

## Caveats

- 30 inputs are still small. The 4 inputs that resist cancel-pos at
  cos ≈ 0.72 to phase1u (idx 3, 5, 7, 8) are an interesting subgroup —
  three are XML, one is HTML. What makes these specific inputs resist
  the merge is a sub-question worth investigating. Hypothesis: their
  token sequences have particular `wpe`-interaction patterns. A
  10-line follow-up that runs the same trace with `wpe` zeroed (Variant
  C2 posfree) on these 4 inputs would tell us whether the residual
  hybrid attractor is wpe-driven or block-driven.
- "Cancel-pos" does not literally remove `wpe`'s effect on the *first*
  forward pass — `h_0` is still computed via the standard wpe-included
  path. Iterations 1+ are wpe-net-zero. So the comparison is not
  "with-wpe vs without-wpe initially" but "with continuously-injected-wpe
  vs single-wpe-injection-then-cancelled". Interpretation should match.
- Mean cos to B baseline 0.881 is dominated by the 10 capital-basin
  inputs that moved a lot. For lowercase- and hybrid-basin inputs
  separately, mean cos is ≈ 0.97 (no change). The aggregate number
  hides this asymmetry; the per-input table shows it cleanly.

## Pointers

- raw traces: `data/raw/phase1_2_basin_position_ablation/trace_*.pt`
- per-input records: `data/processed/phase1_2_basin_position_ablation/records.json`
- 30×30 cos similarity: `data/processed/phase1_2_basin_position_ablation/similarity.npz`
- aggregate summary: `data/processed/phase1_2_basin_position_ablation/summary.json`
- figures: `outputs/figures/phase1_2_basin_position_ablation/cancelpos_vs_b_per_input.png`
- source: `src/run_basin_position_ablation.py`

## Conclusion for the master report

**Q4 answered: the capital basin is wpe-dependent; the lowercase basin is
wpe-invariant.** Cancelling repeated `wpe` injection eliminates 100% of
capital-basin outcomes (10 → 0) and substantially shrinks the hybrid
basin (12 → 5). The lowercase basin, by contrast, is preserved across
ablation (8 → 8 plus migration in).

This refines the Phase 1.1 multi-basin finding: the trained blocks define
**one** architectural attractor (lowercase / unigram prior). Position
embedding interacts with markup-style inputs to produce a **secondary**
attractor (capital) that disappears when wpe is no longer being
continuously re-injected.

For the master report: this changes the "small attractor manifold by
register" framing into "one architectural attractor + one wpe-induced
secondary attractor". Closer to Phase 1's original "single universal
attractor" reading, but with the qualification that wpe interaction can
push some inputs to a secondary basin during normal mode-A iteration.
