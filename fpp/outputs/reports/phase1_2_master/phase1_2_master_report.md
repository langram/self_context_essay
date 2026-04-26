# FPP Phase 1.2 — Master Report

**Project:** FixedPointProbe (FPP)
**Phase:** 1.2 — basin-selector and contraction-source diagnosis
**Date:** 2026-04-26
**Plan:** `docs/FPP_experiment_plan_v0_3.md`
**Source commits:** plan `628b000`, E `66837a5`, F `b9f47bd`, G `1d2229c`, H `31d3098`

---

## TL;DR

Phase 1.2 was designed to convert Phase 1.1's three remaining ⚠️s (residual
probe-signal nature, basin selector identity, contraction source localisation)
into ✅s. All three ✅. Plus the cross-cutting Q4 (basin × position
ablation) ✅. The four sub-experiments produce a strikingly coherent
mechanistic account of GPT-2 small mode-A iteration:

1. **One architectural attractor.** Trained LayerNorms have spectral
   norm ≈ 0.12–0.20 at the fixed point (vs random-init's flat 1.0).
   Training has driven LayerNorm gain values such that LayerNorm
   contracts. This is the contraction source. (Experiment F)
2. **One position-driven secondary attractor.** Without continuous
   `wpe` re-injection (Experiment C1 cancel-pos), 100% of capital-basin
   inputs migrate out — the secondary attractor disappears. (Experiment G)
3. **Markup is the trigger.** In a factorial 24-input zoo, markup
   presence (`<...>` tokens) flips basin 12/12 times; case, punct
   density, and content type each flip basin 0 times. The "Capital
   basin" is misnamed: it's a *markup-induced* basin whose LM-head
   projection happens to be capitalised tokens. (Experiment H)
4. **Step-10 hidden state encodes basin label cleanly.** Probe-basin
   accuracy 0.90 vs chance 0.40. Within-basin sub-information varies
   by basin: 1.00 in the rank-1.5 hybrid basin, 0.80 in capital,
   ≈ chance in lowercase. (Experiment E)

The picture is mechanistically clean: GPT-2 small mode-A iteration has
**one architectural attractor + one position-embedding-induced
secondary attractor selected by angle-bracket tokens**. The §4 hypothesis
of "rich attractor structure as a locus of emergence" is **not vindicated**
on this architecture. What looked like multiple basins in Phase 1.1 has
narrowed to one architectural fixed point with a wpe-induced satellite.

The plan §11 meta-warning applies: this is a suspiciously coherent
account. Caveats and a suggested triangulation experiment are in §6.

---

## Per-question answers

### Q1 — what is the step-10 transient probe residue?

**Answer (Experiment E): basin-label-dominated, with sub-basin structure
that varies sharply by basin.**

| probe (LOO CV, n=30) | accuracy ± std | chance | accuracy / chance |
| --- | ---: | ---: | ---: |
| probe-basin (3-way) | 0.900 ± 0.300 | 0.400 | 2.25× |
| probe-fine global (6-way) | 0.833 ± 0.373 | 0.167 | 4.99× |
| probe-fine within capital | 0.800 ± 0.400 | 0.500 | 1.60× |
| probe-fine within hybrid | 1.000 ± 0.000 | 0.417 | 2.40× |
| probe-fine within lowercase | 0.750 ± 0.433 | 0.625 | 1.20× |

The Phase 1.1 A 0.40 step-10 accuracy is now decomposed: most of it is
basin label (probe-basin 0.90), with additional sub-basin structure
that exists in the hybrid basin (probe-fine within hybrid 1.00) but
barely in the lowercase basin (1.2× chance) and modestly in the
capital basin (1.6× chance).

The strong-rank-1 basins (capital ≈ 1.11, lowercase ≈ 1.02) are
effectively semantic dead zones at the fixed point. The
intermediate-rank (1.5) hybrid basin retains per-category structure.

### Q2 — what factor selects basin?

**Answer (Experiment H): markup presence (`<...>`) is the sole selector
among the four factors tested.**

| factor | within-factor-flip pairs | basin-flip pairs | flip rate |
| --- | ---: | ---: | ---: |
| case | 12 | 0 | 0.000 |
| **markup** | **12** | **12** | **1.000** |
| punct | 12 | 0 | 0.000 |
| content | 24 | 0 | 0.000 |

The 12 with-markup inputs all converge to `‖C*‖=1042, cos=0.50 to phase1u,
rank 1.11, n_steps=129`. The 12 without-markup inputs all converge to
`‖C*‖=2563, cos=1.00, rank 1.02, n_steps=10`. **Within-class C\*'s are
pairwise identical to floating-point precision** — once an input lands
in a basin, all inputs in that basin reach literally the same point.

The Phase 1.1 B "Capital basin" naming was misleading: the LM-head
projects the basin to capital tokens because the attractor sits in a
region that maps to capitals, not because inputs are capitalised.

### Q3 — what is the contraction source?

**Answer (Experiment F): trained LayerNorm — `σ_max(ln_1) = 0.12`,
`σ_max(ln_2) = 0.20`, `σ_max(ln_f) = 2.5` at the fixed point — vs
random-init's flat `σ_max = 1.00` for all LayerNorms.**

| sub-module (layer 6) | trained at h_fixed | random at h_fixed |
| --- | ---: | ---: |
| ln_1 | **0.12** | 1.00 |
| attn-sublayer | 7.4 | 1.16 |
| ln_2 | **0.20** | 1.00 |
| mlp-sublayer | 2.7 | 1.11 |
| ln_f | 2.5 | 1.00 |
| full block | 7.5 | 1.19 |
| full posfree stack | 16.3 | 1.26 |

Training has driven LayerNorm gain values to small magnitudes that
contract; the LayerNorm formula itself is norm-preserving without
training (random init's σ_max = 1.00 exactly). Attention and MLP
sub-modules are *expansive* in spectral norm (σ_max > 1) but operate
along the attractor manifold; the rank-reducing flow comes from
LayerNorm contracting orthogonal directions while attn/mlp preserve
the attractor direction.

(σ_max > 1 at the fixed point with empirical convergence is reconciled
by the spectral-norm vs spectral-radius distinction — see Experiment F
report. The directions where σ_max is large are along the rank-1
attractor; convergence is in the orthogonal complement.)

### Q4 — does the basin survive position-embedding ablation?

**Answer (Experiment G): no, the capital basin is wpe-dependent.
Lowercase basin is wpe-invariant.**

| transition under cancel-pos | n |
| --- | ---: |
| capital → capital | **0** |
| capital → lowercase | 6 |
| capital → hybrid (cos ≈ 0.72) | 4 |
| lowercase → lowercase | 8 |
| hybrid → lowercase | 11 |
| hybrid → hybrid | 1 |

All 10 capital-basin inputs from Phase 1.1 B migrate out under
cancel-pos. Six collapse fully into lowercase; four end up at a
*weakened* secondary attractor at cos ≈ 0.72 (still distinct from
lowercase but not the original cos = 0.49 capital fixed point). The
8 lowercase-basin inputs are unchanged.

This **directly contradicts** Phase 1.1 Experiment C's headline
("position embedding is not the contraction source"). The earlier
finding was correct *for the natural-text 60-input zoo* — those inputs
were all in the lowercase basin, which is wpe-invariant. With Phase
1.1 B's nested zoo (which spans both basins), cancel-pos eliminates
the secondary basin almost entirely.

---

## Re-verdict on the four alternative explanations from Phase 1.1

| explanation | Phase 1.1 verdict | Phase 1.2 evidence | Phase 1.2 verdict |
| --- | --- | --- | --- |
| E1 — FPP wrong on GPT-2 | weakened | Q3 confirms trained-blocks contraction is real and trained; Q1 shows fixed point is not informationless | **partially supported in spirit**: there is structure, but not the per-input semantic structure §4 predicted |
| E2 — Interface mismatch | refuted | (no new test; reading unchanged) | refuted |
| E3a — Position embedding artefact | refuted | Q4 reverses this for capital basin: it IS a wpe artefact | **partially supported** for the secondary basin only |
| E3b — LayerNorm contraction | elevated by elimination | Q3 confirms directly: trained LN σ_max ≈ 0.12 at fixed point vs random 1.00 | **confirmed** |
| E4 — Semantics in transient | partially supported | Q1 shows residue is mostly basin label; per-input semantics minimal at fixed point | **weakened** — residue is mostly a categorical (basin) signal, not a graded semantic signal |

**Cleanest one-paragraph reading.** Trained GPT-2 small under mode-A
iteration has a *single architectural fixed point* (the lowercase /
unigram-prior attractor, rank 1.02, ‖C\*‖ ≈ 2563). Position-embedding
re-injection at every iteration step interacts with angle-bracket
tokens to maintain a *secondary attractor* (rank 1.11, ‖C\*‖ ≈ 1042).
The contraction source is *learned LayerNorm gain values* (σ_max ≈ 0.12).
Hidden state at step 10 carries clean basin-label information plus
modest sub-basin structure. The §4 hypothesis of per-input semantic
attractors is **not** what is happening on this architecture.

---

## The "everything explains itself" warning (per plan §11)

The Phase 1.2 picture is suspiciously coherent. Four independent
diagnostics each confirm or refine the previous. Per the plan §11
discipline, this should trigger explicit questions about what is
being missed.

Three concrete confounds that have not been addressed by Phase 1.2:

1. **Markup vs angle brackets specifically.** Experiment H's markup
   factor is exclusively `<...>`. We have not tested whether `[...]`,
   `{...}`, `"..."`, or other "tag-like" markup tokens trigger the
   same secondary basin. **The data so far is consistent with both
   "markup as a category triggers the basin" and "specifically
   angle-bracket BPE tokens trigger the basin"**, and these are very
   different mechanistic claims.

2. **Hand-constructed input distribution.** Phase 1.2 H's 24 inputs
   were constructed by hand for clean factor isolation. They are not
   a representative sample of GPT-2's training distribution. Effects
   that depend on natural co-occurrence patterns (e.g. case + punct
   correlations in real text) cannot be detected by this design.

3. **Single architecture.** Everything is on GPT-2 small (124M).
   Whether the picture (one main basin + wpe-induced satellite + LN
   contraction) generalises to other architectures (Pythia rotary,
   LLaMA RMSNorm + RoPE) is untested. Plan §1.2 explicitly excludes
   scale ladder; Phase 1.3 should reconsider.

A natural Phase 1.3 single-experiment triangulation:

- **Experiment I**: factorial zoo varying *type of markup*: angle
  brackets, square brackets, parentheses, braces, quotes — five
  variants under otherwise identical content. Tests whether the
  basin selector is "markup as a category" or specifically the
  angle-bracket BPE token IDs. Cost: ≈ 1 hour total. Discriminates
  the most concerning confound.

---

## Concrete essay-revision recommendations

Per plan §10 the Phase 1.1 §7 "Initial Probe Update" subsection needs
to be re-updated for Phase 1.2.

### §7 — replacement subsection (~250 words)

> *Update from initial probes (FPP project, Phase 1 + 1.1 + 1.2,
> 2026-04).* Three rounds of empirical probes on a frozen pretrained
> GPT-2 small have produced a coherent mechanistic account of mode-A
> hidden-state self-iteration on this architecture.
>
> The trained model has **one architectural attractor**: a near-
> rank-1 fixed point at which the hidden state projects through the
> LM head to the unigram prior ( the / , / \\n / and / .). The
> contraction toward this attractor comes from *learned LayerNorm
> gain values*; the spectral norm of each pre-block LayerNorm at the
> fixed point is 0.12–0.20, compared to 1.0 for randomly-initialised
> LayerNorms. Training is the mechanism that makes LayerNorm
> contractive; the formula itself is norm-preserving without
> training. Attention and MLP sub-modules are *expansive* in
> spectral norm but operate along the attractor direction; the
> rank-reducing flow comes from LayerNorm contracting orthogonal
> directions.
>
> A **secondary attractor** exists for inputs containing angle-bracket
> tokens (`<...>`), driven by repeated position-embedding injection
> at every iteration step. Cancelling the wpe re-injection collapses
> 100% of the secondary-basin inputs back into the architectural
> attractor. Surface input case, punctuation density, and content
> type have no effect on basin assignment in a 24-input factorial
> design; only markup presence does.
>
> The §4 prediction of per-input semantic attractors is therefore
> **not supported** on GPT-2 small mode-A iteration. The trained
> blocks define one architectural attractor; the hidden state at
> step 10 carries basin-label information cleanly but per-input
> sub-basin information is small and varies sharply by basin. Full
> data, code, and reports at `github.com/langram/self_context_essay/tree/main/fpp`.

### §4 — additional softening per plan §10.2

The Phase 1.2 F finding that attention/MLP sub-modules are
*expansive* (σ_max > 1) while LayerNorm is the contractive component
is mechanistically informative. The original §4 wording

> *(context, attention) coupled fixed point*

should become, where the data supports it:

> *(context, attention) coupling under normalisation* — the
> normalisation step (per-block LayerNorm) provides the contraction
> while attention and MLP preserve the attractor manifold. The
> coupling is asymmetric in this sense.

This is a strict refinement of the original framing, not a reversal.

### §4 — what NOT to revise

The argument that self-reference matters for emergence is untouched;
the (context, attention) language is correctly pointing at the right
two components, just incomplete; the claim that current LLMs lack
cross-conversation persistence is unaffected by anything in Phase 1.x.

### §10.3 — about FPP-Native

Phase 1.2 leans toward the "GPT-2 mode-A is mostly architectural
contraction with a wpe-induced satellite" reading. **This does not
license the FPP-Native conclusion.** Mode A is one specific probe;
the §4 hypothesis is about a different formulation (continuous
self-dialogue + memory, not iterated forward passes). Phase 1.2
narrows the *cheap mode-A probe* result, not the *original §4
construct*.

The §10.3 plan said:

> "Phase 1.2 完成后，如果数据显示 'GPT-2 完全坍缩到 register-level prior，
> sub-basin 信息为零' —— essay 可以更明确地把 FPP-Native 列为合理方向。"

The data is closer to that scenario but not exactly that. We have
**partial** sub-basin information (the hybrid basin's 100% LOO
accuracy on N=12); the claim "completely zero" is not supported.

Recommendation: keep FPP-Native as an open candidate, **but do not
elevate it on the basis of Phase 1.2 alone**. The Phase 1.3
triangulation experiment (markup-type factorial) is needed first.

---

## Phase 1.3 candidates

Ranked by cost / discriminating power:

1. **Markup-type factorial** (the Phase 1.2 §6 triangulation
   experiment). 30-input zoo varying angle-brackets vs square-brackets
   vs parens vs braces vs quotes. Tests whether the basin selector
   is "angle brackets specifically" or "any tag-like character".
   Cost: ≈ 1 hour. **Strongly recommended.** This is the single
   experiment most likely to either solidify the Phase 1.2 picture
   or reveal a missing confound.
2. **Cross-architecture replication** (one experiment from the four
   on Pythia 1.4B). Pythia has rotary position embeddings, not
   absolute wpe. If the secondary basin still exists, the basin
   trigger is not specifically wpe — it's some more general
   position-encoding interaction. If the secondary basin disappears
   on Pythia, wpe-specific is confirmed. Cost: ≈ 4 hours.
3. **Layer-by-layer LayerNorm σ_max** (extension of Experiment F).
   We only analysed layer 6. Compute σ_max(ln_1, ln_2) for every
   layer 0..11. Tests whether all blocks contribute to contraction
   equally or some layers do most of the work. Cost: ≈ 2 hours.
4. **Larger natural input zoo** (replacing H's hand-constructed inputs
   with a stratified random sample from a corpus). Tests whether
   the markup-only basin selector survives natural input distribution.
   Cost: ≈ 6 hours including zoo construction.
5. **Mode B (temperature sampling)**. Plan §1.2 deferred. Cheap mode C
   variant. Cost: ≈ 4 hours.

**Recommended Phase 1.3 minimum**: item 1 (markup-type factorial).
**Recommended Phase 1.3 maximum**: items 1 + 2.

The plan §11 discipline matters here — Phase 1.2 is too clean to
trust without one more triangulation. Item 1 is the cheapest
experiment that meaningfully tests the central reading.

---

## What to send to the next AI round

A new package (analogous to the previous two) should be assembled
containing:

- This master report
- The four Phase 1.2 sub-reports
- The Phase 1.1 master + four sub-reports (for cross-phase context)
- All key figures
- Per-experiment processed JSONs and selected raw traces
- Source code snapshot
- Both essay versions + plans v0.1, v0.2, v0.3
- An updated README with the Phase 1.2 question set:
  1. Is the markup-as-sole-selector reading robust, or is it specifically
     about angle-bracket BPE token IDs? Does the recommended Phase 1.3
     triangulation discriminate the right thing?
  2. The (context, attention, normalisation) framing for §4 — does it
     match what the data shows, or is "normalisation" the wrong third
     term?
  3. The σ_max > 1 at fixed point with empirical convergence — is the
     spectral-norm-vs-radius distinction the right way to reconcile
     this, or is there a cleaner mechanistic interpretation?
  4. Phase 1.2 F's attn/mlp expansiveness and ln_1/ln_2 contractiveness
     — what should we read into this? Is it consistent with the
     mechanistic-interpretability literature on transformer block
     dynamics, or new?
  5. The §11 meta-warning. What confound have we missed?

Per plan §2.4 workflow agreement: each AI sees raw data + reports
*before* seeing any other AI's response. The user merges across.

---

## Caveats applying to all of Phase 1.2

- **All on GPT-2 small (124M).** Generalisation to other models untested.
- **All inputs are short** (≤ 64 tokens). Whether the basin structure is
  similar at longer contexts (the natural use of LLMs) untested.
- **All inference-only.** The "training is the mechanism that makes
  LayerNorm contractive" finding is correlational — we observe trained
  vs random gap, not a causal training trajectory study.
- **Plan §11 warned about "everything explains itself" feeling.**
  Phase 1.2 has produced exactly that feeling. The recommended Phase 1.3
  triangulation is the appropriate response.

---

## Pointers

- Plans: `docs/FPP_experiment_plan_v0_{1,2,3}.md`
- Phase 1 baseline: `outputs/reports/phase1_gpt2_small/report.md`
- Phase 1.1 master: `outputs/reports/phase1_1_master/phase1_1_master_report.md`
- Phase 1.2 sub-reports:
  - `outputs/reports/phase1_2_cross_basin_probe/phase1_2_cross_basin_probe.md` (E)
  - `outputs/reports/phase1_2_module_jacobian/phase1_2_module_jacobian.md` (F)
  - `outputs/reports/phase1_2_basin_position_ablation/phase1_2_basin_position_ablation.md` (G)
  - `outputs/reports/phase1_2_factorial_zoo/phase1_2_factorial_zoo.md` (H)
- Per-experiment raw + processed: `data/{raw,processed}/phase1_2_*/`
- Source: `src/{cross_basin_probe, module_jacobian, posfree_iterate,
  run_basin_position_ablation, factorial_zoo, run_factorial_zoo}.py`
- Total wall-clock for Phase 1.2 sub-experiments: ≈ 4 minutes GPU + ≈ 2
  hours analysis & reporting.
