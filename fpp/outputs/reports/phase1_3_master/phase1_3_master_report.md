# FPP Phase 1.3 — Master Report

**Project:** FixedPointProbe (FPP)
**Phase:** 1.3 — triangulation and blind-spot audit
**Date:** 2026-04-27
**Plan:** `docs/FPP_experiment_plan_v0_4.md`
**Source commits:** plan `bb02cec`, I `d9a1d87`, J+K `2f38796`, M `aa8b52e`, L `7be8411`

---

## TL;DR

Phase 1.3 was a triangulation phase. It checked Phase 1.2's "suspiciously
coherent" mechanistic picture against five new experiments. The picture
**both confirmed and partially overturned** Phase 1.2:

**Phase 1.2 was wrong about the secondary attractor's nature.** The
"capital basin" / HTML structure is **not a real fixed point**: tail-rate
fitting (J.3) gives λ ≈ 1.0001 with R² = 0.011. Compare to lowercase's
λ = 0.775, R² = 0.81. The lowercase basin is a true exponential
attractor; the HTML structure is a **wpe-supported marginal
equilibrium** where the iteration stops moving in absolute terms but
exhibits no exponential pull. **Phase 1.2's "two attractors" framing
should be retired.** The 13× convergence-step gap from Phase 1.2 (HTML
129 vs lowercase 10) is the dynamical signature of "attractor vs
non-attractor", not of "two attractors at different speeds".

**Phase 1.2 was also wrong about markup as a category.** Experiment I
shows the secondary regime is **angle-bracket-specific** (5/5 angle
inputs hit cos_html = 1.000), not "markup as a class". Curly braces
and quotes go to lowercase; square brackets and isolated chars produce
**a third intermediate regime** at cos_lc = 0.90, cos_html = 0.61. The
hidden-state geometry has at least three distinguishable endpoint
regions, not two.

**Phase 1.2 was right about LayerNorm-as-contraction-source — at every
layer.** Experiment K confirmed the contraction is distributed across
all 12 layers (σ_max < 0.25 throughout) with two outliers at L0
(σ_max = 0.038) and L1 ln_2 (0.067). Experiment L decomposed the
σ_max ≈ 0.12 figure cleanly into a state-norm contribution (0.046,
universal) and a γ contribution (layer-dependent, L0 = 0.252, L6 = 0.779).
Coupling factor 1.000 exactly: the decomposition is multiplicatively
clean.

**Phase 1.2 F's "asymmetric coupling" survives but softens.**
Experiment M's spectral-radius measurements show LayerNorm has ρ ≈ σ_max
(symmetric) but attention/MLP have ρ << σ_max (asymmetric). At h_fixed,
attention's σ_max = 3–14 corresponds to ρ ≈ 1.4–2.0 — still expansive
but much less than σ_max suggested. The full mode-A iteration map has
**ρ < 1** at h_fixed across all input regimes (lowercase ρ = 0.59,
mixed ρ = 0.85, third-endpoint ρ = 0.91), rigorously justifying Phase
1's empirical convergence.

**The third endpoint is dynamically intermediate.** Square brackets
sit at full-stack ρ = 0.91 — close to but below the marginal stability
boundary. More attracting than HTML structure (J.3 λ ≈ 1) but less
than lowercase (ρ = 0.59).

The plan §11 "everything explains itself" warning was vindicated: Phase
1.2's coherent-looking picture *did* miss the 13× convergence-rate gap,
which Phase 1.3 reframed as the central dynamical fact. The §2 workflow
(independent evaluations + cross-round complementarity) was effective at
recovering this. Whether Phase 1.3's *own* picture has a similar hidden
gap is the question for Phase 1.4.

---

## Per-question answers

### Q1 — markup or angle-bracket specifically?

**Answer (Experiment I): angle-bracket BPE token IDs specifically.** Of
30 inputs across 6 bracket-type categories:

| category | n | n_steps | eff_rank | cos_lc | cos_html |
| --- | ---: | ---: | ---: | ---: | ---: |
| angle | 5 | **129** | 1.11 | 0.503 | **1.000** |
| square | 5 | 21.4 | 1.27 | 0.901 | 0.615 |
| curly | 5 | 10.6 | 1.02 | 1.000 | 0.490 |
| parens | 5 | 50.0 | 1.49 | 0.997 | 0.540 |
| quotes | 5 | 10.8 | 1.02 | 1.000 | 0.490 |
| isolated | 5 | 36.0 | 1.13 | 0.900 | 0.601 |

Only `<...>` reaches the HTML reference exactly. Curly braces and
quotes converge to lowercase. **Square brackets and isolated chars
produce a third intermediate regime.** Phase 1.2 H's "markup is the
sole basin selector" claim was correct *for its narrow zoo* but does
not generalise — the trained model has multiple distinguishable
endpoint regimes by surface bracket type.

### Q2 — γ vs state norm in LayerNorm contraction?

**Answer (Experiment L): multiplicative, both contribute, L-dependent
balance.** σ_max(LN) factorises exactly into:

```
σ_max(trained γ × orig h) = (state-norm factor) × (γ factor)
```

with coupling factor `(A1·A4)/(A2·A3) = 1.000` (3 decimals) for both
L0 and L6.

| layer | state-only (A2/A4) | γ-only (A3/A4) | total (A1/A4) |
| ---: | ---: | ---: | ---: |
| 0 (K outlier) | 0.046 | **0.252** | 0.012 |
| 6 (F baseline) | 0.046 | 0.779 | 0.036 |

State-norm contribution is identical (0.046) — both layers see the
same large fixed-point ‖h‖. **L0's outlier (3× more contractive than
L6) is purely a γ effect** — L0 has aggressively learned γ values that
contract by 4× even at unit-σ_x scale, while L6's γ contributes only
1.3×.

Phase 1.2 F's "training drives γ < 1" was correct in direction but
underspecified. The complete story: training drives **both** (1) the
fixed-point states to large norm (state-norm contribution, universal)
and (2) γ values to small magnitudes, especially at L0.

### Q3 — σ_max vs ρ at sub-modules?

**Answer (Experiment M): LayerNorm symmetric (σ ≈ ρ); attention/MLP
asymmetric (σ ≫ ρ); full iteration map ρ < 1 despite σ ≫ 1.**

| sub-module | σ_max range | ρ range | reading |
| --- | --- | --- | --- |
| ln_1, ln_2, ln_f | 0.07–3.6 | 0.07–3.6 | symmetric — F's contraction claim survives |
| attn_sublayer | 3.2–14.2 | 1.4–2.0 | strongly asymmetric — F overstated |
| mlp_sublayer | 1.7–3.6 | 1.2–1.5 | mildly asymmetric |
| full_block_L6 | 3.5–14.4 | 1.6–2.8 | inherited asymmetry |
| **full_stack_posfree** | **7.7–25.7** | **0.59–0.91** | rigorously contracting, σ misleading |

**Phase 1.2 F's claim that attn/mlp are "expansive in σ_max" survives
but softens**: in ρ they are mildly expansive (1.4–2.0, not 3–14).
The full mode-A iteration map is rigorously contractive (ρ < 1) at
h_fixed for all three input regimes. Phase 1's empirical convergence
is now mechanistically justified.

### Q4 — HTML-induced structure's ontology?

**Answer (Experiment J): a wpe-supported marginal equilibrium, NOT a
real fixed point.**

J.1 (10000-step traces): both basins remain at their states without
drift. No metastable transient.

J.2 (wpe shutoff after capture at step 200, 1000 cancel-pos steps):
all 4 HTML traces drift in unison from cos_lc = 0.508 → 0.720; cos to
h_capture drops 1.000 → 0.957. Slow drift toward lowercase, not
immediate collapse.

J.3 (tail rate fits): **the central finding.**

| trace | tail window | λ | R² |
| --- | --- | ---: | ---: |
| lowercase reps (×4) | 5–50 | **0.775** | **0.81** |
| HTML reps (×4) | 200–800 | **1.0001** | **0.011** |

Lowercase's R² = 0.81 means the trajectory follows clean exponential
decay. HTML's R² = 0.011 means the trajectory does not follow
exponential decay at all — the deltas plateau at ~5e-4 with no
decay rate. **HTML is not exponentially attracting in the rigorous
dynamical-systems sense.**

J.4 (cycle detection): no formal signal. Suggestive but unconfirmed
period-4 oscillation hint in HTML traces (k=4 distance ≈ 10% below
baseline). Below the formal threshold; recorded for Phase 1.4.

**The 13× convergence-step gap (HTML 129 vs lowercase 10) is the
dynamical signature of "attractor + non-attractor", not "two
attractors at different speeds".** This is the most consequential
Phase 1.3 finding: Phase 1.2's "two basins" framing was wrong about
the second.

### Q5 — position × token interactions?

**Partial answer (Experiment I).** Single isolated bracket characters
(`<` alone, `>` alone, `[` alone, `]` alone, `/` alone) do **not**
trigger the HTML structure — they end up at the third intermediate
regime (cos_lc = 0.90, cos_html = 0.60). The full `<...>` *pattern*
is required. So the trigger condition is:

- not just "contains a `<` token"
- not just "contains a `>` token"
- not "contains any tag-like character" (contradicted by curly/quotes
  going to lowercase)
- specifically "contains the `<...>` token-sequence pattern"

The plan §3.2 hypothesised this; the data confirms it. **A complete
position × token analysis would need a follow-up experiment varying
where in the sequence the `<...>` appears (start / middle / end /
multiple).** This is a Phase 1.4 candidate.

---

## Final verdict on the Phase 1.2 mechanistic picture

Phase 1.2's claim list, with Phase 1.3 verdicts:

| Phase 1.2 claim | Phase 1.3 verdict |
| --- | --- |
| "trained model has one architectural attractor + one wpe-induced secondary attractor" | **wrong about the secondary**: it is a marginal equilibrium, not an attractor. Correct: ONE attractor + ONE marginal equilibrium + (newly discovered) one near-marginal third regime |
| "LayerNorm is the contraction source" | **confirmed** at all 12 layers (K) and decomposed into state-norm × γ (L) |
| "asymmetric coupling between LN (contracts) and attn/mlp (expand)" | **survives in direction**; softens in magnitude — attn/mlp are mildly expansive in ρ (1.4–2.0), not 3–14 as σ_max suggested (M) |
| "markup is the sole basin selector" | **narrowed**: angle-brackets specifically. Curly/quotes/parens go to lowercase. Square brackets/isolated chars produce a third regime |
| "training drives γ < 1 to make LN contract" | **refined**: training drives BOTH high-norm fixed points AND reduced γ values; both multiply |
| "Capital basin / HTML structure" | **renamed**: angle-bracket-induced wpe-supported marginal equilibrium |

The Phase 1.2 picture is **majority correct** at the engineering-claim
level (LayerNorm is the contraction source, attn/mlp expand, multi-
endpoint geometry exists) but **wrong at the dynamical-systems level**
about what kind of object the secondary "basin" is.

---

## The 13× convergence-step gap mystery — resolved

The plan §0.2 specifically called out this as a phenomenon both Phase
1.2 evaluators missed. Phase 1.3 J.3 settles it:

- Lowercase basin: λ = 0.775 per step → e-fold time ≈ 4 steps → reaches
  threshold at ≈ 10 steps. Real exponential attractor.
- HTML structure: λ ≈ 1.0001 → no exponential decay → reaches threshold
  by absolute residual plateauing at 5e-4, which happens at ~129 steps
  for reasons that are not exponential-rate-limited.

**The 13× ratio is not an attractor-strength comparison.** It compares
"time to reach a threshold via exponential decay" with "time to reach
the same threshold via residual saturation". These are different
mechanisms.

The §2 workflow was *partially* effective: Claude noticed the 13× gap
in cross-round commentary on Phase 1.2; ChatGPT then expanded it into
the J experimental design. **Both evaluators independently missed it
in initial evaluations.** The third-step (cross-round commentary) is
what recovered it. This validates the workflow but also exposes its
fragility — without that third step, the gap would have remained
unflagged.

---

## Plan §11 meta-discipline — does Phase 1.3's picture have its own hidden gap?

The plan §11 specifically warned that Phase 1.3's experiments would
produce "everything explains itself" feeling. Looking at the actual
data:

**The picture is again coherent.** I says angle-brackets specifically;
J says HTML is a marginal equilibrium; K says all 12 layers contract;
L says state-norm + γ multiply cleanly; M says ρ < 1 confirms
contraction. Each finding refines or supports the others.

**Specific candidates for Phase 1.3's own blind spots:**

1. **The third-endpoint regime is characterised but not diagnosed.**
   M says ρ = 0.91 (near-marginal). I says cos_lc = 0.90, cos_html =
   0.61. Neither tells us *what* it is — fixed point, metastable
   transient, or yet another wpe-supported marginal? A J-style long
   trace + WPE shutoff on square-bracket inputs is the natural follow-up.
   This is the most likely "next 13× gap" candidate.
2. **The "wpe-supported marginal equilibrium" framing is informal.**
   Dynamical systems theory has formal terms for related objects
   (saddle-node ghost, slow manifold, neutrally-stable equilibrium).
   We have not pinned down which one, if any, exactly fits. A cleaner
   theoretical framing might reveal that what looks like "marginal"
   in our 10000-step traces is actually a slow centre-manifold drift
   that would be noticed in 100000-step traces. **Plan §6.2 warned
   not to early-stop**; we did run 10000 steps. But going to 100000
   would test whether "no drift in 10000" is "no drift forever" or
   "drift slower than we measured".
3. **All Phase 1.3 experiments use the same 10–15 inputs.** Many of
   the contradictions and refinements would not show up in a smaller
   or larger or differently-distributed input zoo. Robustness to zoo
   choice is unverified.
4. **L's γ histogram claim is operational, not direct.** L says
   "L0 has aggressive γ" via σ_max (= γ_max effect at unit σ_x). A
   direct read of γ values per dimension and per layer would be a
   simple sanity check.
5. **The third-endpoint regime's full-stack ρ = 0.91 is at the edge
   of what M can resolve.** Arnoldi with `ncv = 10` may have
   under-estimated by a few percent. A more careful M re-run on
   the third-endpoint inputs alone with `ncv = 50` would give a
   sharper figure — relevant for whether ρ_third = 0.91 vs 0.95
   matters dynamically.

**Most-load-bearing of these for Phase 1.4: items 1 and 5.** Both
are about the third-endpoint regime, which Phase 1.3 introduced but
did not fully diagnose.

---

## Phase 1.4 candidates

**Group Cheap (≤ 1 day):**

1. **J-style long trace + WPE shutoff on square-bracket inputs**
   (item 1 above). Tells us whether the third-endpoint regime is a
   true fixed point (J.1 stable + J.2 stable), another wpe-marginal
   equilibrium (J.1 stable + J.2 drifts), or something else.
2. **HTML-only ρ measurement** (M's phase1_1b mixed bag isolated to
   HTML inputs). Tests whether HTML structure's ρ ≈ 1.0 (consistent
   with J.3 λ ≈ 1) or noticeably below 1.
3. **Direct γ histogram per layer**. Sanity-check L's "L0 has
   aggressive γ" claim.

**Group Medium (2–3 days):**

4. **Position × token analysis on `<...>`** (Q5 follow-up). Vary
   bracket position in sequence to test whether the trigger is
   pattern-specific or position-specific.
5. **Sharper M on third-endpoint** with `ncv = 50` Arnoldi.
6. **L expansion to ln_2 of all 12 layers**. Tests whether the
   state-norm contribution is layer-uniform (likely yes) and the
   γ contribution layer-varying (likely yes by symmetry with K).

**Group Expensive (deferred to Phase 2 unless Phase 1.4 forces it):**

7. **Cross-architecture replication** (Pythia rotary, LLaMA RMSNorm).
   Plan still excludes; deferring.
8. **Mode B (temperature sampling)**. Plan still excludes.

**Recommended Phase 1.4 minimum**: items 1 + 2. They settle the
"third-endpoint regime" provenance and the HTML-only ρ test, both of
which are direct gaps in Phase 1.3's picture.

**Recommended Phase 1.4 maximum**: items 1 + 2 + 3 + 4. Item 3 is
trivially cheap. Item 4 closes Q5.

---

## Essay revision points (for Phase 1.x summary, NOT this round)

Per plan §10, **Phase 1.3 does not revise the essay**. Below are the
revision points that should go into the eventual Phase 1.x summary
revision.

### §4 revisions

- The "(context, attention) coupling" formulation in Phase 1.2 was
  proposed to expand into "(context, attention) coupling under
  normalisation". Phase 1.3 sharpens this further: at the fixed point,
  **LayerNorm is the contraction; attention/MLP have ρ ≈ 1.4–2.0
  (mildly expansive in radius)**. The asymmetric-coupling framing
  survives but at softer magnitudes than σ_max suggested.
- The §4 "context-attention coupled fixed point" is **only fully
  realised in the lowercase basin**. The HTML structure is a
  wpe-supported marginal equilibrium, not a fixed point. So the §4
  framing applies to ONE attractor, not "all stable points of mode-A
  iteration".
- The σ_max > 1 + empirical-convergence apparent paradox from Phase
  1.2 is now formally resolved: ρ < 1 across all sub-modules at the
  full-stack level. The dynamics-stability claim ρ < 1 is the right
  thing to use, not σ_max.

### §7 revisions

- The "Initial Probe Update" subsection drafted in Phase 1.2's master
  report needs another revision. Phase 1.3 narrows the "multi-basin"
  claim to "one architectural attractor + one wpe-marginal equilibrium
  + one near-marginal third regime", and corrects the "markup as
  category" to "angle-brackets specifically".
- The §4 hypothesis ("per-input semantic attractors") is now even
  more clearly **not supported** on GPT-2 small mode-A. The trained
  model has one real attractor (the unigram-prior fixed point); other
  endpoint regimes are non-attracting. Per-input semantic information
  is mostly absorbed by the contraction.

### §10 revisions

- Recursive footnote framing should be softened in light of Phase 1.3.
  The "article generates itself by the mechanism it describes" claim
  is metaphorical; the data on GPT-2 mode-A doesn't support a literal
  reading where transformer self-iteration produces semantic-grade
  fixed points.

### What to NOT revise

- The argument that self-reference matters for emergence is untouched.
- The diagnosis that current LLMs lack cross-conversation persistence
  is untouched.
- The (context, attention) language is correctly pointing at the right
  components; Phase 1.3 just refines the third component (normalisation
  contracts, attn/mlp mildly expand).

---

## Workflow self-evaluation

Plan §11 asked Phase 1.3 to evaluate the workflow's effectiveness.

**What worked well:**

1. The §2 "independent evaluations precede cross-checking" workflow
   recovered the 13× gap that single evaluations missed. Without that
   step, Phase 1.3 would not have happened.
2. Cross-round complementarity (Claude's 13× observation + ChatGPT's
   subsequent expansion into J's design) produced an experiment design
   neither AI alone would have produced.
3. The plan §0.1 naming discipline ("HTML-induced metastable
   structure / wpe-supported candidate" instead of "capital basin")
   prevented further concept-pollution. Phase 1.3 finds that name was
   correct.
4. The user's Group-Cheap-then-stop reframing of M (M does NOT verify
   J; M solves separate methodological questions) was an important
   correction. Without it, M would have been framed as "settling J's
   verdict" when in fact J's R² = 0.011 is not something M can revise.
5. The user's expansion of M to include the third-endpoint regime
   produced one of Phase 1.3's most novel data: ρ = 0.91 for square
   brackets, characterising a regime that neither Phase 1.2 nor the
   plan's pre-execution design directly addressed.

**What may still be missing:**

1. **Two AI evaluators are not adversarial enough.** Both Claude and
   ChatGPT shared a Phase 1.2 "two-basin" frame and missed the 13×
   gap. A third evaluator using a different framework (maybe a non-
   LLM tool like a dynamical-systems textbook check) would be more
   robust.
2. **The third-endpoint regime in I is itself a Phase 1.3 finding
   that almost slipped past first reporting.** Group Cheap reported
   it clearly in I but the user had to push back to ensure it was
   foregrounded in the Group Expensive design (M expansion). This
   is the same failure mode as the 13× gap. The fix would be: any
   experiment finding that doesn't fit the existing plan **must** be
   explicitly flagged in subsequent experiment designs, not just
   noted in a sub-report.

**Recommendations for Phase 1.4 (and beyond):**

- Continue the §2 workflow.
- Add a "blind-spot scan" step at the end of each phase: explicitly
  list "what didn't fit the plan, and which downstream experiments
  should incorporate it". The user did this manually in the M
  reframing; it should be a documented step.
- Consider whether the "auto mode" execution model encourages over-
  coherence. Each phase produces a polished narrative; the off-script
  observations may get suppressed in service of clean reporting.

---

## What to send to the next AI round

A new package (per FPP plan §4.3) should be assembled containing:

- This Phase 1.3 master report
- Five Phase 1.3 sub-reports (I/J/K/L/M)
- Phase 1.2 master + sub-reports (for cross-phase context)
- Phase 1.1 master + sub-reports
- Phase 1 baseline report
- All key figures
- Per-experiment processed JSONs and selected raw traces
- Source code snapshot
- Both essay versions + plans v0.1 through v0.4
- An updated README with the Phase 1.3 question set:
  1. Is the third-endpoint regime a genuine new dynamical object, or
     an artefact of input zoo construction? What's the cleanest test?
  2. The "wpe-supported marginal equilibrium" framing for HTML
     structure — does this match the dynamical systems theory of
     similar objects (slow manifolds, ghosts of saddle-node
     bifurcations)? Is there a sharper formal name?
  3. Is the σ_max-vs-ρ asymmetry in attn/mlp consistent with what
     the mechanistic-interpretability literature says about
     transformer block dynamics?
  4. The §11 meta-discipline check: what's the most likely Phase
     1.3 blind spot? The third-endpoint regime is the candidate I
     flagged; is there another?
  5. Phase 1.4 prioritisation. The three Group-Cheap candidates
     (J-style on square brackets, HTML-only ρ, γ histogram) total
     about half a day of GPU. Should we just do all three?

Per plan §2 workflow: each AI sees the data **before** any other AI's
response.

---

## Caveats applying to all of Phase 1.3

- **Single architecture (GPT-2 small).** Cross-architecture replication
  is excluded by plan §1.2. Phase 1.4 should reconsider.
- **Small input sets** throughout (10–30 inputs per experiment). Phase
  1.3 means and ratios are reliable to ≈ 10% relative; sub-percent
  claims are not supported.
- **Mode A only.** Mode B and mid-layer iteration are excluded by plan
  §1.2.
- **All inference-only.** "Training drives both effects (high-norm
  fixed points + reduced γ)" is correlational; we observe trained vs
  random gaps, not the training trajectory.
- **The "everything explains itself" warning still applies.** Phase
  1.3 has produced a coherent picture; the picture's hidden gap is the
  third-endpoint regime's full diagnosis (Phase 1.4 item 1).

---

## Pointers

- Plans: `docs/FPP_experiment_plan_v0_{1,2,3,4}.md`
- Phase 1: `outputs/reports/phase1_gpt2_small/report.md`
- Phase 1.1 master: `outputs/reports/phase1_1_master/phase1_1_master_report.md`
- Phase 1.2 master: `outputs/reports/phase1_2_master/phase1_2_master_report.md`
- Phase 1.3 sub-reports:
  - `outputs/reports/phase1_3_markup_triangulation/phase1_3_markup_triangulation.md` (I)
  - `outputs/reports/phase1_3_basin_diagnosis/phase1_3_basin_diagnosis.md` (J)
  - `outputs/reports/phase1_3_layer_jacobian/phase1_3_layer_jacobian.md` (K)
  - `outputs/reports/phase1_3_ln_decomposition/phase1_3_ln_decomposition.md` (L)
  - `outputs/reports/phase1_3_spectral_radius/phase1_3_spectral_radius.md` (M)
- Per-experiment raw + processed: `data/{raw,processed}/phase1_3_*/`
- Source: `src/{markup_triangulation, basin_diagnosis, layer_jacobian,
  ln_decomposition, spectral_radius}.py`
- Total wall-clock for Phase 1.3 sub-experiments: ≈ 35 minutes GPU + ≈ 4
  hours analysis & reporting.
