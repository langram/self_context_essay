# Phase 1.3 J.3 Corrigendum

**Project:** FixedPointProbe (FPP)
**Type:** Error correction — explicit verdict revocation
**Date:** 2026-04-27
**Source plan:** `docs/FPP_experiment_plan_v0_5.md`
**Methodology:** `docs/FPP_Regulae.md` v0.2

---

## Executive summary

**The Phase 1.3 master report's headline verdict — "HTML structure is NOT
a real fixed point / wpe-supported marginal equilibrium" — was wrong.**

The verdict was derived from J.3's tail-rate fit using window `[200, 800]`,
which gave λ ≈ 1.0001 with R² = 0.011 for the HTML traces. The correct
reading of those numbers is **not** "HTML doesn't exponentially decay" —
it is **"the [200, 800] window is in the noise plateau, after the system
has already converged"**.

Direct test (per Regula VII): apply the **same** [200, 800] window to a
**lowercase** trace, which is unambiguously an exponential attractor.
Result: λ = 1.0000, R² = 0.0003. By the original report's own logic,
lowercase would also be "not exponentially decaying", which is absurd.
**The fitting window itself is the load-bearing variable**, not the λ
value or R² value at any particular window.

In auto-detected transient windows, both basins are clean (or near-clean)
exponential attractors:

| group | n traces | window | λ (mean) | R² (mean) |
| --- | ---: | --- | ---: | ---: |
| lowercase (J.1 lc_rep) | 4 | [0, ~23] | **0.569** | **0.989** |
| HTML (J.1 html_rep_0/4) | 2 | [0, ~181] (whole transient) | 0.941 | 0.731 |
| HTML (J.1 html_rep_8/12) | 2 | [73, ~181] (smooth tail only) | **0.886** | **0.949** |

The HTML structure exhibits a **chaotic early phase** (steps 0–72) followed
by a **smooth exponential decay** (steps 73+) at **λ ≈ 0.89**. The
lowercase basin has no chaotic phase and decays smoothly from step 0 at
**λ ≈ 0.57**. **Both are genuine exponential attractors at different
rates and with different transient structures.**

The Phase 1.2 "two attractors at different convergence speeds" framing
was correct. The Phase 1.3 master report's revision of that framing
should be retracted.

---

## Step 1: Refit results

### Lowercase J.1 long traces (n = 4)

| trace | auto-window | n_points | λ | R² | quality |
| --- | --- | ---: | ---: | ---: | --- |
| j1_lc_rep_3 | [0, 23] | 23 | 0.5724 | 0.9896 | clean_exponential |
| j1_lc_rep_7 | [0, 23] | 23 | 0.5683 | 0.9897 | clean_exponential |
| j1_lc_rep_11 | [0, 23] | 23 | 0.5693 | 0.9894 | clean_exponential |
| j1_lc_rep_15 | [0, 23] | 23 | 0.5647 | 0.9884 | clean_exponential |
| **mean ± std** | | | **0.569 ± 0.003** | **0.989 ± 0.001** | |

All 4 lowercase traces are clean exponential attractors with λ ≈ 0.57
in the auto-detected transient window [0, 23]. R² > 0.988 in all four.

### HTML J.1 long traces (n = 4)

| trace | auto-window | n_points | λ | R² | quality |
| --- | --- | ---: | ---: | ---: | --- |
| j1_html_rep_0 | [0, 181] | 181 | 0.9408 | 0.7313 | ambiguous_exponential |
| j1_html_rep_4 | [0, 181] | 181 | 0.9408 | 0.7311 | ambiguous_exponential |
| j1_html_rep_8 | [73, 181] | 108 | 0.8860 | 0.9485 | ambiguous_exponential |
| j1_html_rep_12 | [73, 181] | 108 | 0.8860 | 0.9485 | ambiguous_exponential |

The auto-detector found two distinct window patterns for nominally-similar
HTML traces. Inspecting the deltas series shows **all 4 HTML traces have a
chaotic early phase (steps 0–72) followed by a smooth exponential phase
(steps 73–181)**. When the detector picks the longer window [0, 181] it
includes the chaotic phase and R² drops to ~0.73; when it skips the
chaotic phase and picks [73, 181] only, R² rises to ~0.95.

The smooth-phase fit (λ = 0.886) is the more accurate estimate of the
HTML basin's local exponential decay rate. The whole-transient fit
(λ = 0.94) is biased by the chaotic phase and should be reported as a
loose upper bound on the smooth-phase λ.

### Markup-triangulation traces (n = 30, 5 per category)

| category | window range | λ (mean) | R² (mean) | quality breakdown |
| --- | --- | ---: | ---: | --- |
| angle | [105, 129] | 0.817 | 0.839 | 5/5 ambiguous |
| square | [0, 52] | 0.598 | 0.959 | 3 clean / 2 ambiguous |
| curly | [1, 11] | 0.450 | 0.991 | 5/5 clean |
| parens | [21, 50] | 0.877 | 0.710 | 3 clean / 2 non-exponential |
| quotes | [0, 13] | 0.456 | 0.989 | 5/5 clean |
| isolated | [0, 129] | 0.568 | 0.958 | 3 clean / 1 ambiguous / 1 no-transient |

Reading:

- **angle** behaves like J.1 HTML (slow exponential after chaotic phase) —
  λ ≈ 0.82, R² ≈ 0.84.
- **curly, quotes** behave like J.1 lowercase (fast clean exponential) —
  λ ≈ 0.45.
- **square, isolated** are also lowercase-like — λ ≈ 0.57–0.60.
- **parens** is the noisiest — λ ≈ 0.88, R² ≈ 0.71. The two parens
  inputs that fitted as "non-exponential" suggest a less-clean attractor.

This refines Experiment I's "third endpoint regime" finding: the regime is
**not** a single dynamical object distinct from both lowercase and HTML.
Square / isolated decay at lowercase-like rates; parens decay at HTML-like
rates with worse R². The third regime is a mixture, not a third basin.

## Step 2: Wrong-window vs right-window comparison

This is the corrigendum's load-bearing demonstration: applying the same
wrong window to both basins makes both look "not exponential", which is
the smoking gun that the window choice (not the basin nature) was the
problem.

### Lowercase trace (j1_lc_rep_3)

| window | λ | R² | conclusion under naïve reading |
| --- | ---: | ---: | --- |
| [200, 800] (master report's HTML window) | **1.0000** | **0.0003** | "no exponential decay" — false |
| [0, 23] (auto-detected) | **0.5724** | **0.9896** | "clean exponential" — true |

Same trace, two windows, opposite conclusions. The trace itself is
unchanged.

Noise floor for j1_lc_rep_3 (median of last 1000 deltas): 2.61e-04.
By step 23 the deltas have already crossed below 100×noise = 0.026,
i.e. the trace has effectively converged. After step ~23 it sits in
floating-point noise around the fixed point.

### HTML trace (j1_html_rep_0)

| window | λ | R² | conclusion under naïve reading |
| --- | ---: | ---: | --- |
| [200, 800] (master report's window) | **1.0001** | **0.0108** | "no exponential decay" — false |
| [0, 181] (auto-detected) | **0.9408** | **0.7313** | "exponential, ambiguous" — partial |
| [73, 181] (smooth tail only — manual) | 0.8838 | 0.9582 | "clean exponential" — true |

For HTML, [200, 800] is well past the convergence step (~159) and well
into the noise plateau. The deltas there are floating-point fluctuations
around 5e-4. Fitting a line through those numbers gives λ ≈ 1 by
construction.

The window mistake was not "off by a factor", it was **categorically
wrong** — fitting in the noise plateau measures noise variance, not the
exponential decay rate. Both basins look "not exponential" under that
window because both basins have a noise plateau after their convergence.

## Step 3: Verdict revocation and replacement

Per the v0.5 plan §6 — explicit revocation language, no defensive
narrative.

### Verdicts revoked

> **Phase 1.3 master report — REVOKED**:
> "HTML-induced metastable structure is NOT a real fixed point. Lowercase
> basin has λ = 0.775 (R²=0.81); HTML has λ = 1.0001 (R²=0.011) — no
> exponential decay at all."

> **Phase 1.3 master report — REVOKED**:
> "Phase 1.2's 'two attractors' framing was wrong about the second
> basin."

> **Phase 1.3 master report — REVOKED**:
> "Wpe-supported marginal equilibrium" naming for the HTML structure.

### Verdicts replaced

> **REPLACEMENT 1**: Both basins are genuine exponential attractors.
> Lowercase: λ ≈ 0.57 in window [0, 23], R² ≈ 0.99. HTML: λ ≈ 0.88 in
> window [73, 181] after a chaotic transient phase, R² ≈ 0.95. The
> 13× wall-clock convergence-step gap reflects a 1.5× difference in
> per-step contraction rate plus a longer chaotic warm-up phase for
> the HTML basin.

> **REPLACEMENT 2**: Phase 1.2's "two attractors at different convergence
> rates" framing is correct. Phase 1.3 master report's revision of that
> framing is withdrawn.

> **REPLACEMENT 3**: The HTML structure is an **exponential attractor with
> a slower local contraction rate and a longer pre-exponential chaotic
> phase**. The "wpe-supported" qualifier from Phase 1.3 J.2 (drift
> toward lowercase under wpe shutoff) survives separately and is not
> revoked — but it characterises a reduction in the HTML basin's pull
> when wpe is removed, not the basin's status as a fixed point.

### Verdicts NOT revoked

The following Phase 1.3 verdicts are **not** affected by this corrigendum:

- **Experiment I**: angle-bracket-specific basin selection (refines
  Phase 1.2 H from "markup as category"). Stands.
- **Experiment K**: per-layer LayerNorm contraction distributed across
  all 12 layers; L0/L1 outliers. Stands.
- **Experiment L**: γ × state-norm multiplicative decomposition;
  L0 is γ-driven outlier. Stands.
- **Experiment M**: σ_max vs ρ asymmetry on attn/mlp; LayerNorm is
  symmetric; full stack ρ < 1. Stands. (See §6.3 caveat regarding
  the M reading of attn/mlp asymmetry.)
- **Experiment J.2** (WPE shutoff after capture): cos drifts from
  0.508 → 0.720 over 1000 cancel-pos steps. Data stands. Interpretation
  changes: the drift is the basin reducing in attractor strength when
  wpe is removed, not "the basin disappearing because it was never
  real".
- **Experiment J.4** (cycle detection): no formal cycle signal. Stands.

## Step 4: Falsifiability conditions (per Regula IV)

For each replacement verdict, explicit falsification conditions:

### Replacement 1 — "Both basins are exponential attractors"

Falsified by:
- HTML auto-detected transient λ has R² < 0.85 *across more than 50% of
  HTML traces* under any reasonable window-detection criterion. (Current
  R² = 0.95 on smooth-tail-only fits; relaxing to whole-transient fits
  drops to R² = 0.73 due to chaotic phase, but R² ≥ 0.95 holds for the
  late-transient component.)
- HTML λ varies by > 0.20 across reasonable window choices. (Current
  range 0.88–0.94, span 0.06 — within tolerance.)
- Cross-architecture (Pythia / LLaMA) replication shows HTML traces have
  qualitatively different dynamical structure (e.g. genuine non-exponential
  behaviour in the smooth tail).

### Replacement 2 — "Phase 1.2's two-rate framing is correct"

Falsified by:
- HTML λ exceeds 1.0 in any reasonable smooth-tail window (would mean
  divergent rather than slowly contracting).
- Cross-architecture replication shows fundamentally different basin
  ordering or numerosity.

### Replacement 3 — "HTML basin has chaotic + exponential phase structure"

Falsified by:
- The chaotic phase persists indefinitely (i.e. λ never stabilises) under
  longer traces or different reps. Currently 4/4 HTML traces show clear
  smooth tails; a future experiment with very different angle-bracket
  inputs producing persistently-non-exponential traces would falsify.
- The chaotic phase is an artefact of seq_len = 64 / specific positional
  encodings and disappears at other seq_len.

## Step 5: Evidence scope (per Regula III)

The replacement verdicts have the following scope:

```
Model:        GPT-2 small (124M, default config)
Regime:       mode-A hidden-state self-iteration via inputs_embeds
Inputs:       4 lowercase J.1 + 4 HTML J.1 long traces (10000 steps each)
              + 30 markup-triangulation traces (≤ 200 steps)
Seq_len:      64
Dtype:        fp32
WPE:          active (continuous re-injection)
Method:       auto-detected transient window per trace, log-linear fit
```

Generalization restrictions (NOT yet validated, per Regula II):
- Cross-architecture (Pythia rotary, LLaMA RMSNorm) — required for any
  "transformer mechanism" claim, NOT yet done.
- Cross-encoding (same content as JSON / Lisp / natural language) —
  required for any "semantic invariant" claim, NOT yet done.
- Mode B (sampling) / Mode C (token argmax) regimes — different fixed-point
  equations.
- Larger context (seq_len > 64) — chaotic-phase length may be seq_len
  dependent.

## §2.1 Regulae compliance check

Per the v0.5 plan §2.1, each Regula must be displayed and answered
explicitly.

### Regula I — minimum-component alternative

The replacement framework needs at most these components:

1. One exponential attractor at λ ≈ 0.57 (the lowercase basin).
2. One exponential attractor at λ ≈ 0.88 with a chaotic warm-up
   phase (the HTML basin).
3. LayerNorm contraction.

3 components, vs the original master report's 7+ (lowercase strong
attractor + HTML wpe-marginal + third intermediate regime + LayerNorm
contraction + attn/mlp expansion + asymmetric coupling + …). The
corrigendum reduces explanatory complexity by removing the spurious
"non-attractor" component.

The 5-bracket-type endpoint structure from Experiment I survives but is
re-read: brackets that *converge fast at λ ≈ 0.45–0.60* (curly/quotes/
square/isolated) belong to the lowercase basin region; brackets that
*converge slow at λ ≈ 0.82–0.88* (angle/parens) belong to the HTML basin
region. The "third endpoint" was a mixture, not a third dynamical object.

### Regula III — evidence scope

Done in §5 above. Each verdict has its scope; generalization restrictions
are explicit.

### Regula IV — falsifiability

Done in §4 above. Each replacement verdict has explicit falsification
conditions naming specific observables.

### Regula V — oscillation discipline

Two questions to answer at the corrigendum's end (per v0.5 plan §2.1):

**1. Ptolemy check: is hidden-state self-iteration the right object at all?**

The J.3 fitting error exposes a deeper concern: we have been working
exclusively in mode-A hidden-state space for four phases. The chaotic
transient phase in HTML traces (steps 0–72 with no clean structure)
suggests that mode-A iteration may not be the natural "self-context
dynamics" projection — it's a mathematical iteration imposed on a model
trained for autoregressive generation.

Alternative coordinate systems / objects worth considering:
- **Token-level dynamics directly** (Mode B/C, plan-excluded so far).
- **Conditional generation as iteration** — running the model
  autoregressively from a fixed prompt and measuring trajectory in
  token-output space rather than hidden-state space.
- **Attention-pattern dynamics** — track how attention matrices evolve
  rather than hidden states.

I cannot answer "which is the right object" within the current framework
— it requires external perspective. **Flagging this honestly per
Regula V**: I am inside the framework and cannot oscillate out of it
without external input. This is a candidate for U2 user perspective
injection in Phase 1.4.

**2. Invariant assumption: under cross-architecture replication of
"two genuine exponential attractors", what most likely fails first?**

Most likely candidates:
- The **rate ratio** (1.5× per-step) is GPT-2-small-specific. In larger
  or different architectures the ratio could flip or vanish.
- The **chaotic warm-up phase** for HTML may be specific to GPT-2's
  positional encoding; rotary or ALiBi might give a clean exponential
  from step 0.
- The **HTML basin existing at all** is the most fragile — Pythia
  trained on different data may not have the same basin structure.
- The **angle-bracket selectivity** (Experiment I) is most likely
  GPT-2-tokenizer-specific.

The most robust prediction that should generalise: trained transformers
under self-iteration produce *some* small number of basins (1–3) that
are exponential attractors with similar rate orderings (natural-text
content fastest, structured-syntax content slower). Whether 2 or 3
basins is GPT-2-specific.

### Regula VI — fit / mechanism / invariant layering

| claim | layer |
| --- | --- |
| "lowercase basin has λ ≈ 0.57 in [0, 23] window" | **fit** (GPT-2 small specific) |
| "HTML basin has λ ≈ 0.88 in [73, 181] smooth tail" | **fit** (GPT-2 small specific) |
| "HTML basin has chaotic + smooth phase structure" | **fit** (GPT-2 small specific) |
| "Trained transformers' self-iteration produces small number of exponential basins with content-dependent rates" | **invariant candidate** (requires Regula II to upgrade) |
| "Both basins exhibit exponential decay near fixed point" | **mechanism** (in current setup, ρ < 1 was confirmed in M; this corrigendum strengthens by showing direct trajectory exponentiality) |

Nothing in this corrigendum is a confirmed invariant claim. Cross-arch
work (Phase 1.4 must include per Regula II) is the bottleneck.

### Regula VII — key statistical recomputation

This corrigendum **is** Regula VII applied to J.3. The independent
recompute that triggered it:

1. Original report claimed HTML λ = 1.0001, R² = 0.011 was evidence
   of "no exponential decay".
2. Recompute with same window on lowercase: λ = 1.0000, R² = 0.0003 —
   absurd if interpreted as "no decay".
3. Therefore the issue was the window choice, not the basin nature.

Subsequent recompute steps:
- Auto-detected windows on all 8 long traces — produced consistent λ
  values within each basin, R² > 0.95 on smooth-tail fits.
- Wrong-vs-right window contrast on representative traces (Step 2) —
  the smoking gun.

Regulae self-consistency check: this corrigendum is itself susceptible
to Regula VII. **What sanity-check have I done on the corrigendum's
own numbers?**

- Auto-window detection: tested on both regimes (long-trace plateau and
  short-trace threshold-stop). Found a bug in the first version (entire-
  trace median for noise floor when n < noise_window_size); fixed.
- Cross-trace consistency: 4/4 lowercase agree to 3 decimals on λ
  (0.564–0.572). 2/4 HTML agree to 4 decimals on smooth-tail λ
  (0.886). 2/4 HTML have whole-transient fits that include the chaotic
  phase — same numbers within rounding (0.9408 in both). The detector
  is deterministic given the data.
- Comparison with non-exponential baseline: a constant-delta trace
  (deltas all = 1) would give λ = 1.0, R² = 0 by construction. The
  noise plateau in J.1 traces is essentially this regime. λ ≈ 1.0 is
  the signature of "no decay", confirmed. λ < 1 with R² > 0.95 is the
  signature of "exponential decay", confirmed.

## §1.3 Known unprocessed issues (recorded, not implemented)

Per the v0.5 plan §1.3, three issues are flagged for Phase 1.4 candidates,
NOT addressed in this corrigendum:

1. **Third-endpoint actual transient λ measurement** — Experiment I's
   square brackets and isolated chars produce intermediate cos values
   (0.90 to lowercase, 0.61 to HTML). This corrigendum's refit shows
   their λ values cluster near lowercase (~0.57–0.60). Whether this
   means "they ARE in the lowercase basin" or "they're in a third
   basin near lowercase" is not settled. A J-style long-trace test
   on these inputs would clarify.
2. **Cross-architecture validation (Regula II)** — required for any
   "transformer mechanism" claim, not done. The replacement verdicts
   in this corrigendum are still GPT-2-small-specific.
3. **Phase 1.3 M's σ_max vs ρ readings on attn/mlp** — separate from
   this corrigendum. M's data stands; M's interpretation that "attn/mlp
   are mildly expansive" is not affected by the J.3 window error.

## Implications for the Phase 1.3 master report

The Phase 1.3 master report's TL;DR and §"Per-question answers Q4"
section are partially invalidated. Specifically:

- The Q4 answer "the HTML structure is a wpe-supported marginal
  equilibrium, NOT a real fixed point" — REVOKED.
- The TL;DR's "Phase 1.2 was wrong about the secondary attractor's
  nature" — REVOKED. Phase 1.2 was right.
- The "13× convergence-step gap is attractor + non-attractor" framing
  — REVOKED.
- The "wpe-supported marginal equilibrium" naming — REVOKED.

The Phase 1.3 master report's other sections (Q1, Q2, Q3, Q5, K, L, M
findings) are unchanged.

A note in the master report linking to this corrigendum should be added.
This corrigendum lives at
`outputs/reports/phase1_3_J3_corrigendum/phase1_3_J3_corrigendum.md`.

## Workflow lesson — the J.3 error and Regula VII's promotion

This corrigendum is the case study that motivated promoting "key
statistical recomputation" from operational utility (v0.1 D2) to formal
Regula VII (v0.2). Without an independent recompute, the master
report's λ = 1.0001 / R² = 0.011 numbers were taken as evidence; with
the recompute (the same window on lowercase giving the same "absurd"
result), the error was immediately visible.

The first-round Phase 1.3 evaluation accepted the J.3 narrative without
recomputing the key fit. This is the failure mode Regula VII formally
prohibits going forward.

The Phase 1.3 master report's section "Plan §11 meta-discipline check"
hypothesised that Phase 1.3's own picture might have a hidden gap.
**This corrigendum is that hidden gap surfaced.** The fact that Phase
1.3's picture was indeed broken in a substantive way — and broken in
the way the meta-warning predicted — is itself a partial validation of
the §11 discipline. But it also exposed the limit: meta-warning alone
didn't catch the error; an external recomputation did.

## Pointers

- Refit data:
  - `data/processed/phase1_3_J3_corrigendum/refits/lowercase_traces_refit.json`
  - `data/processed/phase1_3_J3_corrigendum/refits/html_traces_refit.json`
  - `data/processed/phase1_3_J3_corrigendum/refits/third_endpoint_refit.json`
- Wrong-vs-right window comparison:
  `data/processed/phase1_3_J3_corrigendum/window_comparison.json`
- Revised verdict JSON:
  `data/processed/phase1_3_J3_corrigendum/revised_verdict.json`
- Source: `src/j3_corrigendum.py`
- Regulae document: `docs/FPP_Regulae.md` (v0.2)
- Original task plan: `docs/FPP_experiment_plan_v0_5.md`

## Document version

- v0.1 (2026-04-27): initial corrigendum following v0.5 plan, Regulae v0.2.
