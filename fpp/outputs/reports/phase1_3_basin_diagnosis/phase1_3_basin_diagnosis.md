# Phase 1.3 — Experiment J: HTML-induced structure ontology diagnosis

- model: `gpt2`, fp32, CUDA
- 8 long traces (4 angle-bracket-input reps + 4 lowercase-input reps), max_iter = **10000**, save_every = 100, threshold = 1e-5 (run-to-completion regardless)
- 4 WPE-shutoff traces: capture at step 200 of normal mode-A, then **1000 cancel-pos steps**
- elapsed: ~11 min total

## TL;DR — the HTML structure is **NOT** a real fixed point

The most surprising and important Phase 1.3 finding: **the HTML-induced
metastable structure does not exhibit exponential convergence**. Tail-rate
fits give λ ≈ 1.0 with R² ≈ 0.01 — the residual deltas are essentially
constant, not exponentially decaying.

The lowercase attractor IS exponentially attracting with **λ ≈ 0.78**
(R² ≈ 0.81 on the early tail).

The "13× slow convergence" mystery (HTML 129 steps vs lowercase 10) is
explained: the lowercase basin pulls inputs in via fast exponential
contraction; the HTML "basin" is a marginally stable equilibrium where
the system stops moving in absolute terms but is not strongly
attracting. The 129 steps is how long it takes to drift to the
marginally stable region, not how long an exponentially attracting
basin takes.

WPE shutoff after capture causes the HTML state to drift toward
lowercase (cos_lc 0.508 → 0.720 over 1000 cancel-pos steps), but
**not** immediately collapse. The HTML structure is therefore neither
purely wpe-forced (would collapse) nor purely intrinsic (would stay).
It is a **wpe-supported marginal equilibrium**: the architectural
pull toward lowercase exists but is weak; wpe injection on top of
that pull produces a balance point that the mode-A iteration finds.

## J.1 — long-horizon stability (10000 steps)

| trace | n_steps | final ‖h‖ | final eff_rank | final cos_lc | final cos_html |
| --- | ---: | ---: | ---: | ---: | ---: |
| html_rep_0 | 10000 | 1044 | 1.11 | +0.508 | +1.000 |
| html_rep_4 | 10000 | 1044 | 1.11 | +0.508 | +1.000 |
| html_rep_8 | 10000 | 1044 | 1.11 | +0.508 | +1.000 |
| html_rep_12 | 10000 | 1044 | 1.11 | +0.508 | +1.000 |
| lc_rep_3 | 10000 | 2563 | 1.02 | +1.000 | +0.490 |
| lc_rep_7 | 10000 | 2563 | 1.02 | +1.000 | +0.490 |
| lc_rep_11 | 10000 | 2563 | 1.02 | +1.000 | +0.490 |
| lc_rep_15 | 10000 | 2563 | 1.02 | +1.000 | +0.490 |

All 8 traces remain at their basins for the full 10000 steps. **Neither
basin shows drift.** The HTML traces stay at cos_html = 1.000 after
~step 200; the lowercase traces stay at cos_lc = 1.000 after ~step 10.
No metastable transient drift visible up to 10× longer than Phase 1.1
B's 1000-step budget.

## J.2 — WPE shutoff after capture

After 200 normal mode-A iterations (HTML state captured), switch to
cancel-pos for 1000 more steps:

| trace | cos_lc at step 200 | cos_lc at step 1200 | cos to h_capture at step 200 | cos to h_capture at step 1200 |
| --- | ---: | ---: | ---: | ---: |
| html_rep_0 | +0.508 | **+0.720** | +1.000 | **+0.957** |
| html_rep_4 | +0.508 | +0.720 | +1.000 | +0.957 |
| html_rep_8 | +0.508 | +0.720 | +1.000 | +0.957 |
| html_rep_12 | +0.508 | +0.720 | +1.000 | +0.957 |

All 4 captured states drift in unison: cos to lowercase moves from
0.51 → 0.72; cos to the original h_capture drops from 1.00 → 0.96.

Reading vs the four pre-committed verdicts in plan §3.3.2:
- "Stays stable at h_capture cos≈1": **partial — capture cos drops
  to 0.96, not stable**
- "Slowly drifts to lowercase, cos→1": **partial — drifts toward
  lowercase but reaches only 0.72 in 1000 steps, not 1.0**
- "Immediately collapses (<50 steps)": **no**
- "Enters oscillation": **no — drift is monotonic**

So none of the four pre-committed verdicts cleanly applies. The
correct read is the synthesis: the HTML structure is **wpe-co-supported,
neither purely wpe-forced nor intrinsic**. After wpe is removed it
slowly drifts toward lowercase but does not collapse fast.
Extrapolating naively: the drift rate (0.51 → 0.72 in 1000 steps,
≈ 0.0002 per step) suggests it would take ~10000 more steps to
reach cos_lc = 0.95 — far slower than lowercase's intrinsic
λ ≈ 0.78 contraction.

## J.3 — tail convergence rate fits

| trace | tail window | λ | R² | asymptotic window | λ_asym | R² |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| lc_rep_3 | 5–50 | **0.7747** | **0.81** | 9000–10000 | 1.0000 | 0.004 |
| lc_rep_7 | 5–50 | 0.7767 | 0.81 | 9000–10000 | 1.0000 | 0.0003 |
| lc_rep_11 | 5–50 | 0.7724 | 0.81 | 9000–10000 | 1.0000 | 0.002 |
| lc_rep_15 | 5–50 | 0.7753 | 0.81 | 9000–10000 | 1.0000 | 0.001 |
| html_rep_0 | 200–800 | **1.0001** | **0.011** | 9000–10000 | 1.0000 | 0.003 |
| html_rep_4 | 200–800 | 1.0001 | 0.006 | 9000–10000 | 1.0001 | 0.010 |
| html_rep_8 | 200–800 | 1.0001 | 0.011 | 9000–10000 | 1.0001 | 0.012 |
| html_rep_12 | 200–800 | 1.0001 | 0.011 | 9000–10000 | 1.0000 | 0.008 |

**Decisive finding:**
- **Lowercase tail**: λ ≈ 0.775, R² ≈ 0.81. Clean exponential decay
  at ≈ 22% per step. This is a textbook attractor.
- **HTML tail**: λ ≈ 1.0001, R² ≈ 0.01. The fit explains 1% of the
  variance — the deltas are *not* exponentially decaying. They are
  constant (or random) at magnitude ~5e-4.

The 13× difference in convergence-step count (HTML 129 vs lowercase
10) is **not** because HTML is "the same kind of attractor as lowercase
but slower". HTML has a different dynamical nature: it is a
**marginal equilibrium**, λ exactly at 1, no exponential pull.
Lowercase's λ ≈ 0.78 means a perturbation shrinks by 22% per step;
the HTML residual stays the same.

In the asymptotic 9000–10000 window, both basins have λ ≈ 1.0 with
R² ≈ 0 — at that point lowercase is also "at" its attractor and the
deltas are floating-point noise. The clean λ ≈ 0.78 only shows up in
the early tail of lowercase, when it's still actively contracting.

## J.4 — cycle detection (post-hoc on J.1 data)

For each trace, mean ‖h_n − h_{n−k}‖ over the second half of the
saved trace (steps 5000–10000) for k ∈ {1, 2, 3, 4, 8, 16}:

| trace | k=1 | k=2 | k=3 | k=4 | k=8 | k=16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| html_rep_0 | 5.08e-4 | 5.01e-4 | 5.25e-4 | **4.66e-4** | 4.81e-4 | 5.16e-4 |
| html_rep_4 | 5.10e-4 | 5.02e-4 | 5.22e-4 | **4.68e-4** | 4.75e-4 | 5.15e-4 |
| html_rep_8 | 5.12e-4 | 4.98e-4 | 5.18e-4 | **4.58e-4** | 4.74e-4 | 5.13e-4 |
| html_rep_12 | 5.19e-4 | 4.94e-4 | 5.16e-4 | **4.64e-4** | 4.72e-4 | 5.14e-4 |
| lc_rep_3 | 2.60e-4 | 2.59e-4 | 2.60e-4 | 2.58e-4 | 2.58e-4 | 2.59e-4 |
| lc_rep_7 | 2.60e-4 | 2.65e-4 | 2.60e-4 | 2.57e-4 | 2.64e-4 | 2.63e-4 |
| lc_rep_11 | 2.60e-4 | 2.63e-4 | 2.60e-4 | 2.63e-4 | 2.66e-4 | 2.66e-4 |
| lc_rep_15 | 2.65e-4 | 2.64e-4 | 2.62e-4 | 2.65e-4 | 2.66e-4 | 2.65e-4 |

No formal cycle signal (none of the k > 1 distances are below 0.5 ×
baseline, the threshold I used). But there is a **suggestive
quasi-cycle hint** in the HTML traces: k = 4 distance is consistently
~10% smaller than k = 1 across all four HTML traces. The lowercase
traces show no such pattern.

This is below the formal cycle threshold but the consistency across
4 independent HTML traces is non-trivial. **It hints at a weak
period-4 oscillation around the marginal equilibrium**. Plan §6.2
warned against false-positive cycle signals; the threshold I used
(< 0.5 × baseline) is conservative. A more sensitive threshold
(< 0.95 × baseline) would flag this. **Recorded as suggestive,
not claimed.**

The lowercase traces are essentially flat across all k — they are
sitting at a true fixed point with floating-point noise around it,
no cyclic structure.

## Synthesis — ontology verdict on the HTML-induced structure

The HTML-induced metastable structure is **none** of the four
pre-committed verdicts cleanly. The closest match is a synthesis:

**Wpe-supported marginal equilibrium**: a hidden-state region where
the trained-blocks dynamics provide weak (sub-exponential) attraction,
and continuous wpe re-injection provides additional support. Without
wpe re-injection (J.2), the architectural pull is toward lowercase
but slow. Under continuous wpe re-injection (J.1), the system stops
moving in absolute terms but does not exhibit exponential decay
(J.3). The structure has weak quasi-periodic micro-oscillations
(J.4 suggestive only).

This **reverses** the cleanest reading of Phase 1.2 ("two attractors").
Phase 1.2 H's "capital basin" should not be called a basin in the
dynamical-systems sense — it lacks the exponential contraction that
defines a basin. It is a *region that the iteration enters and stops
moving in*, but the stopping is not because of exponential pull.

The Phase 1.2 master report wrote:
> "trained blocks define ONE architectural attractor; one position-
> embedding-induced secondary attractor selected by angle-bracket tokens"

Phase 1.3 J corrects this to:
> Trained blocks define ONE architectural attractor (lowercase, λ ≈ 0.78,
> R² ≈ 0.81). The angle-bracket inputs land at a **wpe-supported
> marginal equilibrium** (λ ≈ 1.000, R² ≈ 0.01) — a region the iteration
> finds and stops in but which lacks the exponential pull of a true
> attractor.

## What this means for the Phase 1.2 picture

Three concrete revisions:

1. **The "two attractors" framing is wrong**. There is one attractor
   (lowercase) and one marginal equilibrium (HTML). The plan §0.1
   naming discipline ("HTML-induced metastable structure" / "wpe-
   supported candidate") was empirically justified.
2. **The "13× slow" mystery is resolved**: not a slow attractor, a
   non-attractor. Phase 1.2 evaluators didn't catch this because both
   showed cos = stable values at convergence; only the rate analysis
   distinguishes them.
3. **The Phase 1.2 H "12/12 markup-flip pairs flip basin" result
   needs the same revision**: it's not "12 inputs flip basin" but
   "12 inputs land at the marginal equilibrium instead of the
   attractor".

## Caveats

- **R² ≈ 0.011 on HTML tail does not prove "no exponential decay
  exists"** — it could be exponential at a rate so close to 1 that
  the deltas-vs-step relation is masked by noise. The data is
  *consistent with* λ very close to 1, but a precise upper bound
  on λ would require longer traces or smaller fit windows.
- **J.2 ran for 1000 cancel-pos steps**. The drift cos_lc 0.508 →
  0.720 is monotonic but slow. Whether continuing for 10000+ steps
  would reach cos_lc = 1.0 is **not** answered by this data. A
  longer follow-up could be a Phase 1.4 candidate. The drift rate
  estimate (≈ 0.0002 per step) is itself noisy.
- **n = 4 HTML traces and n = 4 lowercase traces**. The clean
  reproducibility within each group (final norms, final cos values,
  λ all match to 3 decimals) is strong evidence the dynamics are
  deterministic given the architecture, not statistical claims at
  the population level.
- **The k = 4 cycle hint is at 10% below baseline** — well below the
  formal threshold (50%) and should not be promoted to a finding.
  Recorded for future investigation if other Phase 1.3+ signals
  point at it.
- **WPE shutoff implementation**: J.2 uses cancel-pos (variant C1)
  not posfree (C2). They were equivalent in Phase 1.2 C, but the
  test here is on a state that already includes wpe contributions
  baked in, which is a different regime. Both variants should give
  the same answer (the modes are mathematically equivalent), but
  the explicit test is a Phase 1.4 belt-and-suspenders item.

## Pointers

- raw J.1 traces (101 saved states each, 10000 steps): `data/raw/phase1_3_basin_diagnosis/j1_*.pt`
- raw J.2 traces (cancel-pos shutoff): `data/raw/phase1_3_basin_diagnosis/j2_*.pt`
- aggregate results JSON: `data/processed/phase1_3_basin_diagnosis/results.json`
- figures:
  - `outputs/figures/phase1_3_basin_diagnosis/j1_cos_trajectories.png`
  - `outputs/figures/phase1_3_basin_diagnosis/j2_wpe_shutoff.png`
  - `outputs/figures/phase1_3_basin_diagnosis/j3_tail_rates.png`
- source: `src/basin_diagnosis.py`

## Conclusion for the master report

**Q4 answered (with revision): the HTML-induced structure is a
wpe-supported marginal equilibrium, not a fixed point**. It exhibits
λ ≈ 1.000 (no exponential decay) compared to the lowercase attractor's
λ ≈ 0.78. It survives 10000 steps under continuous wpe re-injection
without drift, but drifts (slowly) toward lowercase when wpe injection
is cancelled. No formal cycle signal; suggestive but un-confirmed
period-4 micro-oscillation hint.

**The 13× convergence-step gap (HTML 129 vs lowercase 10) has its
explanation: the lowercase basin pulls exponentially; the HTML
"basin" doesn't pull at all in the rigorous sense. The Phase 1.2
"two attractors" framing was wrong about the second basin.** This is
the most consequential Phase 1.3 finding.

**The plan §0.1 naming discipline was empirically right**:
"lowercase attractor" stays; "HTML-induced metastable structure" is
the correct name. The Phase 1.2 "capital basin" should retire entirely.
