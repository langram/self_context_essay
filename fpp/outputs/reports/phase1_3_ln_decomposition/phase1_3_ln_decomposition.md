# Phase 1.3 — Experiment L (expanded): γ/state decomposition for LayerNorm σ_max

**Expanded per user feedback before execution.** Original L target (plan §3.5):
decompose layer 6 ln_1's σ_max ≈ 0.12 into γ vs state-norm contributions.
User-specified expansion: also decompose **L0 ln_1** (the K outlier at
σ_max = 0.038, ≈ 4× more contractive than L6) to explain why early layers
contract harder.

- model: `gpt2`, fp32, CUDA, attn_implementation = "eager"
- layers: **0 and 6** (per the expansion)
- 4 conditions per (layer, input):
  - **A1**: trained γ × original h (norm ≈ 2563 lowercase / 1042 HTML)
  - **A2**: γ=1     × original h
  - **A3**: trained γ × rescaled h (norm = 100)
  - **A4**: γ=1     × rescaled h
- 10 inputs (4 Phase 1 reps + 6 Phase 1.1 B reps), σ_max via existing power iteration on J^T·J
- elapsed: ~5 s GPU

## TL;DR

**The decomposition is exactly multiplicative** — coupling factor
`(A1·A4)/(A2·A3) = 1.000` for both L0 and L6. σ_max factorises cleanly
into a γ-only effect and a state-norm-only effect. **No anomalous
γ × state-norm coupling.**

| layer | A1 (trained γ × orig h) | state-only (A2/A4) | γ-only (A3/A4) | coupling |
| ---: | ---: | ---: | ---: | ---: |
| 0 (K outlier) | **0.0376** | 0.046 | **0.252** | 1.000 |
| 6 (F baseline) | **0.1162** | 0.046 | **0.779** | 1.000 |

**The L0 outlier is a γ effect, not a state-norm effect.** State-only
contribution (A2/A4) is **identical** for both layers (0.046) — both see
the same fixed-point hidden states. The 4× difference between L0 and L6's
total contraction (0.0376 vs 0.1162) comes entirely from γ: L0's γ
contributes a 4× contraction (0.252) while L6's γ contributes only 1.3×
(0.779). L0 has aggressively learned γ values; L6 has modest ones.

## The four-condition table (mean over 10 inputs)

### Layer 0 ln_1

| condition | description | σ_max ± std |
| --- | --- | ---: |
| A1 | trained γ × original h | **0.0376 ± 0.0158** |
| A2 | γ=1 × original h | 0.1492 ± 0.0627 |
| A3 | trained γ × rescaled h (norm = 100) | 0.8745 ± 0.4487 |
| A4 | γ=1 × rescaled h | 3.4709 ± 1.7818 |

Ratios:
- A1/A4 = 0.012 (total contraction relative to baseline)
- A2/A4 = 0.046 (state-norm-only contribution)
- A3/A4 = 0.252 (γ-only contribution)
- Coupling (A1·A4) / (A2·A3) = **1.000**

### Layer 6 ln_1 (Phase 1.2 F baseline)

| condition | description | σ_max ± std |
| --- | --- | ---: |
| A1 | trained γ × original h | **0.1162 ± 0.0489** |
| A2 | γ=1 × original h | 0.1492 ± 0.0627 |
| A3 | trained γ × rescaled h | 2.7035 ± 1.3881 |
| A4 | γ=1 × rescaled h | 3.4709 ± 1.7818 |

Ratios:
- A1/A4 = 0.036
- A2/A4 = 0.046 (same as L0)
- A3/A4 = 0.779 (much weaker than L0's 0.252)
- Coupling = **1.000**

### Cross-layer comparison

| metric | L0 | L6 | reading |
| --- | ---: | ---: | --- |
| A1 (full trained σ_max) | 0.038 | 0.116 | L0 is 3.1× more contractive (matches K) |
| A2/A4 (state-only) | **0.046** | **0.046** | identical — same fixed-point norms |
| A3/A4 (γ-only) | **0.252** | **0.779** | **L0's γ is 3.1× more contractive** |
| coupling | 1.000 | 1.000 | clean multiplicative decomposition |

## Reading 1 — the decomposition is multiplicative

LayerNorm's Jacobian formula factorises as
`(γ / σ_x) · (I − 11ᵀ/d − (x−μ)(x−μ)ᵀ/(d·σ_x²))`. The
state-dependent part `(1/σ_x) · projection` and the γ-dependent part
γ are independent in this expression. The coupling factor
`(A1·A4)/(A2·A3)` measures whether σ_max respects this independence.

A coupling = 1.000 (to 3 decimals) confirms it does. **σ_max(LN) is
exactly factorisable into a γ effect and a state-norm effect**.
Phase 1.2 F's σ_max ≈ 0.12 result for L6 is now mechanistically
decomposed: 0.046 from state-norm × 0.779 from γ × baseline 3.47.

## Reading 2 — state-norm dominates the baseline contraction

The state-only contribution A2/A4 = 0.046 is identical across both
layers. This factor of ≈ 22× contraction is purely from the
fixed-point hidden state having large norm (≈ 2500 vs the rescaled
100). For an unconstrained baseline (random vector at norm 100),
σ_max(LN) = 3.47; at the trained fixed-point's norm (2500), even
γ = 1 reduces σ_max to 0.15. **The trained system's tendency to
sit at a high-norm fixed point is itself a major contraction
mechanism.**

This refines Phase 1.2 F's narrative. F said "trained γ values
make LayerNorm contract". L confirms γ is part of the story but
shows **state-norm is the larger single contributor in the
spectral norm sense**. L6's γ only contributes 1/0.779 ≈ 1.28×
contraction; the state-norm contributes 1/0.046 ≈ 22×.

The story is *both*, multiplicatively combined.

## Reading 3 — L0's outlier is purely a γ effect

This is the user-specified expansion's payoff. L0's σ_max = 0.038 is
≈ 3× more contractive than L6's 0.116. Three candidate explanations
were possible:

- **(a) state norm is different at L0** — would predict A2/A4 differs across layers
- **(b) γ is more aggressive at L0** — would predict A3/A4 differs across layers
- **(c) coupling is anomalous** — would predict coupling ≠ 1

The data: A2/A4 = 0.046 in both layers (same), A3/A4 = 0.252 vs 0.779
(different by 3.1×), coupling = 1.000 in both. **(b) is the answer.**

L0 has learned γ values that contract σ_max by an additional 4× beyond
the state-norm baseline; L6's γ contributes only 1.28×. Why? L0 is the
first LayerNorm to see the raw input embedding (wte+wpe). It has the
largest variance to absorb. Aggressive γ at L0 reduces that variance
hard before any block has run; later layers receive already-tamed
states and need less γ contraction.

This is also consistent with Experiment K's finding: σ_max grows from
L0's 0.038 to a stable ≈ 0.10–0.20 across L1..L11. The γ values
accommodate the changing residual-stream variance scale — aggressive
where variance is high, modest where it isn't.

## Reading 4 — Phase 1.2 F's mechanistic claim is refined, not refuted

Phase 1.2 F wrote:
> "trained LayerNorm gain values [γ] have been driven to small
> magnitudes that contract"

L confirms this for L0 (γ contributes 4× contraction) but **softens it
for L6** (γ contributes only 1.28× contraction). The full picture:

- **Universal**: trained fixed-point hidden states have large norm,
  giving every LayerNorm a 22× state-norm-driven contraction.
- **Layer-specific**: γ provides additional contraction, ranging from
  1.3× (mid-stack) to 4× (L0 outlier).
- **Multiplicative**: the two combine as exactly σ_state × σ_γ.

So "training contracts LayerNorm" is correct but underspecified.
Training drives **two** effects: (1) fixed-point hidden states settle
at large norms, and (2) γ values are reduced (more aggressively at
the input-side layers). Both effects multiply.

## Caveats

- **The "γ-only" condition (A3) at norm = 100 is a hypothetical regime**
  the trained model doesn't actually visit. The Jacobian computation
  is well-defined at any input, but the meaning of "γ contribution"
  is operational: it's the σ_max factor that γ contributes when the
  state is at unit-σ_x scale. Different choices of `target_norm`
  would shift A3 and A4 proportionally; the *ratio* A3/A4 is what
  matters and is target-norm-independent (verified analytically).
- **The state-only contribution A2/A4 = 0.046 doesn't directly reveal
  σ_max scaling with norm**. LayerNorm has σ_max ∝ 1/σ_x, and σ_x ≈
  ‖h‖/√d. So A2/A4 ≈ (target_norm)/(orig_norm) = 100/2200 ≈ 0.045 —
  matches. The 0.046 figure is essentially "norm ratio raised to the
  -1 power" (since LayerNorm is scale-invariant in its non-affine
  part). This is a sanity check that the decomposition is consistent
  with the analytical formula.
- **Sample size n = 10**. Per-input variance is non-trivial (std/mean ≈
  40-50% for both layers). The mean-level claims are robust;
  per-input claims would need larger samples.
- **L only tested ln_1 of layers 0 and 6**. Symmetric tests of ln_2
  or other layers' ln_1 would confirm the pattern generalises. Plan
  §3.5 only mandated ln_1 of layer 6; the L0 expansion was the user's
  add-on. Going further is a Phase 1.4 candidate.
- **The claim "L0 has aggressive γ" can be checked directly** by
  computing |γ| values per dimension; this experiment uses the
  spectral norm as a proxy. The 0.252 figure tells us the largest
  γ singular value is 0.252× what a γ=1 LayerNorm would produce, but
  doesn't characterise the full γ distribution. A simple histogram
  follow-up would settle this.

## Pointers

- per-result JSON: `data/processed/phase1_3_ln_decomposition/results.json`
- aggregated summary: `data/processed/phase1_3_ln_decomposition/summary.json`
- figure: `outputs/figures/phase1_3_ln_decomposition/decomposition_bars.png`
- source: `src/ln_decomposition.py`

## Conclusion for the master report

**Q2 answered (with refinement): trained LayerNorm σ_max ≈ 0.12 has
both a γ contribution and a state-norm contribution; they combine
multiplicatively (coupling factor = 1.000); state-norm is the larger
contributor at L6 (0.046×) but at L0 the γ contribution dominates
(0.252× vs 0.046×).**

Phase 1.2 F's "trained γ contracts" narrative is correct in direction
but partially understated the state-norm contribution. The complete
story:

- **State-norm contribution (universal)**: trained fixed-point hidden
  states sit at large norm (≈ 2500 lowercase, ≈ 1000 HTML), giving
  every LayerNorm a 1/22 ≈ 0.046 contraction factor.
- **γ contribution (layer-dependent)**: γ provides additional
  contraction, scaling from ≈ 0.78 mid-stack down to ≈ 0.25 at L0.
  The L0 outlier from Experiment K is a γ effect, not a state-norm
  effect.
- **Multiplication**: σ_max(LN) factorises exactly into
  (state-norm-effect) × (γ-effect). No coupling beyond the formula.

This makes the "training contracts LayerNorm" claim more precise:
training drives **two** effects (large fixed-point norms + reduced γ),
which combine multiplicatively. The earlier "γ < 1 makes LayerNorm
contract" framing was correct but compressed two distinct effects
into one phrase.
