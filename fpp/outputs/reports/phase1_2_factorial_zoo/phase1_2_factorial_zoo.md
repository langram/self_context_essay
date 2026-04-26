# Phase 1.2 — Experiment H: Factorial input zoo

- model: `gpt2`, fp32, CUDA
- input zoo: 24 inputs, 2 (case) × 2 (markup) × 2 (punct) × 3 (content), seq_len = 64
- max_iter = 1000, threshold = 1e-3, save_every = 10
- elapsed: 13.8 s
- input zoo human-reviewed and approved before run (per plan §3.5 / §9.2)

## Headline result — markup is the sole basin selector

Across the 24-input factorial zoo, basin assignment is **completely
determined by markup presence**. Case, punct density, and content type
have **zero** measurable effect.

| factor | n within-factor-flip pairs | n basin-flip pairs | flip rate |
| --- | ---: | ---: | ---: |
| case (capital ↔ lowercase) | 12 | 0 | **0.000** |
| **markup (with ↔ without)** | **12** | **12** | **1.000** |
| punct (high ↔ low) | 12 | 0 | **0.000** |
| content (natural / code / random) | 24 | 0 | **0.000** |

Basin distribution is exactly partitioned by markup:

| basin | n | composition |
| --- | ---: | --- |
| capital (cos = 0.50 to phase1 universal) | 12 | every input with `<...>` |
| lowercase (cos = 1.00 to phase1 universal) | 12 | every input without `<...>` |
| hybrid | 0 | — |

## Per-input table

The 24 inputs partition into exactly two final states:

| input class | n | n_steps | eff_rank | ‖C\*‖ | cos to phase1u |
| --- | ---: | ---: | ---: | ---: | ---: |
| with markup (idx 0,1,4,5,8,9,12,13,16,17,20,21) | 12 | 129 | 1.11 | 1042 | 0.5026 |
| without markup (idx 2,3,6,7,10,11,14,15,18,19,22,23) | 12 | 9–10 | 1.02 | 2563 | 1.0000 |

Within-class C\*'s are pairwise identical to floating-point precision
(same cos, same norm, same rank). Once an input lands in a basin, the
basin is so strongly attracting that all inputs reach literally the
same point.

## What this answers

**Q2 answered: among the four factors tested, markup is the only
factor that selects basin.**

Phase 1.1 B's "Capital basin vs lowercase basin" naming was about the
LM-head projection (capital words `The/In/For/I/A` vs lowercase
` the/,/\\n/ and/.`). The naming was descriptively correct but
mechanistically misleading: the *selector* was assumed to be
case-related (sentence-start vs sentence-medial register), but Phase
1.2 H shows the selector is **markup token presence**, not case.

The Capital basin's projected tokens are capitalised because the
attractor sits in a region of hidden-state space that the LM head
maps to capitalised vocabulary tokens. Inputs *get there* by virtue
of containing `<...>` tokens, regardless of their own letter case.

## Combined picture with E + F + G

This is the cleanest account of GPT-2 small mode-A iteration we have:

1. **One architectural attractor** (Experiment F): trained LayerNorms
   contract σ_max ≈ 0.12 toward a fixed point; the architectural
   attractor is the lowercase / unigram-prior fixed point. Random init
   has no contraction (σ_max = 1.00 across all sub-modules).
2. **One position-driven secondary attractor** (Experiment G): with
   continuous `wpe` re-injection, markup-token sequences are pulled
   into a secondary attractor (the capital basin). Without `wpe`
   re-injection (cancel-pos), 100% of capital-basin inputs migrate
   out — the secondary attractor disappears.
3. **Markup is the trigger** (Experiment H, this report): only
   `<...>`-containing inputs end up in the capital basin. Case, punct,
   and content type are absorbed by the LayerNorm contraction.
4. **Step-10 hidden state encodes basin label plus residual
   sub-basin structure** (Experiment E): probe-basin 0.90 vs chance
   0.40; within-basin probes vary from 0.75 (lowercase, near chance)
   to 1.00 (hybrid).

The (context, attention) coupled-system framing in essay §4 needs to
become **(context, attention, normalisation) + (position-embedding
secondary basin)** to be honest to the data.

## Why is the result *so* clean?

The 12/12 vs 0/12 vs 0/12 vs 0/24 result is striking. Possible
readings:

- **Real and architectural.** Markup tokens have specific position-
  embedding interactions that produce the secondary basin; other
  surface factors don't have a comparable mechanism. This is consistent
  with Experiment G's finding that the capital basin disappears
  without `wpe`.
- **Tokeniser-driven.** The angle bracket characters `<`, `>`, `/`
  may produce specific GPT-2 BPE tokens that sit at distinct positions
  in the embedding manifold. Inputs containing those tokens get
  pushed in a direction that the trained LayerNorm contraction
  cannot fully nullify.
- **Zoo-design-amplified.** Our markup is exclusively `<...>`.
  Replacing markup with `[...]` or `{...}` or `"..."` might give a
  different basin assignment (or no secondary basin). This is a
  Phase 1.3 candidate.

The cleanness is consistent with Phase 1.2 G's finding that the
secondary attractor's existence depends on `wpe` interaction with
specific token sequences. If `wpe` only "engages" with markup
patterns, then markup is the only feature that selects basin.

## Caveats — read before claiming "markup determines basin"

- **n = 24 is small.** With perfect 12/0 vs 0/12 vs 0/12 vs 0/24
  rates, the data does not have the resolution to detect modest
  effects. A 5% true effect from punct or content would not show
  up in this zoo. The right read is: **markup has a much stronger
  effect than any of the other three factors at the magnitudes we
  tested**, not "markup is the only thing that ever matters".
- **"Markup" here means specifically `<...>` angle-bracket tags.**
  Not parentheses, not braces, not quotes. The Phase 1.1 B "pure
  brackets" inputs (parentheses) all landed in the lowercase /
  hybrid basins, not capital. So the basin trigger is more specific
  than "markup-style tokens" — it's specifically angle brackets,
  and likely the BPE token IDs they produce. **A different language
  with different BPE tokens for `<`/`>` might give different
  results entirely.**
- **Within-basin C\*'s are identical to the precision shown** —
  this is informative for the contraction strength but means the
  factorial design cannot study sub-basin variation. Experiment E's
  within-basin probes did show some sub-basin information for the
  hybrid basin, but the capital and lowercase basins (which fully
  populate this zoo) are essentially information-free at C\*.
- **Plan §11's "everything explains itself" warning applies here.**
  The picture across E + F + G + H is suspiciously coherent. Either
  the diagnostic suite is correctly nailing down the actual
  mechanism, or we're systematically missing a confound. Two
  candidate confounds to flag for outside review:
  1. The same `<` and `>` tokens may be triggering both the
     "capital basin" *and* a position-embedding interaction; we
     have not separated "markup-tokens" from "wpe-interaction-with-
     specific-token-positions".
  2. All three content types in this zoo were constructed by hand
     for clean factor isolation. They are *not* a representative
     sample of GPT-2's training distribution. A run on a larger,
     unbiased input zoo could give different basin distributions.

## Pointers

- raw traces: `data/raw/phase1_2_factorial_zoo/trace_*.pt`
- per-input records: `data/processed/phase1_2_factorial_zoo/records.json`
- aggregate summary: `data/processed/phase1_2_factorial_zoo/summary.json`
- figures:
  - `outputs/figures/phase1_2_factorial_zoo/factor_flip_rates.png`
  - `outputs/figures/phase1_2_factorial_zoo/per_input_basin.png`
- input zoo source: `src/factorial_zoo.py`
- runner source: `src/run_factorial_zoo.py`
- approved candidate review: `outputs/reports/phase1_2_factorial_zoo/candidate_zoo_for_review.md`

## Conclusion for the master report

**Q2 answered: markup (specifically `<...>` angle-bracket presence) is
the sole basin selector among the four tested factors.** Case, punct
density, and content type produce 0% basin flip rate; markup produces
100%. The Phase 1.1 B "Capital basin" should be re-named to
"markup-induced basin" — the LM-head projection is capitalised words
because the attractor sits in a hidden-state region that maps to
capital tokens, not because the inputs themselves are capitalised.

Together with Experiments E + F + G, the Phase 1.2 picture is a
single coherent mechanistic account: trained LayerNorms contract toward
a single architectural attractor; angle-bracket tokens interact with
position embedding at every iteration to maintain a secondary
attractor; without either the markup tokens or the position embedding,
the secondary attractor collapses into the architectural one.

The plan §11 warning about "everything explains itself" applies — this
account is suspiciously coherent on a small and hand-constructed input
zoo. A natural Phase 1.3 candidate is to test whether the basin
selector is "angle brackets specifically" or "any tag-like
character". The factorial design should be expanded with
brackets/quotes/braces as additional markup variants.
