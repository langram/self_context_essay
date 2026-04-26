# Phase 1.1 — Experiment D: Mode C token interface

- model: `gpt2`, fp32, CUDA
- input zoo: same Phase 1 60-input set
- pipeline: `t_n → forward → logits → argmax(per-position) → t_{n+1}`
- convergence test: `t_n == t_{n+1}` (sequence equality) **OR** sequence appears in earlier history (cycle)
- max_iter = 50
- elapsed: 26 s

## Headline result

| outcome | count |
| --- | ---: |
| reached a token-fixed-point (t_n = t_{n+1}) | **0 / 60** |
| entered a limit cycle within 50 steps | **4 / 60** |
| neither — still wandering at step 50 | **56 / 60** |
| distinct final sequences (at step 50) | **48 / 60** |

Discrete argmax iteration on GPT-2 small does **not** find a fixed point in
token space. Mode A's universal attractor in hidden space (10-step convergence,
1.02 effective rank, all 60 inputs at the same point) becomes, under mode C,
a **chaotic-looking trajectory in token space** that doesn't settle in 50
steps. 48 of the 60 inputs end up at a unique-to-themselves token sequence
at step 50 — diversity is preserved at the trajectory level, lost at the
fixed-point level.

## The four limit cycles

| idx | category | input excerpt | period | step at detection | sample of final tokens |
| ---: | --- | --- | ---: | ---: | --- |
| 16 | gram_random_semantics | "Quiet algebra danced..." | 4 | 50 | `\n/lib\n following\nusr\n//usr\n///////\n\n` |
| 25 | common | "Coffee shops in this neighborhood..." | 8 | 32 | `\n/lib\n/\n\nusr\nusr\n/\n/\n/\nusrusr//////usr\n\n\n\n\n\n` |
| 37 | ambiguous | "Visiting relatives can be exhausting..." | 16 | 44 | `\n\n_ b\n\n\n.\na\n_\n\n\n\n\nIIII\n\nIII\n\n\n\n_\n` |
| 49 | nonsense_grammatical | "Quiet contradictions inhabit..." | 8 | 32 | `\n/lib\n/\n\nusr\nusr\n/\n/\n/\nusrusr//////usr\n\n\n\n\n\n` |

Two of the four cycles (idx 25 and idx 49) end up at *the same cyclic state*
— a "Linux-path soup" mix of `/lib`, `usr`, `/`, `\n` tokens. Two completely
unrelated inputs (a coffee shop sentence, a nonsense-grammatical "Quiet
contradictions...") land in the same period-8 cycle. This is the closest
mode C gets to mode A's "universal attractor" behaviour: a *cyclic* pseudo-
attractor that two inputs share.

The other two cycles are idiosyncratic (period 4 and 16, distinct token
content).

## What the trajectories look like

For one random-token input, the first 8 steps:

```
step 0: " divest anxYou coasts Oz Vi Happy appreciate tcp guilty"
step 1: ",ieties are on,agra Birthday your- of"
step 2: " the, not the and,, birthdayself-"
step 3: " the and the,, and and of.ident"
step 4: " thero other and and the the,\nity"
step 5: " thegyn than the the otherro and\n,"
step 6: " theous the othergyn waygyn the\n as"
step 7: " the,ous,ous.. other\nous"
```

Within a few steps the sequence becomes a soup of high-frequency tokens
(` the`, `,`, `\n`, `and`, `the`) interspersed with lower-frequency tokens
that change at every step. **It never settles** — by step 50 it is still
re-shuffling these tokens.

The frequency tokens are exactly Phase 1's universal-attractor projection
(` the / , / \n / and / .`). So mode C is being pulled toward the same
unigram-prior region as mode A, but the discreteness of argmax prevents
it from reaching a stationary point. Instead the system perpetually
oscillates between configurations of high-frequency tokens.

## Final sequences at step 50 — semantic content

A representative sample from each category:

| input category | input excerpt | final at step 50 |
| --- | --- | --- |
| random_tokens | "divest anxYou..." | ` thegyn\n11 . .( . ..... .\n.....\n\n(((((::\n\n` |
| gram_random_sem | "The blue eight runs..." | `\nThe\n\n_\n_\n _\n\n\n _\n___\n _oryoryTheTheThe\n"\n\n\n"""` |
| common | "The cat sat on the mat..." | `\n first following's\na\n'm\nIIIIIIIThe\n\n\n\n\n"""___"\n"\n` |
| ambiguous | "The trophy doesn't fit..." | `\nTheTheTheTheTheI\n\n\n\n\n"\n\n__\n\n\n\n\n__\n\n\n"""\n\n` |
| nonsense_gram | "Colorless green ideas..." | `\nTheThea\n'm\nI\n\n\n\n\n\n\n\n\n""_\n\n\n\n\n"\n"""_"` |
| structured | "def fibonacci(n)..." | `\n_ the\n of,al\n first firstoryThe'm\nThe"""\n\n\n\n\n""\n\n\n"""\n` |

None of these are coherent text. They are all soups of high-frequency
tokens — `\n`, `The`, `I`, ` the`, `"`, `_`, `(`. The pattern is
**"unigram-prior tokens scrambled across positions, with no syntactic
structure"**.

## Reading

**Q4 answered: token-level interface (mode C) does not restore semantic
attractor diversity.** It does dramatically change the dynamics — no fixed
point, mostly non-cyclic, 48/60 distinct final sequences — but the
diversity is *trajectory-level chaos*, not *attractor-level structure*.
The system is still being pulled toward the unigram prior; it just can't
land there because argmax forces discrete jumps.

This decisively distinguishes mode C from mode A:

- Mode A (continuous hidden interface) → smooth contraction → universal
  attractor (or a small number of basins per Experiment B).
- Mode C (discrete token interface) → no contraction, perpetual rearrangement
  among high-frequency tokens.

Both modes are dominated by the model's vocabulary prior — the prior
manifests as a fixed point in mode A and as a pseudo-cyclic high-frequency
soup in mode C.

**This is informative pushback against the Phase 1 plan §1.1 Explanation 2
("interface mismatch — last_hidden_state is OOD; argmax token interface
fixes it").** The argmax interface does change the dynamics fundamentally,
but it does not surface a cleaner attractor structure. Whatever the model
"contains" that mode A misses (per Experiment A's transient probe finding),
mode C also misses it.

## Caveats

- max_iter = 50 was the planned budget. Several inputs might converge or
  enter a cycle later. A follow-up at max_iter = 1000 would be cheap (a
  few minutes) and would tell us how many of the 56 "neither" cases would
  eventually find a cycle. Recommended as a Phase 1.2 diagnostic if the
  master report flags this as load-bearing.
- argmax is a strict winner-take-all; even a 1% chance of a different top-1
  token across iterations breaks fixed-point convergence. The
  non-convergence may not reflect "no real attractor" so much as "the
  attractor is in distribution space (a top-K cloud), not in argmax space".
  A natural Phase 1.2 follow-up is mode B (temperature sampling with
  fixed seed) — same pipeline, sample instead of argmax. Plan §3.6
  explicitly defers mode B; it's a clean follow-up.
- Cycle detection uses exact list equality, which is very strict. Near-
  cycles (sequences that differ by one token) won't be detected. The
  4 cycles found are genuine; the 56 "neither" cases may include many
  near-cycles that this scheme can't see.
- 60 inputs is small for cycle-frequency statistics. Treat the 4/60
  cycle rate as a qualitative observation, not a quantitative claim.

## Pointers

- raw trajectories: `data/raw/phase1_1_mode_c/trace_*.pt` (each contains the full token-id sequence per step)
- per-input records: `data/processed/phase1_1_mode_c/records.json`
- aggregate summary: `data/processed/phase1_1_mode_c/summary.json`
- figures: `outputs/figures/phase1_1_mode_c/{convergence_breakdown,step_distribution}.png`
- source: `src/mode_c_iterate.py`, `src/run_mode_c.py`

## Conclusion for the master report

**Q4 answered: token interface does not restore semantic attractor
diversity.** It produces trajectory-level chaos (48 distinct step-50
sequences) but no fixed point and only 4 limit cycles. The dynamics are
qualitatively different from mode A but pulled by the same unigram-prior
sink.

Combined with Experiments A, B, C, this gives a sharper picture of the
mode-A "universal attractor" finding from Phase 1: it is one of several
basins in continuous hidden-state dynamics; it is not a position-embedding
artefact; and switching to discrete argmax interface changes the dynamics
qualitatively but doesn't expose a richer attractor structure underneath.
The natural next step (Phase 1.2 candidate, **not** in scope for Phase 1.1)
is the LayerNorm Lipschitz / Jacobian diagnostic the plan §3.6 marks as
deferred.
