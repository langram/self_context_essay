# Phase 1.1 — Experiment B: Nesting / outlier deep-dive

- timestamp: see `data/processed/phase1_1_html_outlier/summary.json`
- model: `gpt2`, fp32, CUDA
- input zoo: 30 inputs, 6 categories × 5 (HTML, XML, JSON, LaTeX, pure brackets, Markdown lists)
- seq_len = 64, max_iter = 1000, threshold = 1e-3, save_every = 10
- elapsed: 17 s

## Headline result: Phase 1's "universal attractor" was not universal

All 30 nested-structure inputs converge within 1000 steps. But they do **not**
all land at the same fixed point. Comparing each new C\* (mean-pooled
[seq_len, 768] → 768-vec, normalised) to a representative Phase 1 universal
attractor:

| input cluster | n | n_steps | eff_rank | ‖C\*‖ | cos to Phase 1 universal |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Phase 1 "lowercase" attractor** (xml#9, latex#15-16, markdown#28) | 4 | 10–12 | 1.02 | 2564 | 1.0000 |
| **HTML/XML "capital" attractor** (html#0-4, xml#5-8, latex#18) | 9 | 109–129 | 1.11 | 1042 | **0.4903** |
| Pure-brackets attractor (all 5) | 5 | 50 | 1.49 | 2453 | 0.9975 |
| Markdown attractor (4 of 5) | 4 | 12 | 1.24 | 2538 | 0.9995 |
| JSON attractor (5/5) | 5 | 38–41 | 1.54–1.59 | 2412–2431 | 0.9947–0.9962 |
| LaTeX attractor pair (#17, #19) | 2 | 61–65 | 1.86 | 2060 | 0.985 |

So mode-A iteration on GPT-2 small actually has **at least two structurally
distinct strong attractors**, plus several intermediate "almost the universal,
but slightly off" basins. The HTML/XML cluster lands at a fixed point with
mean-pooled cosine **0.49** to Phase 1's "universal" — these are not the same
state in any reasonable sense.

This refutes the Phase 1 narrative that the trained model collapses every
input to a single trivial attractor. Phase 1's 60 inputs were all
natural-language-shaped, and they all happened to lie in one basin. The
basin landscape is richer than that.

## What separates the two strong attractors? Lexical register, not depth.

Projecting each cluster's C\* through `ln_f + lm_head` gives the qualitative
signature:

| cluster | top-5 at each position |
| --- | --- |
| Phase 1 "lowercase" attractor | ` the / , / \n / and / .` (sentence-medial connectors) |
| HTML/XML "capital" attractor | `The / In / For / I / A` (sentence-initial capitalised words) |

The HTML attractor's top-5 are all capitalised sentence-openers. The Phase 1
attractor's top-5 are all lowercase mid-sentence tokens (note the leading
space on ` the` and ` and`). These are two different *registers* the model
is converging to.

The hybrid attractors (markdown, brackets, JSON) all show position 0 as
capital tokens and later positions as lowercase — a rank-2-like compromise
between the two strong attractors.

## Convergence is not predicted by nesting depth

Within each lexical category, **all five inputs converge in essentially the
same step count regardless of nesting depth**:

- HTML depth 1 → step 129; HTML depth 5 → step 129
- Pure brackets depth 1 (`( )`) → step 50; depth 5 (`( ( ( ( ( ) ) ) ) )`) → step 50
- Markdown list depth 1 → step 12; depth 6 → step 12

The Phase 1 plan §3.3 hypothesised "抗坍缩 vs 嵌套层数 是否相关" (does anti-collapse
correlate with nesting depth?). The answer is **no**. Two inputs from the same
lexical category converge to the same C\* in the same number of steps,
regardless of how nested they are. The relevant variable is **surface
character distribution / sentence-start probability**, not structural depth.

## Per-input table

See `data/processed/phase1_1_html_outlier/records.json`. Selected outputs:

```
idx category       depth conv n_steps eff_rank ||C*||  cos_to_phase1_universal
  0 html               1 True     129     1.11   1042   0.4903   ← capital basin
  4 html               5 True     129     1.11   1042   0.4903   ← capital basin
  9 xml                4 True      11     1.02   2564   1.0000   ← phase1 basin
 15 latex              1 True      12     1.02   2564   1.0000   ← phase1 basin
 18 latex              3 True     109     1.12   1044   0.5950   ← capital basin
 20 pure_brackets      1 True      50     1.49   2453   0.9975   ← hybrid
 24 pure_brackets      5 True      50     1.49   2453   0.9975   ← hybrid (same C*)
 25 markdown_list      1 True      12     1.24   2538   0.9995   ← hybrid
 28 markdown_list      3 True      10     1.02   2563   1.0000   ← phase1 basin
```

Note `xml#9` lands at the Phase 1 attractor, but `xml#5–8` (which look superficially
similar) land at the HTML attractor. The difference is `xml#9`'s leading
`<?xml version="1.0"?>` — at seq_len=64 with the `?` and `version` tokens, the
input distribution leans toward natural text and the basin pull flips.

## Implications for the Phase 1 narrative

**Q3 answered.** Phase 1's HTML outlier was real but mislabelled. It was not
"the one input that escaped collapse" — it was "the one input drawn from a
lexical class that has its own attractor basin". In Phase 1 the HTML basin
required >100 steps to settle, so within Phase 1's max_iter=100 budget it
showed up as non-converging. With max_iter=1000 it converges cleanly to a
fixed point that is structurally distinct from the Phase 1 universal one.

**For the master report:**

- The four alternative explanations from plan §1.1 should now read:
  1. *FPP hypothesis literally wrong* — weakened. There exist at least two
     strong basins; trained-model dynamics have non-trivial structure.
  2. *Interface mismatch* — orthogonal. Mode A still produces multiple
     attractors; the issue isn't that mode-A is unreadable, it's that the
     Phase 1 input zoo only sampled one basin.
  3. *Engineering artefact (LayerNorm + position embedding)* — partially
     supported by the Capital-vs-lowercase split (the position embedding's
     position-0 capitalisation pull seems involved), but does not explain
     everything (within-category C\*'s are identical regardless of depth).
  4. *Semantics live in transient* — still alive (Experiment A confirmed),
     and now also clearly lives in *which basin you fall into*, not just the
     trajectory.

**What to do next**: Experiment C (position-embedding ablation) becomes
particularly load-bearing. If removing repeated `wpe` injection collapses
the Capital and lowercase basins into one, the Capital basin is a position-
embedding artefact. If they remain separate, the basins reflect genuine
trained structure (different registers the model has learned to
discriminate).

## Caveats

- 30 inputs × 6 categories × 5 each — small per-category sample. The "all
  5 HTML converge to identical C\*" observation is striking but the per-class
  sample size is 5, not 50.
- seq_len=64 differs from Phase 1's seq_len=32, so the cos-to-phase1
  comparison is via mean-pool rather than direct alignment. The 0.49 figure
  is robust to this choice (verified by spot-check at first/last 32 positions),
  but treat it as approximate.
- "Identical C\*'s within a basin" means cosine similarity 1.0000 to 6
  decimals on the mean-pooled vector. Per-position differences may exist
  below the rounding floor.
- Random-init twin not run on this zoo — the question was about trained-model
  basin structure, and the answer is informative regardless of random
  baseline. A random twin run would cost another ~17 s and be a useful
  appendix if the master report needs it.

## Pointers

- raw traces (with save_every=10): `data/raw/phase1_1_html_outlier/trace_*.pt`
- per-input records: `data/processed/phase1_1_html_outlier/records.json`
- 30×30 cos similarity matrix: `data/processed/phase1_1_html_outlier/similarity.npz`
- aggregate summary: `data/processed/phase1_1_html_outlier/summary.json`
- figures: `outputs/figures/phase1_1_html_outlier/{convergence_vs_depth,non_convergence_traces,cos_to_phase1_universal}.png`
- source: `src/extended_inputs.py`, `src/run_html_outlier.py`

## Conclusion for the master report

**Q3 answered: HTML's anti-collapse in Phase 1 was a basin selection effect,
not a depth-of-nesting effect.** The "universal attractor" of Phase 1 is one
of at least two strong attractors plus several intermediate basins. Lexical
register (capital sentence-starts vs lowercase mid-sentence connectors) is
the discriminator, not structural nesting depth.

This makes Experiment C (position-embedding ablation) particularly important:
if removing `wpe` injection merges the basins, position embedding is doing
real work; if not, the basins reflect trained register structure.
