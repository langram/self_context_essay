# Phase 1.3 — Experiment I: Markup-type triangulation

- model: `gpt2`, fp32, CUDA
- input zoo: 30 inputs, 6 bracket-type categories × 5 (angle / square / curly / parens / quotes / isolated single chars)
- mode-A iteration, max_iter = 1000, threshold = 1e-3, save_every = 10, seq_len = 64
- elapsed: 10.7 s

## TL;DR

**The HTML-induced metastable structure is angle-bracket-specific, not a
general "markup" phenomenon.** Only `<...>`-shaped inputs land at the
attractor at cos = 1.000 to the Phase 1.1 B HTML reference. Curly braces
and quotes converge cleanly to the lowercase attractor; square brackets
and parentheses produce intermediate states distinct from both reference
attractors.

The Phase 1.2 H finding "markup is the sole basin selector" was correct
*for that zoo's choice of markup* (exclusively `<...>`). With a wider
bracket zoo, the picture is **richer**: at least 4 distinguishable
endpoint regimes by bracket type.

## Per-category aggregate

| category | mean n_steps | mean eff_rank | mean ‖C\*‖ | mean cos_lc | mean cos_html |
| --- | ---: | ---: | ---: | ---: | ---: |
| **angle** `<tag>...</tag>` | **129.0** | 1.11 | **1042** | 0.503 | **1.000** |
| square `[tag]...[/tag]` | 21.4 | 1.27 | 2217 | 0.901 | 0.615 |
| curly `{tag}...{/tag}` | 10.6 | 1.02 | 2564 | **1.000** | 0.490 |
| parens `(tag)...(/tag)` | 50.0 | 1.49 | 2453 | 0.997 | 0.540 |
| quotes `"tag"..."tag"` | 10.8 | 1.02 | 2563 | **1.000** | 0.490 |
| isolated single chars | 36.0 | 1.13 | 2243 | 0.900 | 0.601 |

Reference attractors:
- **lowercase attractor**: ‖h‖ = 2563, cos to itself = 1.000, cos to html_metastable = 0.490
- **HTML metastable structure**: ‖h‖ = 1042, cos to itself = 1.000, cos to lowercase = 0.490

Reading rules: cos_lc ≈ 1.0 means the input collapses to the lowercase
attractor; cos_html ≈ 1.0 means it lands at the HTML metastable structure;
intermediate values mean somewhere else in hidden-state space.

## Reading by category

**Angle brackets (5/5 inputs):** all five reach the HTML metastable
structure exactly. Same n_steps (129), same ‖C\*‖ (1042), cos to HTML
ref = 1.000. This reproduces Phase 1.2's HTML behaviour cleanly. **The
HTML structure is angle-bracket-specific.**

**Curly braces (5/5):** all five reach the lowercase attractor in 10–11
steps. Indistinguishable from natural-text inputs. The `{` and `}`
characters do not trigger any secondary structure.

**Quotes (5/5):** same as curly braces — fast convergence to lowercase
attractor (~11 steps), cos_lc = 1.000.

**Parens (5/5):** end up at cos_lc = 1.00 but the convergence path is
different — ‖C\*‖ = 2453 (lower than lowercase's 2564), eff_rank = 1.49
(higher than lowercase's 1.02), n_steps = 50 (slower than 10). This
matches Phase 1.1 B's "pure_brackets" behaviour: lands in the lowercase
basin but at a distinct, lower-rank corner of it. **There is intra-basin
structure.**

**Square brackets (5/5):** **a third regime.** cos_lc = 0.90 (close
but not at the lowercase attractor), cos_html = 0.62 (closer than to
lowercase but not at HTML). ‖C\*‖ = 2217. Intermediate state in
hidden-state space. This was not visible in Phase 1.1 B because
square brackets weren't in the input zoo.

**Isolated single chars (5/5):** also intermediate (cos_lc = 0.90,
cos_html = 0.60). Notably, single `<` or `>` alone does **not** drive
the system to the HTML structure — only the full `<...>` token-sequence
pattern does. This was the Phase 1.2 plan's third hypothesis ("token
*pattern* not single-token trigger") and the data confirms it.

## What this answers

**Q1 answered (decisively): the secondary attractor is triggered by
angle-bracket BPE tokens specifically, in their tag-pattern context.**
Square, curly, parens, and quote characters do not trigger the HTML
structure. Square brackets and single-char isolates produce *different*
intermediate states.

This narrows Phase 1.2 H's "markup is the sole basin selector":
- The trained model has **at least three** distinguishable endpoint
  regimes (lowercase attractor, HTML metastable structure, square-
  bracket intermediate state).
- The HTML structure is **specific to angle-bracket BPE token IDs**,
  not to "markup" or "tag-like characters" as a category.
- Different bracket types map to different endpoints — so basin
  geometry has more structure than Phase 1.2 H showed.

## Implications for Phase 1.2's claim

Phase 1.2 H wrote *"markup is the sole basin selector among the four
factors tested"*. Experiment I refines this:

1. The "markup" factor in Phase 1.2 H = "presence of angle brackets".
   What was tested wasn't really markup as a category, just one specific
   bracket type.
2. The clean 12/12 vs 0/12 result was a Phase 1.2 H zoo artefact:
   expanding to 30 inputs across 6 bracket types reveals 4–6
   endpoint regimes, not 2.
3. Phase 1.3's §0.1 naming discipline is justified: the "HTML-induced
   metastable structure" naming is right; the "capital basin" naming
   from Phase 1.2 was misleading because the basin is angle-bracket-
   induced, not capital-induced or markup-induced.

## Unexpected sub-findings

- **Square brackets produce a state distinct from both references.**
  This wasn't in any pre-committed prediction. It suggests there may
  be more than 2–3 attractor-like regions in hidden-state space.
- **Parens land at lowercase but via a slow path.** 50 steps and
  rank 1.49 differs from natural-text's 10 steps and rank 1.02 even
  though both end at cos_lc = 1.0. The lowercase basin contains
  internal sub-structure that different inputs explore differently
  — relevant to Experiment J's "tail rate" diagnostic.

## Caveats

- 5 inputs per category is small. Square's intermediate state could be
  a slowly-converging trajectory toward lowercase (would resolve at
  larger max_iter); a follow-up at max_iter = 10000 on these specific
  inputs would tell us. But max_iter = 1000 was already 10× Phase 1.2's
  budget; if 1000 isn't enough, the "intermediate" region is at minimum
  a long-lived metastable state, which is itself informative.
- "Intermediate" is loosely defined as "cos < 0.95 to either reference".
  The geometry of these intermediate states is not yet characterised
  (is it one attractor or multiple? is it a stable point or a
  trajectory?). Phase 1.3 J's long-trace experiment will give first
  data on this for the HTML structure; analogous probes for square
  and parens are Phase 1.4 candidates.
- The "isolated single chars" inputs were constructed by adding fluff
  text after the lone bracket char to fill seq_len. Their endpoint
  states are influenced both by the bracket char AND by the fluff
  text. A cleaner test would put the bracket alone with EOS padding,
  but EOS-padded inputs have known artefacts in iteration. Live with
  the imperfection; it's enough to see that single chars don't trigger
  HTML structure.

## Pointers

- raw traces: `data/raw/phase1_3_markup_triangulation/trace_*.pt`
- per-input records: `data/processed/phase1_3_markup_triangulation/records.json`
- aggregate summary: `data/processed/phase1_3_markup_triangulation/summary.json`
- figure: `outputs/figures/phase1_3_markup_triangulation/cos_scatter.png`
- source: `src/markup_triangulation.py`

## Conclusion for the master report

**Q1 answered: the HTML metastable structure is triggered by angle-
bracket BPE token IDs specifically, not by "markup" as a category.**
The Phase 1.2 H "markup is the sole basin selector" claim was correct
for its single-markup-type input zoo but does not generalise. The
trained model's hidden-state geometry has at least 3 distinguishable
endpoint regimes (lowercase / HTML / square-bracket intermediate),
plus internal structure within the lowercase region (parens land there
but at a distinct corner with rank 1.49).

This further weakens any "GPT-2 has rich attractor structure that
encodes per-input semantics" reading: the multiple endpoint regimes
revealed by I are still *not* per-input semantic — they are
bracket-type-specific. But it also weakens the Phase 1.2 picture's
simplicity: there are more than 2 attractors.

The naming discipline in Phase 1.3 §0.1 is now empirically justified.
The Phase 1.2 "Capital basin" should retire entirely; the right name
is **"angle-bracket-induced metastable structure"**, with ontology
status (fixed point vs metastable transient vs limit cycle vs
wpe-forced) to be settled by Experiment J.
