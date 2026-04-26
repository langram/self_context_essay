# FPP Phase 1.1 — Master Report

**Project:** FixedPointProbe (FPP)
**Phase:** 1.1 — diagnostic post-mortem of Phase 1's universal-attractor finding
**Date:** 2026-04-26
**Plan:** `docs/FPP_experiment_plan_v0_2.md`
**Source commits:** Phase 1 = `c7ff116`; Phase 1.1 sub-experiments = `d6a6bc9` (A), `954fb1e` (B), `a44c020` (C), `ced5a1e` (D)

---

## TL;DR

Phase 1's headline finding — *"trained GPT-2 small under mode-A iteration
collapses every input to one trivial universal attractor"* — was true on
Phase 1's input zoo but **not universal in the architecture**. The Phase 1.1
diagnostic suite reveals a richer picture:

1. **Multiple basins exist** (Experiment B). With nested-structure inputs,
   GPT-2 small has at least two structurally distinct strong attractors
   (cos similarity 0.49 between them) plus several intermediate basins.
   Phase 1's 60 natural-language inputs all happened to lie in one basin.
2. **Basin selection is by lexical register**, not nesting depth (B). The
   two strong basins project to different LM-head signatures: lowercase
   mid-sentence connectors (` the / , / \n / and / .`) vs capitalised
   sentence-starters (`The / In / For / I / A`).
3. **Position embedding is not the homogeniser** (Experiment C). Cancelling
   repeated `wpe` injection leaves the trained fixed point at cos 0.9985
   to the mode-A baseline. Whatever causes the contraction comes from
   elsewhere — most plausibly LayerNorm at every block input.
4. **Semantics live partly in the transient, partly at the fixed point**
   (Experiment A). A 6-way linear probe on mean-pooled hidden states
   achieves 0.82 at step 0, decays monotonically to 0.40 at step 10
   (≈2.4× chance), while a random-init twin sits flat at ~0.35. Information
   *is* attenuated during contraction but not erased.
5. **Discrete token interface gives chaos, not richer attractors**
   (Experiment D). 0/60 token-fixed-points; 4 limit cycles; 48 distinct
   step-50 endpoints. Mode C is dominated by the same unigram-prior
   sink as mode A, but argmax discreteness prevents it from settling.

The original §4 hypothesis ("training enriches attractor structure") is
neither cleanly confirmed nor cleanly refuted. The truth is more nuanced:
training **does** produce a structured attractor manifold, but the manifold
is much smaller than a "rich landscape" and the convergent attractors
correspond to vocabulary register, not to per-input semantic interpretation.

The **single most natural Phase 1.2 candidate** is now the **LayerNorm
Lipschitz / Jacobian diagnostic** the plan §3.6 deferred. It is the
remaining cheap probe that has not been done and would pinpoint the
contraction source.

---

## How the four alternative explanations fared

The plan §1.1 listed four pre-committed alternative readings of Phase 1.
Phase 1.1 evidence on each:

### Explanation 1 — "FPP hypothesis is just wrong on GPT-2 mode A"

> Prediction: under all interface variants, trained GPT-2 collapses to one trivial attractor.

**Verdict: weakened.** Experiment B refutes the "one trivial attractor"
part — there are at least two structurally distinct strong basins, and
the input determines which one is reached. Experiment A's transient
probe also shows non-trivial input information at every step. So Phase 1's
"FPP totally fails on GPT-2" framing is too strong. There **is** structure;
it's just not the per-input semantic structure §4 originally predicted.

### Explanation 2 — "Interface mismatch (last_hidden_state is OOD)"

> Prediction: mode C (token interface) restores attractor diversity.

**Verdict: refuted.** Experiment D shows mode C produces *trajectory*
diversity (48/60 distinct sequences at step 50) but no attractor structure
(0/60 fixed points, 4/60 cycles). The unigram-prior pull is interface-
independent — both modes converge toward the same vocabulary-frequency
sink, just expressed differently (smooth in A, chaotic in C).

### Explanation 3 — "Engineering artefact: position embedding + LayerNorm"

> Prediction: removing wpe injection breaks the universal-attractor.

**Verdict: split.** The position-embedding half is **refuted** by
Experiment C — both C1 and C2 ablation produce a trained fixed point
at cos 0.9985 to the mode-A baseline. Position embedding is not the
contraction source. The LayerNorm half is **untested but elevated**: by
elimination, LayerNorm is the most likely remaining engineering source
of the contraction.

### Explanation 4 — "Semantics live in transient, not at the fixed point"

> Prediction: linear probe on hidden states reads category in transient, not at convergence.

**Verdict: partially supported with a non-trivial residue.** Experiment A
shows the predicted monotonic decay (trained probe 0.82 → 0.40 over 10
steps) — but the step-10 accuracy stays well above chance and above the
random-init baseline. The fixed point is *not informationless*; it is
*less informative than the transient*. The hypothesis as stated is
correct in direction, incomplete in magnitude.

### Summary table

| explanation | Phase 1.1 evidence | Phase 1.1 verdict |
| --- | --- | --- |
| 1 — FPP wrong on GPT-2 | Multiple basins exist (B); residual probe info (A) | weakened |
| 2 — Interface mismatch | Mode C produces chaos, no fixed points (D) | refuted |
| 3a — Position embedding | wpe ablation leaves fixed point at cos 0.998 (C) | refuted |
| 3b — LayerNorm contraction | Untested directly; only candidate left after 3a | elevated |
| 4 — Semantics in transient | Decay 0.82 → 0.40 (A); residual at fixed point | partially supported |

**The single new live track**: LayerNorm as the contraction driver.

---

## Cross-experiment synthesis

A few observations that emerge only when the four sub-reports are read
together:

### The "universal attractor" was Phase 1 zoo bias

Experiment B used 30 nested-structure inputs at seq_len = 64; nine of them
land at a fixed point with cosine 0.49 to Phase 1's universal attractor —
i.e., a clearly distinct point. Phase 1's 60 inputs were all natural-text
shaped at seq_len = 32; they all happened to belong to one basin.

This is methodologically important. Phase 1's report wrote: *"Trained
GPT-2 small dramatically changes the dynamics, but in the opposite
direction from §4 — making the system more contractive toward a single
trivial attractor."* The "single" in that sentence was wrong, conditional
on the input distribution. The corrected reading is: *"The trained model
has multiple stable basins; Phase 1's input zoo sampled only one."*

### The basins survive position-embedding ablation

Experiment C ran on Phase 1's natural-text 60-input zoo, not Experiment B's
nested zoo. So we don't directly know whether the Capital basin survives
wpe ablation. But the natural-text basin (cos 0.998 to mode-A) clearly
does. **Hypothesis (Phase 1.2 candidate)**: re-running Experiment B's nested
zoo through C1 will leave the Capital and lowercase basins distinct. If
that is right, the basin structure is intrinsic to the trained blocks,
and position embedding is not even the basin selector. Cost: ~30 s. Worth
doing as a small Phase 1.2 follow-up.

### The transient-probe residue and the multi-basin finding may be the same fact

Experiment A reports the trained model's step-10 fixed-point probe
accuracy at 0.40 — well above chance. Experiment B says different inputs
go to different basins. **If a linear probe at step 10 is partly reading
basin assignment**, the 0.40 residue may be the basin label, not per-input
semantic detail. A clean test: train a 2-way probe (Capital basin vs
lowercase basin) on Experiment B's 30 inputs at step 10. If accuracy is
high (~1.0), the residue is basin label. If low, there is sub-basin
information too. This is a straightforward Phase 1.2 candidate.

### Mode A and mode C share the unigram-prior sink

Mode A's universal attractor projects to ` the / , / \n / and / .`. Mode C
trajectories visibly converge into rearrangements of these same tokens
within ~5 steps and then chaotically reshuffle. The two modes are not
unrelated systems — they share a common attracting region in vocabulary
space. The difference is that mode A's continuous space lets the system
settle into a smooth fixed point while mode C's discrete argmax produces
chaotic motion within the same region.

This is a small but important conceptual unification: when reading either
mode's results, the underlying force is "the model's vocabulary-frequency
prior pulls every iteration toward high-prior tokens".

---

## Specific essay-revision recommendations

Per plan §10. The **structural** revisions Phase 1.1 enables, plus the
ChatGPT-suggested wording softenings the plan pre-committed.

### Add a new short subsection at the end of §7 — "Update from initial probes"

Suggested wording (about 250 words):

> *Initial empirical probes (FPP project, 2026-04).* Two rounds of
> diagnostic experiments were run on a frozen pretrained GPT-2 small,
> implementing the §7 Experiment One in the form of mode-A hidden-state
> self-iteration via the `inputs_embeds` path.
>
> The first round (Phase 1, 60 inputs across 6 categories of natural-language
> shape) showed that the trained model converges, in ~10 iterations, to
> a near-rank-1 fixed point that projects to the model's unigram prior
> ( the / , / \\n / and / .) at every position. All 60 inputs reached the
> same point. A randomly-initialised twin did not converge in 100 steps.
>
> The second round (Phase 1.1) ran four cheap diagnostics: (A) a transient
> linear probe, (B) a nested-structure input zoo, (C) position-embedding
> ablation, (D) a discrete argmax token interface. Findings: at least two
> structurally distinct strong attractors exist (basin chosen by surface
> lexical register, not nesting depth); position embedding is not the
> contraction source; semantic input information attenuates over the
> transient but does not fully vanish; discrete-token iteration produces
> chaotic trajectories with the same unigram-prior pull as mode A.
>
> The §4 hypothesis that training enriches attractor structure is neither
> cleanly confirmed nor cleanly refuted by these results. Training **does**
> produce a small manifold of structurally distinct attractors, but the
> attractors correspond to vocabulary register, not to per-input semantic
> interpretation. The most likely remaining contraction driver is the
> per-block LayerNorm; a Lipschitz / Jacobian diagnostic is the natural
> next step. Full data, code, and reports at
> `github.com/langram/self_context_essay/tree/main/fpp`.

### Wording softenings in §4 (independent of Phase 1.1, pre-committed in plan §10)

| original | revised |
| --- | --- |
| "context-attention 耦合不动点与场论是真同构 / structurally identical to field theory" | "structurally analogous to field theory: the same self-consistency equation form" |
| "Goedel self-reference is the necessary condition for systems to transcend themselves" | "self-reference is an important mechanism by which formal systems generate undecidability, reflexivity, and meta-level extension" |
| "AI cannot produce truly new integrative perspectives" | "current models lack a cross-conversation, cross-instance, persistently accumulating mechanism for self-integration" |
| §10 "the article itself is the existence proof" | "the article itself is a conceptual example / a generation-process analogue of the proposed mechanism" |

### What Phase 1.1 does **not** support changing in §4

- The argument that self-reference matters for emergence — untouched.
- The (context, attention) coupled-system formulation — untouched.
- The claim that current LLMs lack cross-conversation persistence — untouched.
- The §10 recursive footnote framing — softened in wording, not retracted.

### What Phase 1 + 1.1 do **not** support but the data suggests

The data shows trained GPT-2 has multi-basin structure, with basins
selected by surface lexical register. This is a *fact about the trained
model*, not a vindication of the §4 prediction. The §4 prediction was
about per-input semantic attractors; the data shows per-register attractors.
That's a different (and weaker) form of the claim. The essay revision should
acknowledge this difference rather than fold the multi-basin finding into
"§4 was right after all".

---

## Next-step recommendations (for human + AI co-decision)

The plan §11 (meta-discipline) explicitly warned not to inflate any single
finding into "FPP is back" or "must build FPP-native". Following that
discipline, Phase 1.2 candidates ranked by cost / discriminating-power ratio:

1. **LayerNorm Lipschitz / Jacobian diagnostic** (plan §3.6, deferred).
   Now uniquely indicated by Experiment C. Compute the Jacobian of the
   block-only forward map at the fixed point; report Lipschitz constant
   and dominant eigenvalues. Cost: 1–2 days. **Strongly recommended.**

2. **Cross-basin transient probe** (extension of A using B's data). Re-run
   Experiment A on Experiment B's nested zoo, with basin label as the
   probe target. Tests whether the step-10 residue is basin label or
   sub-basin variation. Cost: 0.5 days.

3. **Position-ablation + nested zoo** (B × C joint follow-up). Re-run
   Experiment B's nested zoo through C1 cancel-pos. Tests whether the
   Capital basin is preserved without wpe. Cost: 30 s GPU + half day analysis.

4. **Mode B (temperature sampling)** (plan §3.6, deferred). Same pipeline
   as mode C but with temperature ≥ 0 sampling instead of argmax. Tests
   whether mode C's 0/60 fixed-points is an argmax-strictness artefact
   or genuine non-contraction. Cost: 1 day.

5. **Scale ladder** (plan §1.2 explicitly excluded from Phase 1.1, but
   defensibly in scope for Phase 1.2). Repeat all 4 sub-experiments on
   GPT-2 medium / large / xl / Pythia-1.4B. Tests whether the multi-basin
   structure scales with model size. Cost: 2–3 days.

**My recommended Phase 1.2 minimum**: items 1 + 2 (LayerNorm diagnostic +
cross-basin probe). Together they tell us (a) what is doing the contraction
and (b) what the residual probe accuracy is reading. Items 3–5 are good
follow-ups but not on the critical path.

**My recommended Phase 1.2 maximum**: items 1 + 2 + 3. Item 3 is so cheap
it should just go in.

---

## What to send to the next AI round

A new feedback package (analogous to `fpp_phase1_package_2026-04-26.zip`)
should be assembled for the next AI consultation, containing:

- This master report
- The four sub-reports (`phase1_1_{transient_probe, html_outlier, position_ablation, mode_c}.md`)
- Key figures from each experiment
- The Phase 1.1 plan v0.2 (already in `docs/`)
- A handful of representative raw traces from each experiment (1 per category, trained variant)
- The processed JSONs (records, summaries) — small, full coverage
- Source code snapshot
- An updated README with the question set:
  - Is the LayerNorm-as-contraction-source reading well-supported, or are there
    other engineering candidates we missed?
  - Is the multi-basin / register-selection reading a real architectural property
    or a Phase 1.1 input-zoo artefact?
  - How should §4 be revised to reflect "small attractor manifold by register,
    not per-input semantics"?
  - Of the 5 Phase 1.2 candidates above, which would each AI prioritise?

Plan §4.3 calls for this package. Recommend assembling it next, using the
same layout as the Phase 1 package.

---

## Caveats applying to the whole Phase 1.1

- **Sample size.** All experiments use ≤ 60 inputs (≤ 30 in B, ≤ 12 per
  fold in A). The signals reported are clean enough to read but the
  *quantitative* claims (effective rank 1.002, accuracy 0.400, etc.) have
  ±10–20% relative uncertainty. Treat directions and orderings as
  reliable; treat exact numbers as approximate.
- **Single model.** Everything was on GPT-2 small (124M) for cost reasons.
  A scale-ladder check (Phase 1.2 item 5) is essential before generalising
  any architectural claim.
- **Mode A only for §4-aligned testing.** The §4 hypothesis is about the
  coupled (context, attention) dynamical system. Mode A is one
  instantiation. Modes B and C exist but have different fixed-point
  equations; their results inform the question but don't directly test §4.
- **Phase 1.1 was a diagnostic, not an attempt to revive FPP.** This was
  explicit in plan §11 and held throughout. The "LayerNorm is the next
  candidate" reading is offered as the most data-supported next probe,
  not as a re-formulation of the §4 hypothesis.

---

## Pointers

- Plan: `docs/FPP_experiment_plan_v0_2.md`
- Phase 1 baseline: `outputs/reports/phase1_gpt2_small/report.md`
- Sub-reports:
  - `outputs/reports/phase1_1_transient_probe/phase1_1_transient_probe.md`
  - `outputs/reports/phase1_1_html_outlier/phase1_1_html_outlier.md`
  - `outputs/reports/phase1_1_position_ablation/phase1_1_position_ablation.md`
  - `outputs/reports/phase1_1_mode_c/phase1_1_mode_c.md`
- Per-experiment raw / processed data: `data/{raw,processed}/phase1_1_*/`
- Source: `src/{transient_probe, extended_inputs, run_html_outlier, posfree_iterate, run_position_ablation, mode_c_iterate, run_mode_c}.py`
- Total disk usage: ~1.8 GB (raw traces); processed + figures + reports < 10 MB.
- Total wall-clock: ~3 hours (mostly analysis + report-writing; GPU time was ~3 minutes).
