# FPP run report — `phase1_gpt2_small`

- model: `gpt2` (124M, fp32, CUDA on RTX 4070 Ti)
- inputs: 6 categories × 10 = 60, all padded/truncated to seq_len = 32
- iteration mode: A — feed `last_hidden_state` back via `inputs_embeds`
- max_iter = 100, convergence_threshold = 1e-3 (relative Frobenius delta)
- variants: trained checkpoint vs random-init twin (same architecture, same seed)
- elapsed: ~71 s wall-clock for the full run incl. robustness checks

This run completes the Phase 1 gate (§6) — all 120 traces ran, all §3.5 metrics
computed, all §5.2 figures rendered, both robustness checks ran. What follows is
the §5.3 narrative.

---

## TL;DR

Mode-A iteration on a frozen pretrained GPT-2 small **does** have a fixed point.
The fixed point is the *same* for every one of the 60 inputs (cosine similarity
between any two final states ≥ 0.88, mean 0.996). Effective rank ≈ 1.02 — the
state has collapsed to a near-rank-1 matrix in which all 32 token positions
share essentially the same hidden vector. Projecting that universal C\* through
the LM head yields ` the` / `,` / `\n` / ` and` / `.` at every position — i.e.
the GPT-2 unigram prior.

The randomly-initialised twin **does not** converge in 100 steps under the same
threshold; its iterates contract slowly (delta drops from ~0.19 to ~0.03 but
never below 1e-3) and the final states have effective rank ≈ 7 with much
weaker (~0.72) but unstructured cross-input similarity.

The headline reading: training does change the dynamics dramatically, but it
makes the system *more contractive* toward a single global attractor, not
*more diverse* in attractor structure. Mode-A iteration on this checkpoint
computes something close to the model's mean-field vocabulary prior, not a
per-input "stable interpretation". This is a clean negative result for the
naive form of the §4 hypothesis ("training enriches attractor structure"),
and a clean positive result for the §7.6 caveat ("LayerNorm + position
embedding can drive iteration to trivial fixed points").

---

## Q1 — does iteration converge?

**Trained.** 59/60 inputs converge below `delta < 1e-3` within 100 steps.
Mean steps to convergence: 10.0 (essentially identical across all categories).
The single non-converging input is in `structured` (a multi-line code snippet)
and it would converge given more iterations — its long-trace check confirms
this trend.

**Random-init.** 0/60 converge in 100 steps. All deltas are still in the
`[2e-2, 5e-2]` band at step 100 — contractive but slow. No divergence (no
norm explosion or collapse). LayerNorm pins `||h||` at ~157 throughout.

**Long-trace robustness check** (5 inputs, max_iter = 1000) on the trained
model confirms the 10-step convergence is genuine: those 5 traces still stop
at exactly 10 steps when the threshold is reached, and never restart — there
is no "looks converged then drifts" failure mode.

## Q2 — does iteration reach diverse attractors, or one trivial attractor?

**Trained model: one universal attractor.**

- Mean off-diagonal cosine similarity between the 60 C\* states: **0.996**
  (median 1.000, min 0.880, max 1.000).
- Intra-category mean = inter-category mean = 0.996. There is no category
  structure visible in the similarity matrix.
- Effective rank of C\* per input: **≈ 1.02** for every category except
  `structured` (≈ 1.11). The 32-token state has collapsed to essentially a
  single direction repeated at every position.
- `||C*||` ≈ 1812 across all inputs — uniform.
- Perturbation robustness (5 inputs × 10 ε-perturbed initialisations,
  ε = 0.01·||h₀||): off-diagonal cosine of the 10 perturbed C\*'s = 1.0000
  for every probe. The basin of attraction is extremely deep — small noise
  in `h₀` is fully absorbed.

**Random-init model: weakly structured, no clear attractors.**

- Mean off-diagonal cosine 0.72, range [0.23, 1.00]. There is variation but
  inter-category mean (0.718) is essentially equal to intra-category mean
  (0.742) — no semantic clustering.
- Effective rank ≈ 7 (much higher than the trained model), suggesting the
  random transformer preserves more dimensions during contraction.
- These states are not actually fixed points (Q1 said 0/60 converged); they
  are just the *step-100 snapshot* of slowly-contracting trajectories.

## Q3 — how does training change the dynamics?

Training makes the iteration:

1. **Much more contractive** — ~10 steps to a true fixed point vs not
   converging in 100.
2. **Far more concentrated** — effective rank ~1 vs ~7; cosine similarity
   between any two C\*'s is ~1 vs ~0.72.
3. **Insensitive to input** — every category collapses to the same point.

Read together, training appears to deepen and narrow a single global basin,
not to carve out a richer attractor landscape. This is the opposite of what
§4 of the essay tentatively predicted ("trained model has richer attractor
structure than random init"). The actual signal is closer to: training
imposes a strong global prior that dominates mode-A iteration.

## Q4 — what is the universal attractor, semantically?

Projecting C\* through the final LayerNorm + LM head yields, at *every* of
the 32 token positions, the same top-5: **` the`, `,`, `\n`, ` and`, `.`**.
This is the unigram prior — the most-frequent tokens GPT-2 sees. It is the
same regardless of whether the input was random tokens, ambiguous English,
nonsense-grammatical sentences, or Python code.

Reading: mode-A iteration in this configuration is computing the model's
position-uniform mean prediction over the vocabulary, not a per-input stable
interpretation. The position embedding (§2.2) added at every iteration step
acts as a constant forcing toward the position-conditioned mean, and the
final LayerNorm + lm_head's geometry surfaces the highest-prior tokens
once the input-dependent variation has been damped out.

The random-init projection by contrast gives idiosyncratic clusters of
low-frequency tokens (` scholarships`, `Inv`, `netflix`, `asaki`, …) — the
result of the random embedding/lm-head matrix multiplied with the contracted
state. There is no unigram prior to leak through because there is no trained
prior.

---

## What's stable, what's worth re-checking, what's likely an artefact

- **Stable.** The 10-step convergence on the trained model. The
  rank-1 collapse. The unigram-prior projection. All robust to ε ≤ 1%
  initial-state perturbation. All robust to max_iter ≤ 1000.
- **Worth re-checking on bigger models.** The §3.1 ladder (medium / large /
  xl / Pythia 1.4B) — does the universal attractor persist? Does effective
  rank stay near 1? If yes across the family, this is a structural property
  of decoder-only Transformers under mode-A iteration. If no, GPT-2 small
  is anomalous.
- **Likely artefact.** The exact ` the` / `,` / `\n` / ` and` / `.` ordering
  is an artefact of GPT-2's training corpus (heavy WebText). The
  *phenomenon* — collapse to the unigram prior — should generalise; the
  *specific tokens* should differ across model families.
- **Methodological caveat (§7.6 confirmed).** LayerNorm at every block
  input clamps `||h||` and probably contributes most of the contraction.
  Mode-A is therefore not a clean test of "context-attention coupling
  dynamics" — it's a test of *that coupling under a strong norm-control
  feedback*. Mode B (sample-then-feed-back) and mode C (argmax-then-feed-back)
  break the LayerNorm chain and may give qualitatively different
  attractor structure. A future run should add at least one of those for
  comparison.
- **Position embedding caveat (§2.2 confirmed).** The constant
  position-embedding forcing at every iteration step is a non-trivial
  external input. A true "free" iteration would need to subtract it or
  use a position-free architecture. This is a clean follow-up.

---

## Concrete next steps

1. **Mode B / mode C iteration.** Cheapest variant test of whether the
   collapse is intrinsic or LayerNorm-driven.
2. **Mid-layer iteration.** §4.3 in the plan. Iterate at layer 6 instead
   of the last layer; see if effective rank and attractor diversity recover.
3. **Position-embedding ablation.** Subtract the position embedding before
   feeding back, or use Pythia (rotary) which doesn't have this exact
   structure.
4. **Scale ladder.** Repeat the run on `gpt2-medium`, `gpt2-large`,
   `gpt2-xl`, `EleutherAI/pythia-1.4b`. Watch effective rank and attractor
   diversity as functions of scale.

These are ordered by cost (cheapest first) and by how directly they
discriminate between competing readings of the Phase 1 result.

---

## Pointers

- raw per-trace tensors: `data/raw/phase1_gpt2_small/{trained,random}/trace_*.pt`
- per-trace metrics JSON: `data/processed/phase1_gpt2_small/records_{trained,random}.json`
- similarity matrices: `data/processed/phase1_gpt2_small/similarity_{trained,random}.npz`
- robustness JSON: `data/processed/phase1_gpt2_small/robustness_{perturbation,long_trace}.json`
- figures: `outputs/figures/phase1_gpt2_small/`
- top-5 vocab projection tables: `outputs/reports/phase1_gpt2_small/top5_{trained,random}.md`
- aggregate summary: `data/processed/phase1_gpt2_small/summary.json`
