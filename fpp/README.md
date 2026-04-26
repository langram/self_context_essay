# FixedPointProbe (FPP)

Concept-level experiment for `Self Context Is All AI Need?` §7 Experiment One.

**Question.** When a pretrained Transformer is iterated in hidden-state space — feeding `last_hidden_state` back through itself via `inputs_embeds` — does it exhibit non-trivial fixed-point dynamics? If so, do trained weights produce richer attractor structure than randomly initialised weights of the same architecture?

This is an inference-only probe. No training, no new architecture, no new loss. It treats a frozen Transformer as a discrete dynamical system `C_{n+1} = transformer_θ(C_n)` and observes what happens.

## Layout

```
fpp/
├── configs/                  experiment configs (JSON)
├── src/
│   ├── inputs.py             60-input zoo (6 categories × 10), 32-token padded
│   ├── iterate.py            core hidden-state iteration loop
│   ├── metrics.py            convergence, effective rank, vocab projection, similarity
│   ├── visualize.py          histograms, traces, similarity heatmap, dendrogram
│   └── run_experiment.py     orchestrator
├── data/
│   ├── raw/                  full hidden-state traces (.pt) — gitignored
│   └── processed/            metric arrays (.npz)
├── outputs/
│   ├── figures/              generated plots
│   └── reports/              markdown reports
└── tests/
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate    # Windows
pip install -r requirements.txt
```

## Running

```bash
# Phase 0 smoke test (1 input, 1 model)
python -m src.run_experiment --config configs/phase0_smoke.json

# Phase 1 full run (60 inputs × trained + random init)
python -m src.run_experiment --config configs/phase1_gpt2_small.json
```

## Acceptance gates

See `../docs/FPP_experiment_plan_v0_1.md` §6.

- **Phase 0** — single trace runs end-to-end in < 5 s on the target GPU.
- **Phase 1** — all 120 traces complete; all §3.5 metrics computed; all §5.2 figures rendered.
- **Phase 2** — markdown report answers Q1–Q4 with data, including negative results.

## Hardware target

RTX 4070 Ti 12GB, Ryzen 7 7800X3D, 64GB RAM, Windows 11. GPT-2 small fits in fp32 with room to spare.
