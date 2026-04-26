# FPP run report — `phase0_smoke`

- timestamp: 20260426_172704
- git commit: `c7df348`
- model: `gpt2` (dtype=float32, device=cuda)
- max_iter: 100, convergence_threshold: 0.001
- inputs: 1 categories × 1

## Aggregate convergence

| variant | n_inputs | %converged | %diverged | mean steps (converged) |
| --- | --- | --- | --- | --- |
| trained | 1 | 100.0% | 0.0% | 10.0 |

## Pointers

- raw per-trace tensors: `data/raw/<run_id>/<variant>/trace_*.pt`
- per-trace metrics JSON: `data/processed/<run_id>/records_<variant>.json`
- figures: `outputs/figures/<run_id>/`
- top-5 vocab projection: `outputs/reports/<run_id>/top5_<variant>.md`

## Open questions

- Q1 existence — see convergence histograms.
- Q2 diversity — see C* similarity heatmap and dendrogram.
- Q3 trained-vs-random — compare both variants in the histograms / similarity grids.
- Q4 semantic correspondence — see top-5 vocab projection.

Fill in the §5.3 narrative answers after inspecting the figures.