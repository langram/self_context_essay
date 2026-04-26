"""Phase 1.1 Experiment A — transient linear probe per FPP plan v0.2 §3.2.

Loads the full 120 hidden-state traces saved by Phase 1, mean-pools each h_n
over the seq_len dimension to a 768-vector, and trains a 6-way logistic
regression with 5-fold CV on a chosen probe step. Repeats per step, plots
accuracy vs step for trained vs random init, and writes a markdown sub-report.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]


PROBE_STEPS = (0, 1, 2, 3, 5, 7, 10)
N_FOLDS = 5
N_INPUTS = 60
N_CLASSES = 6
CHANCE = 1.0 / N_CLASSES


@dataclass
class ProbeResult:
    step: int
    mean_acc: float
    std_acc: float
    fold_accs: list[float]


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def load_traces(raw_dir: Path) -> tuple[list[torch.Tensor], list[str], list[int]]:
    """Returns (traces, categories, n_steps) sorted by trace index."""
    files = sorted(raw_dir.glob("trace_*.pt"))
    traces: list[torch.Tensor] = []
    categories: list[str] = []
    n_steps_list: list[int] = []
    for f in files:
        d = torch.load(f, weights_only=False)
        traces.append(d["trace"])  # [n_saved, seq_len, hidden_dim]
        categories.append(d["category"])
        n_steps_list.append(len(d["deltas"]))
    return traces, categories, n_steps_list


def hidden_at_step(trace: torch.Tensor, step: int) -> torch.Tensor:
    """Return the hidden state at iteration `step`. Clamp to last available state if past convergence.

    Trace storage convention: index 0 is h_0; index k is h_k. For a trace that converged at step c,
    saved length is c+1 (h_0..h_c). Past that, the system is at the fixed point so we return h_c.
    """
    last = trace.shape[0] - 1
    return trace[min(step, last)]


def pool_mean(h: torch.Tensor) -> np.ndarray:
    """Mean over seq_len → [hidden_dim] np array."""
    return h.mean(dim=0).numpy()


def build_features(traces: list[torch.Tensor], step: int) -> np.ndarray:
    """[n_traces, hidden_dim] feature matrix for the given probe step."""
    return np.stack([pool_mean(hidden_at_step(t, step)) for t in traces], axis=0)


def cv_probe(X: np.ndarray, y: np.ndarray, n_folds: int = N_FOLDS, seed: int = 42) -> ProbeResult:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs: list[float] = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=seed)
        clf.fit(X_train, y[train_idx])
        fold_accs.append(float(clf.score(X_test, y[test_idx])))
    return ProbeResult(step=-1, mean_acc=float(np.mean(fold_accs)),
                       std_acc=float(np.std(fold_accs)), fold_accs=fold_accs)


def run_probe_set(
    traces: list[torch.Tensor],
    categories: list[str],
    label: str,
    seed: int,
) -> dict[int, ProbeResult]:
    cat_to_idx = {c: i for i, c in enumerate(sorted(set(categories)))}
    y = np.asarray([cat_to_idx[c] for c in categories])
    out: dict[int, ProbeResult] = {}
    for step in PROBE_STEPS:
        X = build_features(traces, step)
        res = cv_probe(X, y, seed=seed)
        res.step = step
        out[step] = res
        print(f"  [{label}] step {step:>3d}  acc = {res.mean_acc:.3f} ± {res.std_acc:.3f}")
    return out


def plot_accuracy_curve(
    results: dict[str, dict[int, ProbeResult]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"trained": "tab:blue", "random": "tab:orange"}
    for label, by_step in results.items():
        steps = sorted(by_step.keys())
        means = [by_step[s].mean_acc for s in steps]
        stds = [by_step[s].std_acc for s in steps]
        ax.errorbar(steps, means, yerr=stds, marker="o", label=label,
                    color=colors.get(label, None), capsize=3, linewidth=1.6)
    ax.axhline(CHANCE, color="grey", linestyle="--", linewidth=0.8, label=f"chance = {CHANCE:.3f}")
    ax.set_xlabel("iteration step")
    ax.set_ylabel("6-way classification accuracy (5-fold CV)")
    ax.set_title("Transient linear probe — can a linear classifier read input category from h_n?")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_subreport(
    results: dict[str, dict[int, ProbeResult]],
    out_path: Path,
    cfg: dict,
    figure_path: Path,
    git_commit: str,
    timestamp: str,
) -> None:
    lines = [
        "# Phase 1.1 — Experiment A: Transient Linear Probe",
        "",
        f"- timestamp: {timestamp}",
        f"- git commit: `{git_commit}`",
        f"- source traces: `data/raw/phase1_gpt2_small/{{trained,random}}/trace_*.pt`",
        f"- probe steps: {list(PROBE_STEPS)}",
        f"- pool: mean over seq_len → 768-dim vector",
        f"- model: sklearn LogisticRegression (lbfgs, C=1.0, max_iter=2000) on standard-scaled features",
        f"- CV: {N_FOLDS}-fold stratified, seed = {cfg['seed']}",
        f"- chance = {CHANCE:.3f} (6 categories × 10 inputs each)",
        "",
        "## Accuracy table",
        "",
        "| step | trained mean ± std | random mean ± std |",
        "| ---: | --- | --- |",
    ]
    for s in PROBE_STEPS:
        t = results["trained"][s]
        r = results["random"][s]
        lines.append(f"| {s} | {t.mean_acc:.3f} ± {t.std_acc:.3f} | {r.mean_acc:.3f} ± {r.std_acc:.3f} |")

    trained_curve = [results["trained"][s].mean_acc for s in PROBE_STEPS]
    random_curve = [results["random"][s].mean_acc for s in PROBE_STEPS]

    early = results["trained"][0].mean_acc
    late = results["trained"][10].mean_acc
    drop = early - late
    rand_early = results["random"][0].mean_acc
    rand_late = results["random"][10].mean_acc

    if early > 0.5 and late < 0.3:
        verdict = (
            "**Strong support for Explanation 4 (semantics live in transient).** "
            "Trained-model accuracy is well above chance early and decays toward chance "
            "by the time the iteration converges to the universal attractor."
        )
    elif early > 0.5 and late > 0.5:
        verdict = (
            "**Mixed.** Probe accuracy stays high even at the converged state. "
            "This may indicate the universal attractor still leaks input-dependent "
            "information through residual variation, or the probe is reading a "
            "non-semantic signal (positional, length, token-distribution). "
            "Needs an ablation on the input zoo to disambiguate."
        )
    elif early < 0.3:
        verdict = (
            "**Probe-design failure.** Step-0 accuracy is at or near chance — the probe "
            "cannot read category even from the initial hidden state. Mean-pool over "
            "seq_len is likely too lossy. Re-run with last-token or attention-weighted "
            "pooling before drawing any conclusion about transient information."
        )
    else:
        verdict = (
            "**Marginal.** Probe accuracy is above chance but not by a wide margin. "
            "60 inputs is a small sample; the per-fold std is non-negligible. "
            "Cannot conclude either direction without a larger input zoo."
        )

    lines += [
        "",
        "## Reading",
        "",
        verdict,
        "",
        f"- trained step-0 acc {early:.3f} → step-10 acc {late:.3f}  (drop {drop:+.3f})",
        f"- random  step-0 acc {rand_early:.3f} → step-10 acc {rand_late:.3f}",
        "",
        "## Caveats",
        "",
        "- N = 60 trains a 6-way classifier with 12 examples per class per fold; per-fold variance is large.",
        "- Mean-pooling over seq_len destroys positional structure; if the probe sees nothing it does not "
          "imply the hidden states have lost semantics — only that the *mean* has.",
        "- Standard-scaling is fit per fold but the feature space is 768-dim with 48 train examples; the "
          "scaler is itself noisy. Results below ~0.4 should be read as 'probe couldn't find a signal', "
          "not as 'no signal exists'.",
        "- The trained 'step 10' state for converged traces *is* the fixed point C\\* (per the storage "
          "convention — past convergence, hidden_at_step clamps to the last saved state).",
        "",
        "## Pointers",
        "",
        f"- figure: `{figure_path.relative_to(PROJECT_ROOT)}`",
        f"- raw fold accuracies: `{out_path.with_name('transient_probe_results.json').relative_to(PROJECT_ROOT)}`",
        "",
        "## Next-action implications (per plan §3.2)",
        "",
        ("- If verdict is 'Strong support for Explanation 4': the §4 fixed-point framing for GPT-2 is "
         "the wrong observation level; transient computation theory becomes the live track."),
        ("- If verdict is 'Probe-design failure': try last-token pooling, then attention-weighted pooling, "
         "before concluding anything about transient semantics."),
        ("- If verdict is 'Mixed' or 'Marginal': proceed to Experiments B/C/D as planned; the transient "
         "track is neither confirmed nor ruled out and needs the other diagnostics for context."),
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-run-id",
        default="phase1_gpt2_small",
        help="Phase 1 run-id whose traces are probed",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_root = PROJECT_ROOT / "data" / "raw" / args.source_run_id
    out_run_id = "phase1_1_transient_probe"
    fig_dir = PROJECT_ROOT / "outputs" / "figures" / out_run_id
    report_dir = PROJECT_ROOT / "outputs" / "reports" / out_run_id
    processed_dir = PROJECT_ROOT / "data" / "processed" / out_run_id
    for d in (fig_dir, report_dir, processed_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[probe] loading traces from {raw_root}")
    trained_traces, trained_cats, trained_steps = load_traces(raw_root / "trained")
    random_traces, random_cats, random_steps = load_traces(raw_root / "random")
    assert len(trained_traces) == N_INPUTS, f"expected {N_INPUTS} trained traces, got {len(trained_traces)}"
    assert len(random_traces) == N_INPUTS, f"expected {N_INPUTS} random traces, got {len(random_traces)}"

    print(f"[probe] trained n_steps distribution: min={min(trained_steps)} max={max(trained_steps)}")
    print(f"[probe] random  n_steps distribution: min={min(random_steps)} max={max(random_steps)}")

    print("[probe] training probes (trained)")
    trained_results = run_probe_set(trained_traces, trained_cats, "trained", args.seed)
    print("[probe] training probes (random)")
    random_results = run_probe_set(random_traces, random_cats, "random", args.seed)

    results = {"trained": trained_results, "random": random_results}

    # Persist raw fold accs
    payload = {
        "probe_steps": list(PROBE_STEPS),
        "n_folds": N_FOLDS,
        "n_inputs": N_INPUTS,
        "chance": CHANCE,
        "seed": args.seed,
        "source_run_id": args.source_run_id,
        "results": {
            label: {str(s): {"mean": r.mean_acc, "std": r.std_acc, "folds": r.fold_accs}
                    for s, r in by_step.items()}
            for label, by_step in results.items()
        },
    }
    (processed_dir / "transient_probe_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    fig_path = fig_dir / "transient_probe_accuracy.png"
    plot_accuracy_curve(results, fig_path)

    cfg = {"seed": args.seed, "source_run_id": args.source_run_id}
    write_subreport(
        results,
        report_dir / "phase1_1_transient_probe.md",
        cfg,
        fig_path,
        _git_commit(),
        dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    print(f"[probe] done. report: {report_dir / 'phase1_1_transient_probe.md'}")


if __name__ == "__main__":
    main()
