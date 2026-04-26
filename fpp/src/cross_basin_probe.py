"""Phase 1.2 Experiment E — cross-basin probe per FPP plan v0.3 §3.2.

Distinguishes whether the 0.40 step-10 probe accuracy from Phase 1.1 Experiment A
is reading basin label or sub-basin per-input information. Uses the saved
nested-zoo traces (Phase 1.1 Experiment B), no new compute.

Three probes, all on h_10 mean-pooled to 768-vec:
  Probe-basin: 3-way classifier {lowercase, capital, hybrid}
  Probe-fine-global: 6-way classifier across all 30 inputs (the 6 nested categories)
  Probe-fine-within-basin: 6-way classifier within each basin separately

Per §6.2 — leave-one-out CV due to small per-basin sample size.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "phase1_2_cross_basin_probe"

# Basin thresholds per FPP plan v0.3 §6.2 (refined to better match the plan §3.2 expected distribution)
LOWERCASE_COS_MIN = 0.999
CAPITAL_COS_MAX = 0.7


@dataclass
class ProbeOutcome:
    name: str
    n_classes: int
    n_samples: int
    cv: str
    accuracy: float
    accuracy_std: float
    chance: float
    fold_accs: list[float]


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def load_b_traces() -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[int]]:
    """Returns (h_10_pooled_per_input, h_final_pooled_per_input, category, idx) ordered by idx.

    h_10 is the FEATURE for the probe; h_final is used to assign the basin LABEL.
    """
    raw_root = PROJECT_ROOT / "data" / "raw" / "phase1_1_html_outlier"
    h10_list: list[np.ndarray] = []
    hfin_list: list[np.ndarray] = []
    cats: list[str] = []
    idxs: list[int] = []
    for i in range(30):
        d = torch.load(raw_root / f"trace_{i:03d}.pt", weights_only=False)
        trace = d["trace"]
        assert trace.shape[0] >= 2, f"trace {i} has only {trace.shape[0]} saved states"
        h10 = trace[1]  # h_10 (saved at step=9 since save_every=10)
        hfin = trace[-1]  # final saved state — basin assignment reference
        h10_list.append(h10.mean(dim=0).numpy())
        hfin_list.append(hfin.mean(dim=0).numpy())
        cats.append(d["category"])
        idxs.append(i)
    return h10_list, hfin_list, cats, idxs


def load_phase1_universal_ref() -> np.ndarray:
    """Mean-pooled C* of a known phase 1 lowercase-basin converger, for cosine reference."""
    raw_root = PROJECT_ROOT / "data" / "raw" / "phase1_gpt2_small" / "trained"
    # Use the same canonical reference as Phase 1.1 B did: any converged trained trace's final state.
    for i in range(60):
        d = torch.load(raw_root / f"trace_{i:03d}.pt", weights_only=False)
        if d["converged"]:
            ref = d["trace"][-1].numpy()
            return ref.mean(axis=0)
    raise RuntimeError("No converged Phase 1 trace found.")


def assign_basin(cos_to_ref: float) -> str:
    if cos_to_ref > LOWERCASE_COS_MIN:
        return "lowercase"
    if cos_to_ref < CAPITAL_COS_MAX:
        return "capital"
    return "hybrid"


def _normalise(v: np.ndarray) -> np.ndarray:
    return v / max(np.linalg.norm(v), 1e-12)


def cv_probe(
    X: np.ndarray,
    y: np.ndarray,
    name: str,
    chance: float,
    use_loo: bool,
    seed: int,
) -> ProbeOutcome:
    if use_loo or X.shape[0] < 25:
        cv = LeaveOneOut()
        cv_name = "LOO"
    else:
        cv = StratifiedKFold(n_splits=min(5, np.unique(y).size), shuffle=True, random_state=seed)
        cv_name = "5-fold-stratified"

    fold_accs: list[float] = []
    for train_idx, test_idx in cv.split(X, y):
        if np.unique(y[train_idx]).size < 2:
            # LOO can leave a fold with only one class in training when that class has 1 sample.
            # Skip — degenerate.
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=4000, C=1.0, solver="lbfgs", random_state=seed)
        clf.fit(X_tr, y[train_idx])
        fold_accs.append(float(clf.score(X_te, y[test_idx])))

    if not fold_accs:
        return ProbeOutcome(name=name, n_classes=int(np.unique(y).size), n_samples=int(X.shape[0]),
                            cv=cv_name, accuracy=float("nan"), accuracy_std=float("nan"),
                            chance=chance, fold_accs=[])
    return ProbeOutcome(
        name=name,
        n_classes=int(np.unique(y).size),
        n_samples=int(X.shape[0]),
        cv=cv_name,
        accuracy=float(np.mean(fold_accs)),
        accuracy_std=float(np.std(fold_accs)),
        chance=chance,
        fold_accs=fold_accs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fig_dir = PROJECT_ROOT / "outputs" / "figures" / RUN_ID
    report_dir = PROJECT_ROOT / "outputs" / "reports" / RUN_ID
    processed_dir = PROJECT_ROOT / "data" / "processed" / RUN_ID
    for d in (fig_dir, report_dir, processed_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[{RUN_ID}] loading Phase 1.1 B traces (h_10 = feature, h_final = basin label)")
    h10_list, hfin_list, cats, idxs = load_b_traces()
    h10 = np.stack(h10_list, axis=0)  # [30, 768] — features
    hfin = np.stack(hfin_list, axis=0)  # [30, 768] — for basin assignment only

    ref_vec = _normalise(load_phase1_universal_ref())
    hfin_unit = hfin / np.maximum(np.linalg.norm(hfin, axis=1, keepdims=True), 1e-12)
    cos_to_ref = hfin_unit @ ref_vec  # cosine of the FINAL state to Phase 1 universal
    basins = [assign_basin(c) for c in cos_to_ref]

    basin_dist = Counter(basins)
    print(f"[{RUN_ID}] basin distribution: {dict(basin_dist)}")
    print(f"[{RUN_ID}] thresholds: lowercase if cos > {LOWERCASE_COS_MIN}, capital if cos < {CAPITAL_COS_MAX}")

    # Probe-basin: 3-way (or fewer if some basins are empty)
    basin_to_id = {b: i for i, b in enumerate(sorted(basin_dist.keys()))}
    y_basin = np.asarray([basin_to_id[b] for b in basins])
    chance_basin = max(basin_dist.values()) / sum(basin_dist.values())  # majority-class chance
    probe_basin = cv_probe(h10, y_basin, name="probe-basin", chance=chance_basin, use_loo=True, seed=args.seed)

    # Probe-fine-global: 6-way
    fine_to_id = {c: i for i, c in enumerate(sorted(set(cats)))}
    y_fine = np.asarray([fine_to_id[c] for c in cats])
    fine_dist = Counter(cats)
    chance_fine_global = max(fine_dist.values()) / sum(fine_dist.values())
    probe_fine_global = cv_probe(h10, y_fine, name="probe-fine-global", chance=chance_fine_global,
                                 use_loo=True, seed=args.seed)

    # Probe-fine-within-basin: per-basin 6-way
    within_results: dict[str, ProbeOutcome] = {}
    for basin in sorted(basin_dist.keys()):
        mask = np.asarray([b == basin for b in basins])
        if mask.sum() < 3:
            within_results[basin] = ProbeOutcome(
                name=f"probe-fine-within-{basin}",
                n_classes=int(np.unique(y_fine[mask]).size),
                n_samples=int(mask.sum()),
                cv="too-small-skipped",
                accuracy=float("nan"), accuracy_std=float("nan"),
                chance=float("nan"), fold_accs=[]
            )
            continue
        X_b = h10[mask]
        y_b = y_fine[mask]
        cat_dist_b = Counter(y_b.tolist())
        if len(cat_dist_b) < 2:
            within_results[basin] = ProbeOutcome(
                name=f"probe-fine-within-{basin}", n_classes=1, n_samples=int(mask.sum()),
                cv="single-class-skipped", accuracy=float("nan"), accuracy_std=float("nan"),
                chance=float("nan"), fold_accs=[]
            )
            continue
        chance_b = max(cat_dist_b.values()) / sum(cat_dist_b.values())
        within_results[basin] = cv_probe(X_b, y_b, name=f"probe-fine-within-{basin}",
                                         chance=chance_b, use_loo=True, seed=args.seed)

    # Persist results
    payload = {
        "thresholds": {"lowercase_cos_min": LOWERCASE_COS_MIN, "capital_cos_max": CAPITAL_COS_MAX},
        "basin_distribution": dict(basin_dist),
        "per_input": [
            {"idx": idxs[i], "category": cats[i], "cos_to_phase1_universal": float(cos_to_ref[i]),
             "basin": basins[i]}
            for i in range(30)
        ],
        "probes": {
            "probe-basin": probe_basin.__dict__,
            "probe-fine-global": probe_fine_global.__dict__,
            **{f"probe-fine-within-{b}": v.__dict__ for b, v in within_results.items()},
        },
    }
    (processed_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Plot summary
    fig, ax = plt.subplots(figsize=(9, 5))
    names = ["probe-basin", "probe-fine-global"] + [f"probe-fine-within-{b}" for b in sorted(basin_dist.keys())]
    outcomes = [probe_basin, probe_fine_global] + [within_results[b] for b in sorted(basin_dist.keys())]
    x = np.arange(len(names))
    accs = [o.accuracy if not np.isnan(o.accuracy) else 0.0 for o in outcomes]
    stds = [o.accuracy_std if not np.isnan(o.accuracy_std) else 0.0 for o in outcomes]
    chances = [o.chance if not np.isnan(o.chance) else 0.0 for o in outcomes]
    bars = ax.bar(x, accs, yerr=stds, capsize=4, color="tab:blue", alpha=0.8, label="probe accuracy")
    ax.scatter(x, chances, marker="_", s=200, color="grey", label="majority-class chance")
    for i, o in enumerate(outcomes):
        if np.isnan(o.accuracy):
            ax.text(i, 0.05, "n/a\n(too small)", ha="center", fontsize=8, color="red")
        else:
            ax.text(i, accs[i] + 0.04, f"n={o.n_samples}\nk={o.n_classes}",
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("probe-", "") for n in names], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("CV accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Cross-basin probe — does the step-10 hidden state encode basin label or sub-basin info?")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "cross_basin_probe.png", dpi=140)
    plt.close(fig)

    print(f"[{RUN_ID}] basin: acc={probe_basin.accuracy:.3f} ± {probe_basin.accuracy_std:.3f} (chance={probe_basin.chance:.3f})")
    print(f"[{RUN_ID}] fine-global: acc={probe_fine_global.accuracy:.3f} ± {probe_fine_global.accuracy_std:.3f} (chance={probe_fine_global.chance:.3f})")
    for b in sorted(basin_dist.keys()):
        o = within_results[b]
        if np.isnan(o.accuracy):
            print(f"[{RUN_ID}] fine-within-{b}: n/a (n={o.n_samples}, classes={o.n_classes})")
        else:
            print(f"[{RUN_ID}] fine-within-{b}: acc={o.accuracy:.3f} ± {o.accuracy_std:.3f} "
                  f"(chance={o.chance:.3f}, n={o.n_samples}, classes={o.n_classes})")

    print(f"[{RUN_ID}] results saved to {processed_dir / 'results.json'}")


if __name__ == "__main__":
    main()
