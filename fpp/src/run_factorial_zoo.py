"""Phase 1.2 Experiment H runner per FPP plan v0.3 §3.5.

Runs mode-A iteration on the user-approved 24-input factorial zoo, assigns each
output to a basin via cos-to-Phase-1-universal, then does an ANOVA-style factor
effect analysis: for each factor, what fraction of within-pair flips actually
change the basin?
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.factorial_zoo import build_factorial_zoo
from src.iterate import initial_hidden, iterate_hidden, load_model
from src.metrics import effective_rank


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "phase1_2_factorial_zoo"


@dataclass
class HRecord:
    idx: int
    text: str
    case: str
    markup: str
    punct: str
    content: str
    n_tokens: int
    converged: bool
    diverged: bool
    n_steps: int
    final_norm: float
    effective_rank: float
    cos_to_phase1_universal: float
    basin: str


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _assign_basin(cos_to_p1: float) -> str:
    if cos_to_p1 > 0.999:
        return "lowercase"
    if cos_to_p1 < 0.7:
        return "capital"
    return "hybrid"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()

    fig_dir = PROJECT_ROOT / "outputs" / "figures" / RUN_ID
    report_dir = PROJECT_ROOT / "outputs" / "reports" / RUN_ID
    processed_dir = PROJECT_ROOT / "data" / "processed" / RUN_ID
    raw_root = PROJECT_ROOT / "data" / "raw" / RUN_ID
    for d in (fig_dir, report_dir, processed_dir, raw_root):
        d.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    print(f"[{RUN_ID}] loading trained gpt2")
    model, tokenizer = load_model("gpt2", "float32", "cuda", random_init=False, seed=args.seed)

    items = build_factorial_zoo(tokenizer, seq_len=64)

    # Phase 1 universal reference (mean-pooled)
    p1_ref = None
    for i in range(60):
        d = torch.load(PROJECT_ROOT / "data" / "raw" / "phase1_gpt2_small" / "trained" / f"trace_{i:03d}.pt",
                       weights_only=False)
        if d["converged"]:
            p1_ref = d["trace"][-1].numpy()
            break
    p1_ref_pooled = p1_ref.mean(axis=0)
    p1_ref_pooled /= max(np.linalg.norm(p1_ref_pooled), 1e-12)

    records: list[HRecord] = []
    finals: list[np.ndarray] = []

    t0 = time.time()
    for item in tqdm(items, desc="factorial-zoo"):
        h0 = initial_hidden(model, item.input_ids)
        res = iterate_hidden(
            model, h0, max_iter=args.max_iter,
            convergence_threshold=args.threshold, divergence_factor=100.0,
            save_every=args.save_every,
        )
        fhid = res.final_hidden.numpy()
        fvec_pooled = fhid.mean(axis=0)
        fvec_pooled_n = fvec_pooled / max(np.linalg.norm(fvec_pooled), 1e-12)
        cos_to_p1 = float(fvec_pooled_n @ p1_ref_pooled)
        basin = _assign_basin(cos_to_p1)

        records.append(HRecord(
            idx=item.idx,
            text=item.text,
            case=item.case,
            markup=item.markup,
            punct=item.punct,
            content=item.content,
            n_tokens=item.n_tokens,
            converged=res.converged,
            diverged=res.diverged,
            n_steps=res.n_steps,
            final_norm=float(np.linalg.norm(fhid)),
            effective_rank=effective_rank(fhid),
            cos_to_phase1_universal=cos_to_p1,
            basin=basin,
        ))
        finals.append(fhid)
        torch.save(
            {"trace": res.trace, "deltas": res.deltas, "norms": res.norms,
             "converged": res.converged, "diverged": res.diverged,
             "case": item.case, "markup": item.markup, "punct": item.punct, "content": item.content,
             "text": item.text, "input_ids": item.input_ids, "save_every": args.save_every},
            raw_root / f"trace_{item.idx:03d}.pt",
        )
    elapsed = time.time() - t0

    # Persist records
    (processed_dir / "records.json").write_text(
        json.dumps([asdict(r) for r in records], indent=2), encoding="utf-8"
    )

    basin_dist = Counter(r.basin for r in records)

    # Factor effect analysis: for each factor f, find pairs (a, b) that differ ONLY in f.
    # Compute fraction of such pairs where basin differs.
    factors = ["case", "markup", "punct", "content"]
    other_factors = {f: [g for g in factors if g != f] for f in factors}

    def _key(r: HRecord, except_factor: str) -> tuple:
        return tuple(getattr(r, g) for g in factors if g != except_factor)

    factor_effects: dict[str, dict] = {}
    for f in factors:
        # Group by all-other-factors-equal
        groups: dict[tuple, list[HRecord]] = {}
        for r in records:
            groups.setdefault(_key(r, f), []).append(r)
        # In each group, count pairs of records that differ on `f` and compare basins
        n_pairs = 0
        n_basin_flip = 0
        flip_examples = []
        for key, rs in groups.items():
            for a, b in itertools.combinations(rs, 2):
                if getattr(a, f) != getattr(b, f):
                    n_pairs += 1
                    if a.basin != b.basin:
                        n_basin_flip += 1
                        flip_examples.append({
                            "fixed": dict(zip(other_factors[f], key)),
                            "a_factor": getattr(a, f), "a_idx": a.idx, "a_basin": a.basin,
                            "b_factor": getattr(b, f), "b_idx": b.idx, "b_basin": b.basin,
                        })
        factor_effects[f] = {
            "n_within_factor_pairs": n_pairs,
            "n_basin_flip": n_basin_flip,
            "flip_rate": n_basin_flip / n_pairs if n_pairs else None,
            "flip_examples": flip_examples,
        }

    summary = {
        "timestamp": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "git_commit": _git_commit(),
        "elapsed_seconds": elapsed,
        "n_inputs": len(records),
        "n_converged": sum(1 for r in records if r.converged),
        "basin_distribution": dict(basin_dist),
        "max_iter": args.max_iter,
        "factor_effects": factor_effects,
    }
    (processed_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot: factor flip rates
    fig, ax = plt.subplots(figsize=(7, 5))
    rates = [factor_effects[f]["flip_rate"] for f in factors]
    n_pairs = [factor_effects[f]["n_within_factor_pairs"] for f in factors]
    bars = ax.bar(factors, rates, color="tab:blue", alpha=0.85)
    for bar, n, r in zip(bars, n_pairs, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, (r or 0) + 0.02,
                f"n={n}\nrate={r:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("fraction of within-factor-flip pairs that change basin")
    ax.set_title("Factor effect on basin assignment (24-input factorial zoo)")
    fig.tight_layout()
    fig.savefig(fig_dir / "factor_flip_rates.png", dpi=140)
    plt.close(fig)

    # Plot: per-input basin coloured by case (the strongest a priori factor)
    fig, ax = plt.subplots(figsize=(11, 5))
    basin_to_y = {"capital": 0, "hybrid": 1, "lowercase": 2}
    case_colors = {"capital": "tab:red", "lowercase": "tab:blue"}
    for r in records:
        ax.scatter(r.idx, basin_to_y[r.basin], s=120,
                   color=case_colors[r.case],
                   marker={"with": "s", "without": "o"}[r.markup],
                   edgecolors="black", linewidths=0.5,
                   alpha=0.9 if r.punct == "high" else 0.45)
        ax.text(r.idx, basin_to_y[r.basin] + 0.15, r.content[:3], fontsize=7, ha="center")
    ax.set_yticks(list(basin_to_y.values()))
    ax.set_yticklabels(list(basin_to_y.keys()))
    ax.set_xlabel("input idx")
    ax.set_ylabel("basin")
    ax.set_title("Per-input basin (color=case [red=cap, blue=low], "
                 "shape=markup [□=with, ○=without], alpha=punct [solid=high, faded=low])")
    fig.tight_layout()
    fig.savefig(fig_dir / "per_input_basin.png", dpi=140)
    plt.close(fig)

    print(f"[{RUN_ID}] done in {elapsed:.1f}s")
    print(f"  basin distribution: {dict(basin_dist)}")
    for f in factors:
        fe = factor_effects[f]
        rate = fe["flip_rate"]
        print(f"  factor {f:>8s}: {fe['n_basin_flip']:>2d}/{fe['n_within_factor_pairs']:>2d} pairs flip basin "
              f"(rate = {rate:.3f})" if rate is not None else f"  factor {f:>8s}: no pairs")


if __name__ == "__main__":
    main()
