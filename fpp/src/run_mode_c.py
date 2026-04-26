"""Phase 1.1 Experiment D runner per FPP plan v0.2 §3.5."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from src.inputs import build_input_zoo
from src.iterate import load_model
from src.mode_c_iterate import iterate_mode_c

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "phase1_1_mode_c"


@dataclass
class ModeCRecord:
    idx: int
    category: str
    text: str
    converged: bool
    cycle_period: int | None
    n_steps: int
    final_tokens: list[int]
    final_text: str


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fig_dir = PROJECT_ROOT / "outputs" / "figures" / RUN_ID
    report_dir = PROJECT_ROOT / "outputs" / "reports" / RUN_ID
    processed_dir = PROJECT_ROOT / "data" / "processed" / RUN_ID
    raw_root = PROJECT_ROOT / "data" / "raw" / RUN_ID
    for d in (fig_dir, report_dir, processed_dir, raw_root):
        d.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    print(f"[{RUN_ID}] loading gpt2 (trained)")
    model, tokenizer = load_model("gpt2", "float32", "cuda", random_init=False, seed=args.seed)

    categories = ["random_tokens", "grammatical_random_semantics", "common",
                  "ambiguous", "nonsense_grammatical", "structured"]
    items = build_input_zoo(tokenizer, categories, 10, 32, args.seed)
    print(f"[{RUN_ID}] built {len(items)} inputs")

    records: list[ModeCRecord] = []

    t0 = time.time()
    for item in tqdm(items, desc="mode-c"):
        res = iterate_mode_c(model, tokenizer, item.input_ids, max_iter=args.max_iter)
        records.append(ModeCRecord(
            idx=item.idx,
            category=item.category,
            text=item.text,
            converged=res.converged,
            cycle_period=res.cycle_period,
            n_steps=res.n_steps,
            final_tokens=res.final_tokens,
            final_text=res.final_text,
        ))
        torch.save(
            {"trajectory": res.token_trajectory, "converged": res.converged,
             "cycle_period": res.cycle_period, "n_steps": res.n_steps,
             "final_tokens": res.final_tokens, "final_text": res.final_text,
             "category": item.category, "text": item.text, "input_ids": item.input_ids},
            raw_root / f"trace_{item.idx:03d}.pt",
        )

    elapsed = time.time() - t0

    # Persist records (final_text might contain newlines — escape for JSON)
    payload = []
    for r in records:
        d = asdict(r)
        payload.append(d)
    (processed_dir / "records.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Diversity metrics: how many distinct final token sequences?
    final_seq_strs = [tuple(r.final_tokens) for r in records]
    seq_counts = Counter(final_seq_strs)
    n_distinct = len(seq_counts)

    # By-category breakdowns
    by_cat: dict[str, list[ModeCRecord]] = {}
    for r in records:
        by_cat.setdefault(r.category, []).append(r)

    summary = {
        "timestamp": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "git_commit": _git_commit(),
        "elapsed_seconds": elapsed,
        "n_inputs": len(records),
        "n_converged": sum(1 for r in records if r.converged),
        "n_cycle": sum(1 for r in records if r.cycle_period is not None),
        "n_neither": sum(1 for r in records if not r.converged and r.cycle_period is None),
        "max_iter": args.max_iter,
        "n_distinct_final_sequences": n_distinct,
        "by_category": {
            cat: {
                "n": len(rs),
                "converged": sum(1 for r in rs if r.converged),
                "cycle": sum(1 for r in rs if r.cycle_period is not None),
                "mean_n_steps": sum(r.n_steps for r in rs) / len(rs),
            } for cat, rs in by_cat.items()
        },
    }
    (processed_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot: convergence behaviour distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = sorted(by_cat.keys())
    conv = [sum(1 for r in by_cat[c] if r.converged) for c in cats]
    cyc = [sum(1 for r in by_cat[c] if r.cycle_period is not None) for c in cats]
    neither = [sum(1 for r in by_cat[c] if not r.converged and r.cycle_period is None) for c in cats]
    x = range(len(cats))
    ax.bar(x, conv, label="fixed point (t_n=t_{n+1})", color="tab:green")
    ax.bar(x, cyc, bottom=conv, label="limit cycle", color="tab:orange")
    bottoms = [c + cy for c, cy in zip(conv, cyc)]
    ax.bar(x, neither, bottom=bottoms, label="no convergence in max_iter", color="tab:red")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("count")
    ax.set_title(f"Mode C token-interface convergence (max_iter={args.max_iter})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "convergence_breakdown.png", dpi=140)
    plt.close(fig)

    # Plot: histogram of step counts
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = [r.n_steps for r in records]
    ax.hist(steps, bins=range(0, args.max_iter + 2), color="tab:blue", alpha=0.8)
    ax.set_xlabel("steps to convergence / cycle / max_iter")
    ax.set_ylabel("count")
    ax.set_title("Mode C step distribution")
    fig.tight_layout()
    fig.savefig(fig_dir / "step_distribution.png", dpi=140)
    plt.close(fig)

    print(f"[{RUN_ID}] done in {elapsed:.1f}s")
    print(f"  converged: {summary['n_converged']}/60, cycle: {summary['n_cycle']}/60, neither: {summary['n_neither']}/60")
    print(f"  distinct final sequences: {summary['n_distinct_final_sequences']}")


if __name__ == "__main__":
    main()
