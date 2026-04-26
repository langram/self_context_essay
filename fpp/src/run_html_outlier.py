"""Phase 1.1 Experiment B runner per FPP plan v0.2 §3.3."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.extended_inputs import build_nested_zoo
from src.iterate import initial_hidden, iterate_hidden, load_model
from src.metrics import (
    cosine_similarity_matrix,
    effective_rank,
    project_to_vocab,
    trace_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "phase1_1_html_outlier"


@dataclass
class NestedRecord:
    idx: int
    category: str
    text: str
    nesting_depth: int
    converged: bool
    diverged: bool
    n_steps: int
    final_norm: float
    effective_rank: float
    deltas_summary: dict
    final_top5_per_position: list[list[str]]


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _summarise_deltas(deltas: list[float]) -> dict:
    arr = np.asarray(deltas, dtype=np.float32)
    if arr.size == 0:
        return {"n": 0}
    n = arr.size
    return {
        "n": int(n),
        "first": float(arr[0]),
        "last": float(arr[-1]),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median_last100": float(np.median(arr[-100:])) if n >= 100 else float(np.median(arr)),
        "std_last100": float(np.std(arr[-100:])) if n >= 100 else float(np.std(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save hidden state every N iterations to bound disk usage")
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

    print(f"[{RUN_ID}] loading {args.model}")
    model, tokenizer = load_model(args.model, "float32", "cuda", random_init=False, seed=args.seed)
    items = build_nested_zoo(tokenizer, seq_len=args.seq_len)
    print(f"[{RUN_ID}] built {len(items)} nested-structure inputs")

    records: list[NestedRecord] = []
    final_states: list[np.ndarray] = []
    delta_traces: list[np.ndarray] = []

    t0 = time.time()
    for item in tqdm(items, desc="nested-iterate"):
        h0 = initial_hidden(model, item.input_ids)
        res = iterate_hidden(
            model, h0, max_iter=args.max_iter,
            convergence_threshold=args.threshold, divergence_factor=100.0,
            save_every=args.save_every,
        )
        m = trace_metrics(res.deltas, res.norms, res.converged, res.diverged, res.final_hidden)
        proj = project_to_vocab(res.final_hidden, model, tokenizer, top_k=5)
        records.append(NestedRecord(
            idx=item.idx,
            category=item.category,
            text=item.text,
            nesting_depth=item.nesting_depth,
            converged=m.converged,
            diverged=m.diverged,
            n_steps=m.n_steps,
            final_norm=m.final_norm,
            effective_rank=m.effective_rank,
            deltas_summary=_summarise_deltas(res.deltas),
            final_top5_per_position=proj.top5_token_strs,
        ))
        final_states.append(res.final_hidden.numpy())
        delta_traces.append(np.asarray(res.deltas, dtype=np.float32))
        torch.save(
            {"trace": res.trace, "deltas": res.deltas, "norms": res.norms,
             "converged": res.converged, "diverged": res.diverged,
             "category": item.category, "text": item.text,
             "nesting_depth": item.nesting_depth, "input_ids": item.input_ids,
             "save_every": args.save_every},
            raw_root / f"trace_{item.idx:03d}.pt",
        )

    finals = np.stack(final_states, axis=0)
    sim = cosine_similarity_matrix(finals)
    np.savez(
        processed_dir / "similarity.npz",
        sim=sim,
        labels=np.array([r.category for r in records]),
        depths=np.array([r.nesting_depth for r in records]),
    )

    # Per-record JSON (ASCII-safe top-5 stripped of newline noise)
    payload = []
    for r in records:
        d = asdict(r)
        d["final_top5_per_position"] = [["\\n" if t == "\n" else t for t in row] for row in r.final_top5_per_position]
        payload.append(d)
    (processed_dir / "records.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Phase-1 universal-attractor reference: a known-converged C* from the original 60-input run.
    phase1_records = json.loads(
        (PROJECT_ROOT / "data" / "processed" / "phase1_gpt2_small" / "records_trained.json")
        .read_text(encoding="utf-8")
    )
    converged_phase1 = next(r for r in phase1_records if r["converged"])
    phase1_C_star = torch.load(
        PROJECT_ROOT / "data" / "raw" / "phase1_gpt2_small" / "trained" / f"trace_{converged_phase1['idx']:03d}.pt",
        weights_only=False,
    )["trace"][-1].numpy()  # last saved state = C*
    # Phase 1 used seq_len=32; nested zoo uses seq_len=64. Compare via mean over seq_len → 768-vec.
    ref_vec = phase1_C_star.mean(axis=0)
    ref_vec /= max(np.linalg.norm(ref_vec), 1e-12)
    nested_vecs = finals.mean(axis=1)
    nested_vecs = nested_vecs / np.maximum(np.linalg.norm(nested_vecs, axis=1, keepdims=True), 1e-12)
    cos_to_phase1_universal = nested_vecs @ ref_vec

    # Plot 1 — convergence vs nesting depth
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = sorted(set(r.category for r in records))
    color_map = {c: plt.cm.tab10(i) for i, c in enumerate(cats)}
    for r in records:
        marker = "o" if r.converged else "x"
        ax.scatter(r.nesting_depth, r.n_steps, color=color_map[r.category], marker=marker, s=70,
                   edgecolors="black", linewidths=0.4, label=r.category)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="best")
    ax.set_xlabel("nesting depth")
    ax.set_ylabel("iterations to convergence (or max_iter if not converged)")
    ax.set_title("Convergence vs nesting depth (o = converged, x = max_iter reached)")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(fig_dir / "convergence_vs_depth.png", dpi=140)
    plt.close(fig)

    # Plot 2 — delta trajectories of non-convergers
    non_converged = [(r, d) for r, d in zip(records, delta_traces) if not r.converged]
    if non_converged:
        fig, ax = plt.subplots(figsize=(9, 5))
        for r, d in non_converged:
            ax.plot(np.arange(1, d.size + 1), d, label=f"{r.category} d={r.nesting_depth} #{r.idx}", linewidth=1.0)
        ax.axhline(args.threshold, color="grey", linestyle="--", linewidth=0.8, label=f"threshold={args.threshold}")
        ax.set_yscale("log")
        ax.set_xlabel("iteration step")
        ax.set_ylabel(r"$\|h_{n+1} - h_n\| / \|h_n\|$")
        ax.set_title("Non-converging trajectories (max_iter=1000)")
        ax.legend(fontsize=7, loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / "non_convergence_traces.png", dpi=140)
        plt.close(fig)

    # Plot 3 — cos-to-phase1-universal vs nesting depth (does the converger land at the SAME attractor?)
    fig, ax = plt.subplots(figsize=(8, 5))
    for r, c in zip(records, cos_to_phase1_universal):
        marker = "o" if r.converged else "x"
        ax.scatter(r.nesting_depth, c, color=color_map[r.category], marker=marker, s=70,
                   edgecolors="black", linewidths=0.4, label=r.category)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="best")
    ax.axhline(0.99, color="grey", linestyle="--", linewidth=0.8, label="0.99")
    ax.set_xlabel("nesting depth")
    ax.set_ylabel(r"cos(C*_nested, C*_phase1_universal)  (mean-pooled)")
    ax.set_title("Does the nested-input attractor coincide with Phase 1's universal attractor?")
    fig.tight_layout()
    fig.savefig(fig_dir / "cos_to_phase1_universal.png", dpi=140)
    plt.close(fig)

    elapsed = time.time() - t0

    # Aggregate diagnostics
    n = len(records)
    n_conv = sum(1 for r in records if r.converged)
    n_div = sum(1 for r in records if r.diverged)
    by_cat = {}
    for r in records:
        by_cat.setdefault(r.category, []).append(r)

    # Find non-trivial converged states (cos to phase1 universal < 0.99)
    distinct_attractors = [
        (r, float(c)) for r, c in zip(records, cos_to_phase1_universal)
        if r.converged and c < 0.99
    ]
    same_attractors = [(r, float(c)) for r, c in zip(records, cos_to_phase1_universal) if c >= 0.99]

    summary = {
        "timestamp": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "git_commit": _git_commit(),
        "elapsed_seconds": elapsed,
        "n_inputs": n,
        "n_converged": n_conv,
        "n_diverged": n_div,
        "max_iter": args.max_iter,
        "threshold": args.threshold,
        "seq_len": args.seq_len,
        "save_every": args.save_every,
        "by_category": {
            cat: {
                "n": len(rs),
                "converged": sum(1 for r in rs if r.converged),
                "mean_steps_converged": float(np.mean([r.n_steps for r in rs if r.converged]))
                                        if any(r.converged for r in rs) else None,
                "mean_eff_rank": float(np.mean([r.effective_rank for r in rs])),
                "mean_final_norm": float(np.mean([r.final_norm for r in rs])),
            } for cat, rs in by_cat.items()
        },
        "n_match_phase1_universal": len(same_attractors),
        "n_distinct_attractors": len(distinct_attractors),
    }
    (processed_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{RUN_ID}] done in {elapsed:.1f}s. {n_conv}/{n} converged; "
          f"{summary['n_match_phase1_universal']} match phase1 universal, "
          f"{summary['n_distinct_attractors']} land at distinct attractors.")


if __name__ == "__main__":
    main()
