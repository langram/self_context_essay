"""Phase 1.2 Experiment G — B × C joint per FPP plan v0.3 §3.4.

Re-runs Phase 1.1 Experiment B's 30-input nested zoo through Variant C1
cancel-pos with max_iter=1000, then compares the new C* set to B's baseline
C* to determine whether the Capital basin survives wpe ablation.
"""

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
from src.iterate import initial_hidden, load_model
from src.metrics import effective_rank, project_to_vocab
from src.posfree_iterate import iterate_hidden_cancelpos


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "phase1_2_basin_position_ablation"


@dataclass
class GRecord:
    idx: int
    category: str
    text: str
    nesting_depth: int
    converged: bool
    diverged: bool
    n_steps: int
    final_norm: float
    effective_rank: float
    cos_to_b_baseline: float           # cos to the B (original mode-A) C* of the same input
    cos_to_phase1_universal: float     # cos to a known Phase 1 lowercase converger (mean-pooled)


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


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
    items = build_nested_zoo(tokenizer, seq_len=64)

    # Load B baseline C*'s (mean-pooled) for comparison
    b_baselines: list[np.ndarray] = []
    for i in range(30):
        d = torch.load(PROJECT_ROOT / "data" / "raw" / "phase1_1_html_outlier" / f"trace_{i:03d}.pt",
                       weights_only=False)
        b_baselines.append(d["trace"][-1].numpy())  # final saved state
    b_baselines = np.stack(b_baselines, axis=0)  # [30, 64, 768]

    # Phase 1 universal reference (any converged Phase 1 trace's final state)
    p1_ref = None
    for i in range(60):
        d = torch.load(PROJECT_ROOT / "data" / "raw" / "phase1_gpt2_small" / "trained" / f"trace_{i:03d}.pt",
                       weights_only=False)
        if d["converged"]:
            p1_ref = d["trace"][-1].numpy()
            break
    p1_ref_pooled = p1_ref.mean(axis=0)
    p1_ref_pooled /= max(np.linalg.norm(p1_ref_pooled), 1e-12)

    records: list[GRecord] = []
    finals: list[np.ndarray] = []

    t0 = time.time()
    for item in tqdm(items, desc="cancelpos-nested"):
        h0 = initial_hidden(model, item.input_ids)
        res = iterate_hidden_cancelpos(
            model, h0,
            max_iter=args.max_iter,
            convergence_threshold=args.threshold,
            divergence_factor=100.0,
            save_every=args.save_every,
        )
        fhid = res.final_hidden.numpy()
        fvec_pooled = fhid.mean(axis=0)
        fvec_pooled_n = fvec_pooled / max(np.linalg.norm(fvec_pooled), 1e-12)
        cos_to_p1 = float(fvec_pooled_n @ p1_ref_pooled)

        b_baseline = b_baselines[item.idx]
        # Use full-flat cosine for cos_to_b_baseline so it's the strict same-input comparison
        a = fhid.reshape(-1)
        b = b_baseline.reshape(-1)
        cos_to_b = float((a / max(np.linalg.norm(a), 1e-12)) @ (b / max(np.linalg.norm(b), 1e-12)))

        records.append(GRecord(
            idx=item.idx,
            category=item.category,
            text=item.text,
            nesting_depth=item.nesting_depth,
            converged=res.converged,
            diverged=res.diverged,
            n_steps=res.n_steps,
            final_norm=float(np.linalg.norm(fhid)),
            effective_rank=effective_rank(fhid),
            cos_to_b_baseline=cos_to_b,
            cos_to_phase1_universal=cos_to_p1,
        ))
        finals.append(fhid)
        torch.save(
            {"trace": res.trace, "deltas": res.deltas, "norms": res.norms,
             "converged": res.converged, "diverged": res.diverged,
             "category": item.category, "text": item.text,
             "nesting_depth": item.nesting_depth, "input_ids": item.input_ids,
             "save_every": args.save_every, "variant": "C1_cancelpos"},
            raw_root / f"trace_{item.idx:03d}.pt",
        )

    elapsed = time.time() - t0
    finals = np.stack(finals, axis=0)

    # Pairwise similarity within new C*
    flat = finals.reshape(30, -1)
    flat /= np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)
    sim = flat @ flat.T
    np.savez(processed_dir / "similarity.npz", sim=sim,
             labels=np.array([r.category for r in records]))

    # Persist records
    payload = []
    for r in records:
        payload.append(asdict(r))
    (processed_dir / "records.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Determine basin distribution under cancel-pos vs original B
    def assign(c: float) -> str:
        if c > 0.999: return "lowercase"
        if c < 0.7: return "capital"
        return "hybrid"

    new_basins = [assign(r.cos_to_phase1_universal) for r in records]
    # Original B basins (from records.json of B run, recompute if needed)
    b_basins = []
    for i in range(30):
        b_pooled = b_baselines[i].mean(axis=0)
        b_pooled /= max(np.linalg.norm(b_pooled), 1e-12)
        cos = float(b_pooled @ p1_ref_pooled)
        b_basins.append(assign(cos))

    from collections import Counter
    new_dist = Counter(new_basins)
    b_dist = Counter(b_basins)

    transitions = Counter([(b_basins[i], new_basins[i]) for i in range(30)])

    summary = {
        "timestamp": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "git_commit": _git_commit(),
        "elapsed_seconds": elapsed,
        "n_inputs": 30,
        "max_iter": args.max_iter,
        "B_basin_distribution": dict(b_dist),
        "G_basin_distribution_after_cancelpos": dict(new_dist),
        "basin_transitions_BtoG": {f"{old}->{new}": cnt for (old, new), cnt in transitions.items()},
        "n_converged": sum(1 for r in records if r.converged),
        "mean_cos_to_B_baseline": float(np.mean([r.cos_to_b_baseline for r in records])),
        "min_cos_to_B_baseline": float(np.min([r.cos_to_b_baseline for r in records])),
    }
    (processed_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot: per-input cos comparison (B-baseline cos vs Phase 1 universal cos)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(30)
    ax.bar(x - 0.2, [r.cos_to_b_baseline for r in records], width=0.4, label="cos to B baseline (same input, mode-A C*)")
    ax.bar(x + 0.2, [r.cos_to_phase1_universal for r in records], width=0.4, label="cos to Phase 1 universal (mean-pooled)")
    cats = sorted(set(r.category for r in records))
    cat_colors = {c: plt.cm.tab10(i) for i, c in enumerate(cats)}
    for i, r in enumerate(records):
        ax.scatter([i], [-0.05], color=cat_colors[r.category], s=40, marker="s")
    ax.axhline(0.999, color="grey", linestyle=":", linewidth=0.6)
    ax.axhline(0.7, color="grey", linestyle=":", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.idx}" for r in records], fontsize=7)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel("input idx (color = nested category)")
    ax.set_ylabel("cosine similarity")
    ax.set_title("Cancel-pos basin assignment vs B baseline — does Capital basin survive wpe ablation?")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "cancelpos_vs_b_per_input.png", dpi=140)
    plt.close(fig)

    print(f"[{RUN_ID}] done in {elapsed:.1f}s")
    print(f"  B basin distribution:  {dict(b_dist)}")
    print(f"  G basin distribution:  {dict(new_dist)}")
    print(f"  transitions B -> G: {summary['basin_transitions_BtoG']}")
    print(f"  mean cos to B baseline: {summary['mean_cos_to_B_baseline']:.4f}")


if __name__ == "__main__":
    main()
