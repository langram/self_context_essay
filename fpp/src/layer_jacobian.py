"""Phase 1.3 Experiment K — per-layer LayerNorm σ_max per FPP plan v0.4 §3.4.

Extends Phase 1.2 F's power-iteration to all 12 layers. For each layer's ln_1
and ln_2, compute σ_max(J) at h_fixed on 10 representative inputs (4 Phase 1
reps + 6 Phase 1.1 B reps). Compare trained vs random-init.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.iterate import initial_hidden, iterate_hidden
from src.module_jacobian import load_model_eager_attn, power_iteration_sigma
from src.inputs import build_input_zoo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "phase1_3_layer_jacobian"


@dataclass
class JK:
    label: str
    layer: int
    submodule: str        # ln_1 or ln_2
    input_idx: int
    input_source: str
    sigma_max: float
    n_power_iters: int
    converged: bool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase1-cats", nargs="+",
                        default=["random_tokens", "common", "ambiguous", "structured"])
    parser.add_argument("--max-iter-fixed", type=int, default=200)
    parser.add_argument("--power-iter-max", type=int, default=60)
    args = parser.parse_args()

    fig_dir = PROJECT_ROOT / "outputs" / "figures" / RUN_ID
    report_dir = PROJECT_ROOT / "outputs" / "reports" / RUN_ID
    processed_dir = PROJECT_ROOT / "data" / "processed" / RUN_ID
    for d in (fig_dir, report_dir, processed_dir):
        d.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    all_results: list[JK] = []

    for variant in ("trained", "random"):
        print(f"[{RUN_ID}] loading {variant}")
        model, tokenizer = load_model_eager_attn(
            "gpt2", "float32", "cuda",
            random_init=(variant == "random"), seed=args.seed,
        )
        # Build input set: 4 Phase 1 reps + 6 Phase 1.1 B reps
        phase1_items = build_input_zoo(tokenizer, args.phase1_cats, 1, 32, args.seed)
        from src.extended_inputs import build_nested_zoo
        b_items = build_nested_zoo(tokenizer, seq_len=64)
        b_chosen = [b_items[j] for j in [0, 5, 10, 15, 20, 25]]

        all_inputs: list[tuple[str, int, torch.Tensor]] = []
        for it in phase1_items:
            all_inputs.append(("phase1", it.idx, it.input_ids))
        for it in b_chosen:
            all_inputs.append(("phase1_1b", it.idx, it.input_ids))

        # Compute h_fixed for each input
        fixed_states: dict[tuple[str, int], torch.Tensor] = {}
        for source, idx, ids in tqdm(all_inputs, desc=f"{variant} fixed-points"):
            h0 = initial_hidden(model, ids)
            res = iterate_hidden(model, h0, max_iter=args.max_iter_fixed,
                                 convergence_threshold=1e-3, divergence_factor=100.0, save_every=1)
            fixed_states[(source, idx)] = res.trace[-1].unsqueeze(0).to(h0.device, dtype=h0.dtype)

        # Per-layer power iteration on ln_1 and ln_2
        for layer in tqdm(range(12), desc=f"{variant} layers"):
            block = model.transformer.h[layer]

            def ln1(h, _block=block):
                return _block.ln_1(h)

            def ln2(h, _block=block):
                return _block.ln_2(h)

            for submod_name, fn in (("ln_1", ln1), ("ln_2", ln2)):
                for source, idx, _ in all_inputs:
                    h = fixed_states[(source, idx)]
                    sigma, n_iter, conv = power_iteration_sigma(fn, h, max_iter=args.power_iter_max,
                                                                tol=1e-4, seed=args.seed)
                    all_results.append(JK(
                        label=variant, layer=layer, submodule=submod_name,
                        input_idx=idx if source == "phase1" else 1000 + idx,
                        input_source=source,
                        sigma_max=float(sigma), n_power_iters=int(n_iter), converged=bool(conv),
                    ))
        del model
        torch.cuda.empty_cache()

    # Persist
    payload = [asdict(r) for r in all_results]
    (processed_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Aggregate: mean σ_max per (variant, layer, submodule)
    summary = {}
    for variant in ("trained", "random"):
        for layer in range(12):
            for sm in ("ln_1", "ln_2"):
                vals = [r.sigma_max for r in all_results
                        if r.label == variant and r.layer == layer and r.submodule == sm]
                summary[f"{variant}/L{layer}/{sm}"] = {
                    "n": len(vals),
                    "mean": float(np.mean(vals)) if vals else None,
                    "std": float(np.std(vals)) if vals else None,
                    "min": float(np.min(vals)) if vals else None,
                    "max": float(np.max(vals)) if vals else None,
                }
    (processed_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot: per-layer mean σ_max for each (variant, sm)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, sm in zip(axes, ("ln_1", "ln_2")):
        layers = list(range(12))
        for variant, color in (("trained", "tab:blue"), ("random", "tab:orange")):
            means = [summary[f"{variant}/L{layer}/{sm}"]["mean"] for layer in layers]
            stds = [summary[f"{variant}/L{layer}/{sm}"]["std"] for layer in layers]
            ax.errorbar(layers, means, yerr=stds, marker="o", color=color, label=variant, capsize=3)
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.6)
        ax.set_xlabel("layer index (0–11)")
        ax.set_ylabel(f"σ_max({sm}) at h_fixed")
        ax.set_title(sm)
        ax.legend()
    fig.suptitle("Per-layer LayerNorm σ_max (mean ± std over 10 inputs)")
    fig.tight_layout()
    fig.savefig(fig_dir / "per_layer_sigma.png", dpi=140)
    plt.close(fig)

    print(f"[{RUN_ID}] done")
    for variant in ("trained", "random"):
        print(f"\n  {variant} (mean σ_max ± std per layer):")
        for layer in range(12):
            ln1 = summary[f"{variant}/L{layer}/ln_1"]
            ln2 = summary[f"{variant}/L{layer}/ln_2"]
            print(f"    L{layer:>2d}  ln_1={ln1['mean']:.3f}±{ln1['std']:.3f}  "
                  f"ln_2={ln2['mean']:.3f}±{ln2['std']:.3f}")


if __name__ == "__main__":
    main()
