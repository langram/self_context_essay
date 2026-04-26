"""Phase 1.3 Experiment L (expanded) — γ/state decomposition for LayerNorm σ_max.

Original L target (plan §3.5): for layer 6 ln_1, decompose σ_max ≈ 0.12 into
contributions from learned γ vs fixed-point hidden-state norm.

User-specified expansion: also do the same decomposition for L0 ln_1 (the K
outlier at σ_max = 0.038), to explain why early layers contract harder.

Four conditions per (layer, input):
  A1: trained γ × original h_fixed (norm ≈ 2563)  ← matches Phase 1.2 F
  A2: γ=1     × original h_fixed                  ← isolates state-norm contribution
  A3: trained γ × rescaled h (norm = 100)         ← isolates γ contribution
  A4: γ=1     × rescaled h                        ← baseline, should be ≈ 1.0
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.iterate import initial_hidden, iterate_hidden
from src.module_jacobian import load_model_eager_attn, power_iteration_sigma
from src.inputs import build_input_zoo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "phase1_3_ln_decomposition"

LAYERS_TO_TEST = (0, 6)   # 0 = K's outlier (σ_max = 0.038), 6 = Phase 1.2 F baseline
TARGET_RESCALED_NORM = 100.0


@dataclass
class LDResult:
    layer: int
    submodule: str       # "ln_1" only for now
    input_source: str
    input_idx: int
    h_orig_norm: float
    h_rescaled_norm: float
    sigma_A1: float       # trained γ, orig h
    sigma_A2: float       # γ=1, orig h
    sigma_A3: float       # trained γ, rescaled h
    sigma_A4: float       # γ=1, rescaled h


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
    parser.add_argument("--phase1-cats", nargs="+",
                        default=["random_tokens", "common", "ambiguous", "structured"])
    parser.add_argument("--max-iter-fixed", type=int, default=200)
    parser.add_argument("--power-iter-max", type=int, default=60)
    parser.add_argument("--target-norm", type=float, default=TARGET_RESCALED_NORM)
    args = parser.parse_args()

    fig_dir = PROJECT_ROOT / "outputs" / "figures" / RUN_ID
    report_dir = PROJECT_ROOT / "outputs" / "reports" / RUN_ID
    processed_dir = PROJECT_ROOT / "data" / "processed" / RUN_ID
    for d in (fig_dir, report_dir, processed_dir):
        d.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    print(f"[{RUN_ID}] loading trained gpt2")
    model_orig, tokenizer = load_model_eager_attn(
        "gpt2", "float32", "cuda", random_init=False, seed=args.seed,
    )

    # Build a γ=1 copy ONCE, modify its layer-0 and layer-6 ln_1 weights
    model_g1 = copy.deepcopy(model_orig)
    for layer in LAYERS_TO_TEST:
        model_g1.transformer.h[layer].ln_1.weight.data.fill_(1.0)
        # Plan §6.2: only γ controlled; β stays. We override only weight.

    # Build the input set: 4 Phase 1 reps + 6 Phase 1.1 B reps (same as F/K)
    phase1_items = build_input_zoo(tokenizer, args.phase1_cats, 1, 32, args.seed)
    from src.extended_inputs import build_nested_zoo
    b_items = build_nested_zoo(tokenizer, seq_len=64)
    b_chosen = [b_items[j] for j in [0, 5, 10, 15, 20, 25]]

    all_inputs: list[tuple[str, int, torch.Tensor]] = []
    for it in phase1_items:
        all_inputs.append(("phase1", it.idx, it.input_ids))
    for it in b_chosen:
        all_inputs.append(("phase1_1b", it.idx, it.input_ids))

    # Compute h_fixed using the original (unmodified) model
    fixed_states: dict[tuple[str, int], torch.Tensor] = {}
    for source, idx, ids in tqdm(all_inputs, desc="fixed-points"):
        h0 = initial_hidden(model_orig, ids)
        res = iterate_hidden(model_orig, h0, max_iter=args.max_iter_fixed,
                             convergence_threshold=1e-3, divergence_factor=100.0, save_every=1)
        fixed_states[(source, idx)] = res.trace[-1].unsqueeze(0).to(h0.device, dtype=h0.dtype)

    all_results: list[LDResult] = []

    for layer in LAYERS_TO_TEST:
        ln1_orig = model_orig.transformer.h[layer].ln_1
        ln1_g1 = model_g1.transformer.h[layer].ln_1

        def fn_orig(h, _ln=ln1_orig):
            return _ln(h)

        def fn_g1(h, _ln=ln1_g1):
            return _ln(h)

        for source, idx, _ in tqdm(all_inputs, desc=f"L{layer} decomposition"):
            h_orig = fixed_states[(source, idx)]
            h_orig_norm = h_orig.norm().item()

            # Rescale h to TARGET_RESCALED_NORM while preserving direction
            h_rescaled = h_orig * (args.target_norm / h_orig_norm)
            h_rescaled_norm = h_rescaled.norm().item()

            # Four conditions
            sigma_A1, _, _ = power_iteration_sigma(fn_orig, h_orig, max_iter=args.power_iter_max, tol=1e-4, seed=args.seed)
            sigma_A2, _, _ = power_iteration_sigma(fn_g1, h_orig, max_iter=args.power_iter_max, tol=1e-4, seed=args.seed)
            sigma_A3, _, _ = power_iteration_sigma(fn_orig, h_rescaled, max_iter=args.power_iter_max, tol=1e-4, seed=args.seed)
            sigma_A4, _, _ = power_iteration_sigma(fn_g1, h_rescaled, max_iter=args.power_iter_max, tol=1e-4, seed=args.seed)

            all_results.append(LDResult(
                layer=layer, submodule="ln_1",
                input_source=source,
                input_idx=idx if source == "phase1" else 1000 + idx,
                h_orig_norm=float(h_orig_norm),
                h_rescaled_norm=float(h_rescaled_norm),
                sigma_A1=float(sigma_A1),
                sigma_A2=float(sigma_A2),
                sigma_A3=float(sigma_A3),
                sigma_A4=float(sigma_A4),
            ))

    # Persist
    payload = [asdict(r) for r in all_results]
    (processed_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Aggregate per layer
    summary = {"meta": {"timestamp": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
                        "git_commit": _git_commit(),
                        "target_rescaled_norm": args.target_norm}}
    for layer in LAYERS_TO_TEST:
        rs = [r for r in all_results if r.layer == layer]
        agg = {}
        for label in ("sigma_A1", "sigma_A2", "sigma_A3", "sigma_A4"):
            vals = [getattr(r, label) for r in rs]
            agg[label] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        # Ratio diagnostics — per the plan §3.5 keys
        # σ_max(A2) / σ_max(A4) → state-norm contribution
        # σ_max(A3) / σ_max(A4) → γ contribution
        # σ_max(A1) / σ_max(A4) → total contraction
        ratio_state_only = np.mean([r.sigma_A2 / max(r.sigma_A4, 1e-12) for r in rs])
        ratio_gamma_only = np.mean([r.sigma_A3 / max(r.sigma_A4, 1e-12) for r in rs])
        ratio_total = np.mean([r.sigma_A1 / max(r.sigma_A4, 1e-12) for r in rs])
        # Decomposition cross-term
        coupling = np.mean([
            (r.sigma_A1 * r.sigma_A4) / max(r.sigma_A2 * r.sigma_A3, 1e-12)
            for r in rs
        ])
        agg["ratio_total_A1_over_A4"] = float(ratio_total)
        agg["ratio_state_A2_over_A4"] = float(ratio_state_only)
        agg["ratio_gamma_A3_over_A4"] = float(ratio_gamma_only)
        agg["coupling_factor"] = float(coupling)
        summary[f"layer_{layer}"] = agg

    (processed_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot: bar chart of σ_A1/A2/A3/A4 means per layer
    fig, axes = plt.subplots(1, len(LAYERS_TO_TEST), figsize=(6 * len(LAYERS_TO_TEST), 5), sharey=True)
    if len(LAYERS_TO_TEST) == 1:
        axes = [axes]
    for ax, layer in zip(axes, LAYERS_TO_TEST):
        agg = summary[f"layer_{layer}"]
        names = ["A1\n(trained γ,\norig h)", "A2\n(γ=1,\norig h)", "A3\n(trained γ,\nrescaled h)", "A4\n(γ=1,\nrescaled h)"]
        means = [agg[k]["mean"] for k in ("sigma_A1", "sigma_A2", "sigma_A3", "sigma_A4")]
        stds = [agg[k]["std"] for k in ("sigma_A1", "sigma_A2", "sigma_A3", "sigma_A4")]
        x = np.arange(4)
        ax.bar(x, means, yerr=stds, color=["tab:blue", "tab:orange", "tab:green", "tab:grey"], alpha=0.85, capsize=4)
        for xi, m in zip(x, means):
            ax.text(xi, m + 0.02, f"{m:.3f}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_title(f"Layer {layer} ln_1 σ_max under 4 conditions")
        ax.set_ylabel("σ_max(ln_1)")
    fig.tight_layout()
    fig.savefig(fig_dir / "decomposition_bars.png", dpi=140)
    plt.close(fig)

    # Print summary
    print(f"\n[{RUN_ID}] done")
    for layer in LAYERS_TO_TEST:
        agg = summary[f"layer_{layer}"]
        print(f"\n  Layer {layer} ln_1 (mean σ_max ± std):")
        for k in ("sigma_A1", "sigma_A2", "sigma_A3", "sigma_A4"):
            print(f"    {k}: {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}")
        print(f"    ratio_total (A1/A4):  {agg['ratio_total_A1_over_A4']:.3f}")
        print(f"    state-only (A2/A4):  {agg['ratio_state_A2_over_A4']:.3f}")
        print(f"    γ-only (A3/A4):       {agg['ratio_gamma_A3_over_A4']:.3f}")
        print(f"    coupling (A1·A4)/(A2·A3): {agg['coupling_factor']:.3f}")


if __name__ == "__main__":
    main()
