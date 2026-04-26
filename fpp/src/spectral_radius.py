"""Phase 1.3 Experiment M (reframed) — spectral radius + third-endpoint characterisation.

Reframed per user feedback: M does NOT verify J. M solves two independent
methodological problems:
  1. ρ(J) vs σ_max(J) gap on Phase 1.2 F's sub-modules (the original M target)
  2. Spectral characterisation of Experiment I's "third endpoint regime"
     (square-bracket inputs at cos_lc=0.90, cos_html=0.61) — neither L nor
     the original M handles this Group-Cheap discovery.

Method: Arnoldi iteration via scipy.sparse.linalg.eigs as primary
(robust for asymmetric J), with power iteration on J as a fallback.
σ_max is also computed via the existing J^T·J power iteration for
direct comparison.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla
import torch
from tqdm import tqdm

from src.iterate import initial_hidden, iterate_hidden
from src.module_jacobian import (
    ANALYSE_LAYER,
    load_model_eager_attn,
    make_submodule_fns,
    power_iteration_sigma,
)
from src.inputs import build_input_zoo
from src.markup_triangulation import SQUARE  # 5 square-bracket inputs from Experiment I


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "phase1_3_spectral_radius"


SUBMODULES = (
    "ln_1",
    "attn_sublayer",
    "ln_2",
    "mlp_sublayer",
    f"full_block_L{ANALYSE_LAYER}",
    "ln_f",
    "full_stack_posfree",
)
TEST_POINTS = ("h_0", "h_1", "h_fixed")


@dataclass
class SRResult:
    label: str
    input_idx: int
    input_source: str
    test_point: str
    submodule: str
    sigma_max: float
    rho: float                      # |λ_max| spectral radius via Arnoldi (or NaN if all methods failed)
    rho_method: str                 # "arnoldi", "power_iter_J", or "failed"
    arnoldi_n_iter: int             # iterations Arnoldi used (0 if not used)
    converged: bool


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def spectral_radius_arnoldi(
    f: Callable[[torch.Tensor], torch.Tensor],
    h: torch.Tensor,
    n_eigs: int = 1,
    tol: float = 1e-4,
    max_arnoldi_iter: int = 80,
) -> tuple[float, int, str]:
    """Estimate ρ(J_f(h)) by Arnoldi via scipy.sparse.linalg.eigs.

    Returns (|λ_max|, n_iter, method). On failure returns (nan, 0, 'failed').
    """
    n = h.numel()
    shape_orig = h.shape
    device = h.device
    dtype = h.dtype

    # Counter passed by closure
    state = {"n_calls": 0}

    def matvec(v_np: np.ndarray) -> np.ndarray:
        state["n_calls"] += 1
        with torch.no_grad():
            v_t = torch.from_numpy(v_np.real.astype(np.float32)).reshape(shape_orig).to(device=device, dtype=dtype)
        # jvp with create_graph=False, strict=False
        _, Jv = torch.autograd.functional.jvp(f, h, v=v_t, create_graph=False, strict=False)
        return Jv.detach().cpu().numpy().reshape(-1).astype(np.float64)

    A = spla.LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vals, _ = spla.eigs(A, k=n_eigs, which="LM",
                                ncv=min(2 * n_eigs + 8, n - 1),
                                maxiter=max_arnoldi_iter, tol=tol)
        rho = float(np.max(np.abs(vals)))
        return rho, state["n_calls"], "arnoldi"
    except Exception as e:
        return float("nan"), state["n_calls"], "failed"


def spectral_radius_power_iter_J(
    f: Callable[[torch.Tensor], torch.Tensor],
    h: torch.Tensor,
    max_iter: int = 200,
    tol: float = 1e-4,
    seed: int = 42,
) -> tuple[float, int, bool]:
    """Power iteration on J itself (not J^T·J). For asymmetric J this finds the
    largest-magnitude eigenvalue if there is a strict gap |λ_1| > |λ_2|.
    Used as fallback when Arnoldi fails.
    """
    g = torch.Generator(device=h.device).manual_seed(seed)
    v = torch.randn(h.shape, generator=g, device=h.device, dtype=h.dtype)
    v = v / (v.norm() + 1e-12)
    lam_prev = 0.0
    for it in range(max_iter):
        _, Jv = torch.autograd.functional.jvp(f, h, v=v, create_graph=False, strict=False)
        # Rayleigh quotient: λ ≈ <v, Jv> / <v, v> (= <v, Jv> since v is unit)
        lam_est = float((v.flatten() * Jv.flatten()).sum().item())
        Jv_norm = Jv.norm().item()
        if Jv_norm < 1e-20:
            return abs(lam_est), it + 1, True
        v = Jv / Jv_norm
        if it > 5 and abs(lam_est - lam_prev) < tol * max(abs(lam_est), 1e-12):
            return abs(lam_est), it + 1, True
        lam_prev = lam_est
    return abs(lam_prev), max_iter, False


def get_test_points(model, input_ids: torch.Tensor, max_iter: int = 200) -> dict[str, torch.Tensor]:
    h0 = initial_hidden(model, input_ids)
    res = iterate_hidden(model, h0, max_iter=max_iter,
                        convergence_threshold=1e-3, divergence_factor=100.0, save_every=1)
    return {
        "h_0": h0,
        "h_1": res.trace[1].unsqueeze(0).to(h0.device, dtype=h0.dtype),
        "h_fixed": res.trace[-1].unsqueeze(0).to(h0.device, dtype=h0.dtype),
    }


def encode_square_inputs(tokenizer, seq_len: int = 64) -> list[tuple[int, str, torch.Tensor]]:
    items: list[tuple[int, str, torch.Tensor]] = []
    for i, text in enumerate(SQUARE):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= seq_len:
            ids = ids[:seq_len]
        else:
            pad = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
            ids = ids + [pad] * (seq_len - len(ids))
        items.append((i, text, torch.tensor(ids, dtype=torch.long)))
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase1-cats", nargs="+",
                        default=["random_tokens", "common", "ambiguous", "structured"])
    parser.add_argument("--max-iter-fixed", type=int, default=200)
    parser.add_argument("--power-iter-max", type=int, default=60)
    parser.add_argument("--arnoldi-max-iter", type=int, default=80)
    args = parser.parse_args()

    fig_dir = PROJECT_ROOT / "outputs" / "figures" / RUN_ID
    report_dir = PROJECT_ROOT / "outputs" / "reports" / RUN_ID
    processed_dir = PROJECT_ROOT / "data" / "processed" / RUN_ID
    for d in (fig_dir, report_dir, processed_dir):
        d.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    all_results: list[SRResult] = []

    for variant in ("trained", "random"):
        print(f"[{RUN_ID}] loading {variant}")
        model, tokenizer = load_model_eager_attn(
            "gpt2", "float32", "cuda",
            random_init=(variant == "random"), seed=args.seed,
        )
        submods = make_submodule_fns(model)

        phase1_items = build_input_zoo(tokenizer, args.phase1_cats, 1, 32, args.seed)
        from src.extended_inputs import build_nested_zoo
        b_items = build_nested_zoo(tokenizer, seq_len=64)
        b_chosen = [b_items[j] for j in [0, 5, 10, 15, 20, 25]]

        all_inputs: list[tuple[str, int, torch.Tensor]] = []
        for it in phase1_items:
            all_inputs.append(("phase1", it.idx, it.input_ids))
        for it in b_chosen:
            all_inputs.append(("phase1_1b", it.idx, it.input_ids))
        # Third-endpoint regime: 5 square-bracket inputs from Experiment I
        for j, _, ids in encode_square_inputs(tokenizer, seq_len=64):
            all_inputs.append(("phase1_3i_square", j, ids))

        for source, idx, input_ids in tqdm(all_inputs, desc=f"{variant}"):
            tps = get_test_points(model, input_ids, max_iter=args.max_iter_fixed)
            for tp_name, h in tps.items():
                for sm_name, fn in submods.items():
                    if sm_name not in SUBMODULES:
                        continue
                    # σ_max via J^T·J (existing method)
                    sigma_max, _, _ = power_iteration_sigma(
                        fn, h, max_iter=args.power_iter_max, tol=1e-4, seed=args.seed
                    )
                    # ρ via Arnoldi
                    rho, n_iter, method = spectral_radius_arnoldi(
                        fn, h, n_eigs=1, tol=1e-4, max_arnoldi_iter=args.arnoldi_max_iter
                    )
                    if method == "failed" or np.isnan(rho):
                        # Fallback to direct power iteration on J
                        rho_pi, n_iter_pi, conv_pi = spectral_radius_power_iter_J(
                            fn, h, max_iter=200, tol=1e-4, seed=args.seed
                        )
                        rho = rho_pi
                        n_iter = n_iter_pi
                        method = "power_iter_J"
                        converged = conv_pi
                    else:
                        converged = True
                    all_results.append(SRResult(
                        label=variant,
                        input_idx=idx if source.startswith("phase1") else 1000 + idx,
                        input_source=source,
                        test_point=tp_name,
                        submodule=sm_name,
                        sigma_max=float(sigma_max),
                        rho=float(rho),
                        rho_method=method,
                        arnoldi_n_iter=int(n_iter),
                        converged=bool(converged),
                    ))

        del model
        torch.cuda.empty_cache()

    # Persist
    payload = [asdict(r) for r in all_results]
    (processed_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Aggregate: mean σ_max and ρ per (variant, source, test_point, submodule)
    summary: dict = {}
    for variant in ("trained", "random"):
        for source in ("phase1", "phase1_1b", "phase1_3i_square"):
            for tp in TEST_POINTS:
                for sm in SUBMODULES:
                    rs = [r for r in all_results
                          if r.label == variant and r.input_source == source
                          and r.test_point == tp and r.submodule == sm]
                    sig = [r.sigma_max for r in rs if not np.isnan(r.sigma_max)]
                    rho = [r.rho for r in rs if not np.isnan(r.rho)]
                    summary[f"{variant}/{source}/{tp}/{sm}"] = {
                        "n": len(rs),
                        "sigma_mean": float(np.mean(sig)) if sig else None,
                        "rho_mean": float(np.mean(rho)) if rho else None,
                        "ratio_rho_over_sigma_mean": float(np.mean([r.rho / r.sigma_max
                                                                    for r in rs
                                                                    if not np.isnan(r.rho) and r.sigma_max > 1e-12])) if rs else None,
                    }
    summary["meta"] = {
        "timestamp": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "git_commit": _git_commit(),
    }
    (processed_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot 1: ρ vs σ_max scatter for trained at h_fixed (across all inputs)
    fig, ax = plt.subplots(figsize=(9, 7))
    sm_colors = {sm: plt.cm.tab10(i) for i, sm in enumerate(SUBMODULES)}
    src_markers = {"phase1": "o", "phase1_1b": "s", "phase1_3i_square": "^"}
    for r in all_results:
        if r.label != "trained" or r.test_point != "h_fixed" or np.isnan(r.rho):
            continue
        ax.scatter(r.sigma_max, r.rho, color=sm_colors[r.submodule],
                   marker=src_markers[r.input_source], s=70, alpha=0.85,
                   edgecolors="black", linewidths=0.4)
    # Reference lines
    lim = max(0.5, max(r.sigma_max for r in all_results
                       if r.label == "trained" and r.test_point == "h_fixed" and not np.isnan(r.rho)))
    ax.plot([0, lim], [0, lim], color="grey", linestyle="--", linewidth=0.6, label="ρ = σ_max")
    ax.axhline(1.0, color="red", linestyle=":", linewidth=0.8, label="ρ = 1 (stability boundary)")
    # Legend for submodules
    for sm in SUBMODULES:
        ax.scatter([], [], color=sm_colors[sm], marker="o", s=70, label=sm,
                   edgecolors="black", linewidths=0.4)
    for src, marker in src_markers.items():
        ax.scatter([], [], color="grey", marker=marker, s=70, label=f"input: {src}",
                   edgecolors="black", linewidths=0.4)
    ax.set_xlabel("σ_max(J) (spectral norm)")
    ax.set_ylabel("ρ(J) = |λ_max| (spectral radius)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("ρ vs σ_max at h_fixed (trained)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "rho_vs_sigma_trained_hfixed.png", dpi=140)
    plt.close(fig)

    # Plot 2: third-endpoint vs phase1 + phase1_1b — ρ at h_fixed for full_stack_posfree
    fig, ax = plt.subplots(figsize=(9, 5))
    rs_groups = {
        "phase1 (lowercase)": [r for r in all_results
                               if r.label == "trained" and r.test_point == "h_fixed"
                               and r.submodule == "full_stack_posfree"
                               and r.input_source == "phase1"],
        "phase1_1b (HTML+others)": [r for r in all_results
                                     if r.label == "trained" and r.test_point == "h_fixed"
                                     and r.submodule == "full_stack_posfree"
                                     and r.input_source == "phase1_1b"],
        "phase1_3i_square (third)": [r for r in all_results
                                      if r.label == "trained" and r.test_point == "h_fixed"
                                      and r.submodule == "full_stack_posfree"
                                      and r.input_source == "phase1_3i_square"],
    }
    x = np.arange(len(rs_groups))
    width = 0.35
    for i, (label, rs) in enumerate(rs_groups.items()):
        sigmas = [r.sigma_max for r in rs if not np.isnan(r.sigma_max)]
        rhos = [r.rho for r in rs if not np.isnan(r.rho)]
        ax.bar(i - width / 2, np.mean(sigmas) if sigmas else 0, width, color="tab:blue",
               label="σ_max" if i == 0 else "", alpha=0.8)
        ax.bar(i + width / 2, np.mean(rhos) if rhos else 0, width, color="tab:orange",
               label="ρ" if i == 0 else "", alpha=0.8)
    ax.axhline(1.0, color="red", linestyle=":", linewidth=0.8, label="1.0")
    ax.set_xticks(x)
    ax.set_xticklabels(list(rs_groups.keys()))
    ax.set_ylabel("value")
    ax.set_title("Trained full_stack_posfree at h_fixed: σ_max vs ρ across input groups")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "third_endpoint_full_stack_compare.png", dpi=140)
    plt.close(fig)

    print(f"[{RUN_ID}] done. {len(all_results)} measurements")
    # Quick summary print: mean ρ vs σ_max at h_fixed
    print("\n  trained/h_fixed/<source>/<submod>: σ_max -> ρ  (ρ/σ ratio)")
    for source in ("phase1", "phase1_1b", "phase1_3i_square"):
        for sm in SUBMODULES:
            key = f"trained/{source}/h_fixed/{sm}"
            v = summary[key]
            if v["sigma_mean"] is not None and v["rho_mean"] is not None:
                ratio = v["ratio_rho_over_sigma_mean"]
                print(f"    {source:>20s} {sm:>22s}  σ={v['sigma_mean']:>7.3f}  ρ={v['rho_mean']:>7.3f}  ρ/σ={ratio:.3f}")


if __name__ == "__main__":
    main()
