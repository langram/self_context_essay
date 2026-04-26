"""Phase 1.3 Experiment J — HTML-induced structure ontology diagnosis per FPP plan v0.4 §3.3.

Four sub-stages, all sharing forward-pass infrastructure:
  J.1: long-horizon stability (4 HTML + 4 lowercase reps, max_iter=10000)
  J.2: WPE shutoff after capture (4 HTML traces, cancel-pos from step 200)
  J.3: tail convergence rate fit (post-hoc on J.1 data)
  J.4: cycle detection (post-hoc on J.1 data)

The central question: is the HTML-induced metastable structure a real fixed
point, a long-lived metastable transient, a limit cycle, or a wpe-forced
equilibrium?
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

from src.iterate import initial_hidden, load_model
from src.metrics import effective_rank
from src.posfree_iterate import iterate_hidden_cancelpos


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "phase1_3_basin_diagnosis"


# Pick 4 HTML reps and 4 lowercase reps from Phase 1.2 H factorial zoo (idx ordering)
# H idx 0..7 = natural; 8..15 = code; 16..23 = random. Even idx within each group are capital.
# Use H factorial trace dir + canonical reps:
HTML_REP_IDX = [0, 4, 8, 12]       # 2 natural-with-markup (cap+low), 2 code-with-markup (cap+low)
LOWERCASE_REP_IDX = [3, 7, 11, 15]  # 2 natural-without-markup, 2 code-without-markup
H_TRACE_DIR = "phase1_2_factorial_zoo"  # raw traces under data/raw/...

LONG_MAX_ITER = 10000
LONG_SAVE_EVERY = 100
LONG_THRESHOLD = 1e-5  # tighter than Phase 1.2 by 100x

CAPTURE_STEP = 200
SHUTOFF_EXTRA_STEPS = 1000


@dataclass
class LongTraceRecord:
    label: str  # "html_rep_<idx>" or "lc_rep_<idx>"
    factorial_idx: int                           # idx within Phase 1.2 H zoo
    text: str
    n_saved: int
    n_steps_actually_run: int
    converged: bool
    diverged: bool
    final_norm: float
    effective_rank_final: float
    deltas_summary: dict


@torch.no_grad()
def long_horizon_iterate(
    model,
    h0: torch.Tensor,
    max_iter: int,
    save_every: int,
    threshold: float,
    divergence_factor: float = 100.0,
) -> dict:
    """Like src.iterate.iterate_hidden but with stricter threshold and ample saving for long traces."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    h = h0.to(device=device, dtype=dtype)
    if h.dim() == 2:
        h = h.unsqueeze(0)
    batch, seq_len, _ = h.shape
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
    attention_mask = torch.ones(batch, seq_len, device=device, dtype=torch.long)

    init_norm = h.norm().item()
    saved = [h.detach().to("cpu", dtype=torch.float32).clone()]
    saved_steps = [0]
    norms = [init_norm]
    deltas = []
    converged = False
    diverged = False

    for step in range(max_iter):
        out = model.transformer(
            inputs_embeds=h,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        h_next = out.last_hidden_state
        prev_norm = h.norm().item()
        delta = ((h_next - h).norm() / (prev_norm + 1e-12)).item()
        next_norm = h_next.norm().item()
        deltas.append(delta)
        norms.append(next_norm)

        if (step + 1) % save_every == 0 or step == max_iter - 1:
            saved.append(h_next.detach().to("cpu", dtype=torch.float32).clone())
            saved_steps.append(step + 1)

        # Numerical sanity (tighter than phase-1 because this runs 100x longer)
        if next_norm > divergence_factor * init_norm or next_norm < init_norm / divergence_factor:
            diverged = True
            h = h_next
            break
        # Don't early-stop on convergence threshold — plan §6.2 explicitly says run full length to catch drift
        if delta < threshold:
            converged = True
        h = h_next

    trace = torch.cat(saved, dim=0)
    return {
        "trace": trace,
        "saved_steps": saved_steps,
        "deltas": deltas,
        "norms": norms,
        "converged": converged,
        "diverged": diverged,
        "n_steps": len(deltas),
        "final_hidden": h.detach().to("cpu", dtype=torch.float32).squeeze(0).clone(),
    }


@torch.no_grad()
def shutoff_iterate(
    model,
    h_capture: torch.Tensor,  # state at the capture point (after CAPTURE_STEP normal mode-A iterations)
    n_extra_steps: int,
    save_every: int = 50,
) -> dict:
    """From h_capture, run cancel-pos iteration for n_extra_steps. Save every save_every."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    h = h_capture.to(device=device, dtype=dtype)
    if h.dim() == 2:
        h = h.unsqueeze(0)
    batch, seq_len, _ = h.shape
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
    attention_mask = torch.ones(batch, seq_len, device=device, dtype=torch.long)
    wpe = model.transformer.wpe.weight[:seq_len].to(dtype=dtype)

    init_norm = h.norm().item()
    saved = [h.detach().to("cpu", dtype=torch.float32).clone()]
    saved_steps = [0]
    norms = [init_norm]
    deltas = []

    for step in range(n_extra_steps):
        h_corrected = h - wpe.unsqueeze(0)
        out = model.transformer(
            inputs_embeds=h_corrected,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        h_next = out.last_hidden_state
        prev_norm = h.norm().item()
        delta = ((h_next - h).norm() / (prev_norm + 1e-12)).item()
        deltas.append(delta)
        norms.append(h_next.norm().item())
        if (step + 1) % save_every == 0 or step == n_extra_steps - 1:
            saved.append(h_next.detach().to("cpu", dtype=torch.float32).clone())
            saved_steps.append(step + 1)
        h = h_next

    trace = torch.cat(saved, dim=0)
    return {
        "trace": trace,
        "saved_steps": saved_steps,
        "deltas": deltas,
        "norms": norms,
    }


def _load_h_factorial_input_ids(idx: int) -> tuple[torch.Tensor, str]:
    """Load input_ids and text for a given Phase 1.2 H input from its saved trace."""
    raw_path = PROJECT_ROOT / "data" / "raw" / H_TRACE_DIR / f"trace_{idx:03d}.pt"
    d = torch.load(raw_path, weights_only=False)
    return d["input_ids"], d["text"]


def _normalise_pooled(arr: np.ndarray) -> np.ndarray:
    """Mean-pool over seq dim and L2-normalise."""
    v = arr.mean(axis=-2)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def _load_reference_attractors() -> tuple[np.ndarray, np.ndarray]:
    """(lowercase_ref_pooled_unit, html_ref_pooled_unit). Both [seq_len-agnostic 768]."""
    # Lowercase: any converged Phase 1 trace's final state
    for i in range(60):
        d = torch.load(PROJECT_ROOT / "data/raw/phase1_gpt2_small/trained" / f"trace_{i:03d}.pt",
                       weights_only=False)
        if d["converged"]:
            lc = d["trace"][-1].numpy()
            break
    # HTML: a Phase 1.1 B angle-bracket trace's final state (idx 0 = first html input)
    d_html = torch.load(PROJECT_ROOT / "data/raw/phase1_1_html_outlier" / "trace_000.pt",
                        weights_only=False)
    html = d_html["trace"][-1].numpy()
    lc_pooled = _normalise_pooled(lc)
    html_pooled = _normalise_pooled(html)
    return lc_pooled, html_pooled


def fit_tail_lambda(deltas: np.ndarray, start: int, end: int) -> tuple[float, float]:
    """Fit log(delta_n) ≈ a + n·log(lambda) on the slice [start, end)."""
    if start >= end or end > deltas.size:
        return float("nan"), float("nan")
    n = np.arange(start, end)
    y = np.log(np.maximum(deltas[start:end], 1e-30))
    if not np.isfinite(y).all():
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(n, y, 1)
    lam = float(np.exp(slope))
    # R² of the linear fit
    y_pred = slope * n + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-30
    r2 = 1.0 - ss_res / ss_tot
    return lam, r2


def cycle_distances(trace: np.ndarray, ks: list[int]) -> dict[int, np.ndarray]:
    """For each k, return ‖h_n - h_{n-k}‖ along the trace (mean-pooled and Frobenius)."""
    # trace shape: [n_saved, seq_len, hidden_dim]
    n = trace.shape[0]
    out = {}
    for k in ks:
        if k >= n:
            out[k] = np.array([])
            continue
        diffs = np.linalg.norm(trace[k:] - trace[:n - k], axis=(1, 2))
        out[k] = diffs
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=LONG_MAX_ITER)
    parser.add_argument("--save-every", type=int, default=LONG_SAVE_EVERY)
    parser.add_argument("--threshold", type=float, default=LONG_THRESHOLD)
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
    lc_ref, html_ref = _load_reference_attractors()
    cos_refs = float(lc_ref @ html_ref)
    print(f"[{RUN_ID}] reference cos(lowercase, html) = {cos_refs:.4f}")

    rep_specs: list[tuple[str, int]] = (
        [("html_rep", i) for i in HTML_REP_IDX]
        + [("lc_rep", i) for i in LOWERCASE_REP_IDX]
    )

    # ===================== J.1 =====================
    print(f"[{RUN_ID}] J.1 — long-horizon traces (max_iter={args.max_iter}, save_every={args.save_every}, "
          f"threshold={args.threshold})")
    j1_results = {}
    for label, idx in tqdm(rep_specs, desc="J.1 long-trace"):
        input_ids, text = _load_h_factorial_input_ids(idx)
        h0 = initial_hidden(model, input_ids)
        out = long_horizon_iterate(
            model, h0, max_iter=args.max_iter,
            save_every=args.save_every, threshold=args.threshold,
        )
        # Per-saved-state diagnostics relative to ref attractors
        trace_np = out["trace"].numpy()  # [n_saved, seq_len, hidden_dim]
        pooled = _normalise_pooled(trace_np)  # [n_saved, 768]
        cos_to_lc = pooled @ lc_ref
        cos_to_html = pooled @ html_ref

        # Effective rank trajectory (per saved state) — only for last 20 states to save time
        eff_ranks = np.array([effective_rank(s) for s in trace_np[-20:]])
        # Norms over saved states
        per_saved_norms = np.linalg.norm(trace_np, axis=(1, 2))

        key = f"{label}_{idx}"
        j1_results[key] = {
            "label": label, "factorial_idx": idx, "text": text,
            "saved_steps": out["saved_steps"],
            "n_saved": int(trace_np.shape[0]),
            "n_steps": out["n_steps"],
            "converged": out["converged"],
            "diverged": out["diverged"],
            "final_norm": float(np.linalg.norm(trace_np[-1])),
            "final_eff_rank": float(effective_rank(trace_np[-1])),
            "tail_eff_ranks_mean": float(eff_ranks.mean()),
            "cos_to_lc_traj": cos_to_lc.tolist(),
            "cos_to_html_traj": cos_to_html.tolist(),
            "per_saved_norms": per_saved_norms.tolist(),
            "deltas_first20": out["deltas"][:20],
            "deltas_tail50": out["deltas"][-50:],
        }
        torch.save(
            {"trace": out["trace"], "saved_steps": out["saved_steps"],
             "deltas": out["deltas"], "norms": out["norms"],
             "converged": out["converged"], "diverged": out["diverged"],
             "label": label, "factorial_idx": idx, "text": text,
             "save_every": args.save_every, "threshold": args.threshold},
            raw_root / f"j1_{key}.pt",
        )

    # ===================== J.2 =====================
    print(f"[{RUN_ID}] J.2 — WPE shutoff after capture (capture step={CAPTURE_STEP}, "
          f"extra steps={SHUTOFF_EXTRA_STEPS})")
    j2_results = {}
    for label, idx in tqdm([("html_rep", i) for i in HTML_REP_IDX], desc="J.2 wpe-shutoff"):
        input_ids, text = _load_h_factorial_input_ids(idx)
        # Step 1: normal mode-A for CAPTURE_STEP iterations
        h0 = initial_hidden(model, input_ids)
        capture_out = long_horizon_iterate(
            model, h0, max_iter=CAPTURE_STEP,
            save_every=10, threshold=0.0,  # don't early-stop
        )
        h_capture = capture_out["final_hidden"]
        # Step 2: cancel-pos for SHUTOFF_EXTRA_STEPS more iterations
        shutoff_out = shutoff_iterate(model, h_capture, SHUTOFF_EXTRA_STEPS, save_every=50)

        trace_np = shutoff_out["trace"].numpy()
        pooled = _normalise_pooled(trace_np)
        cos_to_lc = pooled @ lc_ref
        cos_to_html_capture = (pooled @ _normalise_pooled(h_capture.numpy()))
        per_saved_norms = np.linalg.norm(trace_np, axis=(1, 2))

        key = f"{label}_{idx}"
        j2_results[key] = {
            "label": label, "factorial_idx": idx, "text": text,
            "capture_step": CAPTURE_STEP,
            "shutoff_steps": SHUTOFF_EXTRA_STEPS,
            "saved_steps_after_capture": shutoff_out["saved_steps"],
            "cos_to_lc_after_capture": cos_to_lc.tolist(),
            "cos_to_h_capture_after": cos_to_html_capture.tolist(),
            "per_saved_norms": per_saved_norms.tolist(),
            "deltas_first20": shutoff_out["deltas"][:20],
            "deltas_tail20": shutoff_out["deltas"][-20:],
        }
        torch.save(
            {"trace": shutoff_out["trace"], "h_capture": h_capture,
             "saved_steps_after_capture": shutoff_out["saved_steps"],
             "deltas": shutoff_out["deltas"], "norms": shutoff_out["norms"],
             "label": label, "factorial_idx": idx, "text": text,
             "capture_step": CAPTURE_STEP},
            raw_root / f"j2_{key}.pt",
        )

    # ===================== J.3 =====================
    # Tail rate fit on J.1 deltas — use the deltas already saved in raw files
    print(f"[{RUN_ID}] J.3 — tail convergence rate fitting")
    j3_results = {}
    for label, idx in rep_specs:
        key = f"{label}_{idx}"
        d = torch.load(raw_root / f"j1_{key}.pt", weights_only=False)
        deltas = np.asarray(d["deltas"], dtype=np.float64)
        # Choose tail window per plan §6.2: lowercase tail = step 5–10 (when it converges fast),
        # html tail = 100–300, but we have 10000-step traces for both, so use a wider window.
        if label == "lc_rep":
            window = (5, min(50, deltas.size))
        else:
            window = (200, min(800, deltas.size))
        lam, r2 = fit_tail_lambda(deltas, *window)
        # Also fit a "last-1000" window for both to characterise asymptotic behaviour
        far_window = (max(1, deltas.size - 1000), deltas.size)
        lam_far, r2_far = fit_tail_lambda(deltas, *far_window)
        j3_results[key] = {
            "tail_window": list(window),
            "lambda": lam,
            "r2": r2,
            "asymptotic_window": list(far_window),
            "lambda_asymptotic": lam_far,
            "r2_asymptotic": r2_far,
        }

    # ===================== J.4 =====================
    print(f"[{RUN_ID}] J.4 — cycle detection")
    j4_results = {}
    for label, idx in rep_specs:
        key = f"{label}_{idx}"
        d = torch.load(raw_root / f"j1_{key}.pt", weights_only=False)
        trace = d["trace"].numpy()
        n_saved = trace.shape[0]
        # Use the last half of the saved trace (stable region)
        tail_trace = trace[n_saved // 2:]
        ks = [1, 2, 3, 4, 8, 16]
        dists = cycle_distances(tail_trace, ks)
        # For each k, take mean of distances (only over indices that actually exist)
        mean_dists = {k: float(v.mean()) if v.size > 0 else None for k, v in dists.items()}
        # Cycle signal: k where mean_dist[k] < mean_dist[1] meaningfully
        baseline = mean_dists[1]
        cycle_signal = None
        if baseline is not None and baseline > 0:
            for k in ks[1:]:
                if mean_dists[k] is not None and mean_dists[k] < 0.5 * baseline:
                    cycle_signal = k
                    break
        j4_results[key] = {
            "k_distances_mean_in_tail": mean_dists,
            "cycle_signal_k": cycle_signal,
        }

    # ===================== Persist & summarise =====================
    summary = {
        "timestamp": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip(),
        "params": {
            "max_iter": args.max_iter, "save_every": args.save_every, "threshold": args.threshold,
            "capture_step": CAPTURE_STEP, "shutoff_extra_steps": SHUTOFF_EXTRA_STEPS,
            "rep_specs": rep_specs,
        },
        "j1": j1_results,
        "j2": j2_results,
        "j3": j3_results,
        "j4": j4_results,
    }
    (processed_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot J.1 — cos to references over saved steps
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, ref_label, ref_key in zip(axes, ("cos to lowercase ref", "cos to html ref"),
                                       ("cos_to_lc_traj", "cos_to_html_traj")):
        for key, r in j1_results.items():
            steps = r["saved_steps"]
            traj = r[ref_key]
            color = "tab:red" if r["label"] == "html_rep" else "tab:blue"
            ax.plot(steps, traj, color=color, alpha=0.7, label=key)
        ax.set_xlabel("iteration step (long trace)")
        ax.set_ylabel(ref_label)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_title(ref_label)
        ax.set_ylim(-0.1, 1.05)
        ax.legend(fontsize=7)
    fig.suptitle("J.1 — long-horizon trajectories vs reference attractors")
    fig.tight_layout()
    fig.savefig(fig_dir / "j1_cos_trajectories.png", dpi=140)
    plt.close(fig)

    # Plot J.2 — wpe shutoff
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, ref_label, ref_key in zip(axes, ("cos to lowercase ref", "cos to h_capture"),
                                       ("cos_to_lc_after_capture", "cos_to_h_capture_after")):
        for key, r in j2_results.items():
            steps = [s + CAPTURE_STEP for s in r["saved_steps_after_capture"]]
            traj = r[ref_key]
            ax.plot(steps, traj, color="tab:purple", alpha=0.7, label=key)
        ax.axvline(CAPTURE_STEP, color="black", linestyle=":", label="wpe shutoff")
        ax.set_xlabel("global iteration step")
        ax.set_ylabel(ref_label)
        ax.set_title(ref_label)
        ax.set_ylim(-0.1, 1.05)
        ax.legend(fontsize=7)
    fig.suptitle("J.2 — WPE shutoff after capture (capture at step 200, then cancel-pos)")
    fig.tight_layout()
    fig.savefig(fig_dir / "j2_wpe_shutoff.png", dpi=140)
    plt.close(fig)

    # Plot J.3 — tail rates as bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    keys = list(j3_results.keys())
    lams = [j3_results[k]["lambda"] for k in keys]
    lams_far = [j3_results[k]["lambda_asymptotic"] for k in keys]
    x = np.arange(len(keys))
    ax.bar(x - 0.2, lams, width=0.4, label="tail window", color="tab:cyan")
    ax.bar(x + 0.2, lams_far, width=0.4, label="asymptotic last-1000", color="tab:olive")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, label="lambda=1")
    ax.set_ylabel("fitted lambda (per-step contraction rate)")
    ax.set_title("J.3 — tail convergence rate fit (lambda < 1 means contracting; closer to 1 means weak)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "j3_tail_rates.png", dpi=140)
    plt.close(fig)

    # Print quick summary
    print(f"\n[{RUN_ID}] J.1 summary:")
    for key, r in j1_results.items():
        print(f"  {key}: n_saved={r['n_saved']}, n_steps={r['n_steps']}, "
              f"final_norm={r['final_norm']:.0f}, eff_rank={r['final_eff_rank']:.2f}, "
              f"final cos_lc={r['cos_to_lc_traj'][-1]:+.3f}, "
              f"final cos_html={r['cos_to_html_traj'][-1]:+.3f}")

    print(f"\n[{RUN_ID}] J.2 summary (cos to h_capture and to lc):")
    for key, r in j2_results.items():
        cos_lc_final = r["cos_to_lc_after_capture"][-1]
        cos_capture_final = r["cos_to_h_capture_after"][-1]
        cos_lc_first = r["cos_to_lc_after_capture"][0]
        cos_capture_first = r["cos_to_h_capture_after"][0]
        print(f"  {key}: at_capture: cos_lc={cos_lc_first:+.3f}, cos_capture={cos_capture_first:+.3f}; "
              f"after {SHUTOFF_EXTRA_STEPS} steps: cos_lc={cos_lc_final:+.3f}, cos_capture={cos_capture_final:+.3f}")

    print(f"\n[{RUN_ID}] J.3 summary (lambda values):")
    for key, r in j3_results.items():
        print(f"  {key}: tail_lambda={r['lambda']:.4f} (R2={r['r2']:.3f}), "
              f"asym_lambda={r['lambda_asymptotic']:.4f} (R2={r['r2_asymptotic']:.3f})")

    print(f"\n[{RUN_ID}] J.4 summary (cycle signals):")
    for key, r in j4_results.items():
        print(f"  {key}: k_dists={r['k_distances_mean_in_tail']}, cycle_k={r['cycle_signal_k']}")


if __name__ == "__main__":
    main()
