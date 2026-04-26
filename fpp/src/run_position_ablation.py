"""Phase 1.1 Experiment C runner per FPP plan v0.2 §3.4.

Runs both C1 (cancel-pos) and C2 (posfree) variants on the original Phase 1
60-input zoo, on both trained and random-init models, and compares final
states to the Phase 1 mode-A baseline.
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

from src.inputs import build_input_zoo
from src.iterate import initial_hidden, load_model
from src.metrics import cosine_similarity_matrix, project_to_vocab, trace_metrics
from src.posfree_iterate import iterate_hidden_cancelpos, iterate_hidden_posfree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "phase1_1_position_ablation"


@dataclass
class AblationRecord:
    idx: int
    category: str
    text: str
    variant: str
    model_kind: str
    converged: bool
    diverged: bool
    n_steps: int
    final_norm: float
    effective_rank: float
    cos_to_modeA_baseline: float


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def load_phase1_finals(model_kind: str) -> tuple[np.ndarray, list[str]]:
    """Load all 60 final hidden states + categories from Phase 1, in idx order."""
    raw = PROJECT_ROOT / "data" / "raw" / "phase1_gpt2_small" / model_kind
    finals = []
    cats = []
    for i in range(60):
        d = torch.load(raw / f"trace_{i:03d}.pt", weights_only=False)
        finals.append(d["trace"][-1].numpy())
        cats.append(d["category"])
    return np.stack(finals, axis=0), cats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=1e-3)
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

    categories = ["random_tokens", "grammatical_random_semantics", "common",
                  "ambiguous", "nonsense_grammatical", "structured"]

    all_records: list[AblationRecord] = []
    finals_by_kind_variant: dict[tuple[str, str], np.ndarray] = {}

    t0 = time.time()
    for model_kind in ("trained", "random"):
        print(f"[{RUN_ID}] loading {model_kind} model")
        model, tokenizer = load_model("gpt2", "float32", "cuda",
                                      random_init=(model_kind == "random"), seed=args.seed)
        items = build_input_zoo(tokenizer, categories, 10, 32, args.seed)

        # Mode-A baseline finals from Phase 1, for cosine reference
        modeA_finals, modeA_cats = load_phase1_finals(model_kind)
        modeA_vec = modeA_finals.reshape(60, -1)
        modeA_norm = modeA_vec / np.maximum(np.linalg.norm(modeA_vec, axis=1, keepdims=True), 1e-12)

        for variant in ("C1_cancelpos", "C2_posfree"):
            iterator = iterate_hidden_cancelpos if variant == "C1_cancelpos" else iterate_hidden_posfree
            variant_dir = raw_root / model_kind / variant
            variant_dir.mkdir(parents=True, exist_ok=True)

            finals = []
            for item in tqdm(items, desc=f"{model_kind}/{variant}"):
                h0 = initial_hidden(model, item.input_ids)
                res = iterator(model, h0, max_iter=args.max_iter,
                               convergence_threshold=args.threshold, divergence_factor=100.0)
                m = trace_metrics(res.deltas, res.norms, res.converged, res.diverged, res.final_hidden)

                fhid = res.final_hidden.numpy()
                fvec = fhid.reshape(-1)
                fvec_n = fvec / max(np.linalg.norm(fvec), 1e-12)
                cos_to_baseline = float(fvec_n @ modeA_norm[item.idx])

                all_records.append(AblationRecord(
                    idx=item.idx,
                    category=item.category,
                    text=item.text,
                    variant=variant,
                    model_kind=model_kind,
                    converged=m.converged,
                    diverged=m.diverged,
                    n_steps=m.n_steps,
                    final_norm=m.final_norm,
                    effective_rank=m.effective_rank,
                    cos_to_modeA_baseline=cos_to_baseline,
                ))
                finals.append(fhid)
                torch.save(
                    {"trace": res.trace, "deltas": res.deltas, "norms": res.norms,
                     "converged": res.converged, "diverged": res.diverged,
                     "category": item.category, "text": item.text,
                     "variant": variant, "model_kind": model_kind, "input_ids": item.input_ids},
                    variant_dir / f"trace_{item.idx:03d}.pt",
                )
            finals_by_kind_variant[(model_kind, variant)] = np.stack(finals, axis=0)

        # Free GPU before loading the random twin
        del model
        torch.cuda.empty_cache()

    elapsed = time.time() - t0

    # Persist records
    (processed_dir / "records.json").write_text(
        json.dumps([asdict(r) for r in all_records], indent=2), encoding="utf-8"
    )

    # Persist similarity matrices for each (model_kind, variant)
    for (kind, variant), finals in finals_by_kind_variant.items():
        sim = cosine_similarity_matrix(finals)
        np.savez(processed_dir / f"similarity_{kind}_{variant}.npz", sim=sim)

    # Aggregates
    summary: dict = {
        "timestamp": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "git_commit": _git_commit(),
        "elapsed_seconds": elapsed,
        "by_kind_variant": {},
    }
    for kind in ("trained", "random"):
        for variant in ("C1_cancelpos", "C2_posfree"):
            recs = [r for r in all_records if r.model_kind == kind and r.variant == variant]
            n = len(recs)
            conv = sum(1 for r in recs if r.converged)
            steps_conv = [r.n_steps for r in recs if r.converged]
            cos_to_baseline = [r.cos_to_modeA_baseline for r in recs]
            sim = np.load(processed_dir / f"similarity_{kind}_{variant}.npz")["sim"]
            offdiag = sim[~np.eye(n, dtype=bool)]
            summary["by_kind_variant"][f"{kind}/{variant}"] = {
                "n": n,
                "pct_converged": 100.0 * conv / n,
                "mean_steps_converged": float(np.mean(steps_conv)) if steps_conv else None,
                "mean_eff_rank": float(np.mean([r.effective_rank for r in recs])),
                "mean_final_norm": float(np.mean([r.final_norm for r in recs])),
                "mean_cos_to_modeA_baseline": float(np.mean(cos_to_baseline)),
                "min_cos_to_modeA_baseline": float(np.min(cos_to_baseline)),
                "mean_offdiag_pairwise_cos": float(offdiag.mean()),
                "min_offdiag_pairwise_cos": float(offdiag.min()),
            }
    (processed_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Cross-variant agreement check (C1 vs C2 finals): they should be near-identical if both implementations
    # implement the same "blocks-only fixed point"
    for kind in ("trained", "random"):
        c1 = finals_by_kind_variant[(kind, "C1_cancelpos")]
        c2 = finals_by_kind_variant[(kind, "C2_posfree")]
        c1f = c1.reshape(60, -1); c2f = c2.reshape(60, -1)
        c1f /= np.linalg.norm(c1f, axis=1, keepdims=True) + 1e-12
        c2f /= np.linalg.norm(c2f, axis=1, keepdims=True) + 1e-12
        per_input_cos = (c1f * c2f).sum(axis=1)
        summary["by_kind_variant"].setdefault("c1_vs_c2_agreement", {})[kind] = {
            "mean_per_input_cos": float(per_input_cos.mean()),
            "min_per_input_cos": float(per_input_cos.min()),
        }
    (processed_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot: per-input cos to mode-A baseline, four bars per category
    fig, ax = plt.subplots(figsize=(12, 5))
    cats = sorted(set(r.category for r in all_records))
    x = np.arange(len(all_records) // 4)  # 60 indexes
    width = 0.2
    for i, key in enumerate(("trained/C1_cancelpos", "trained/C2_posfree",
                             "random/C1_cancelpos", "random/C2_posfree")):
        kind, variant = key.split("/")
        recs = [r for r in all_records if r.model_kind == kind and r.variant == variant]
        recs.sort(key=lambda r: r.idx)
        vals = [r.cos_to_modeA_baseline for r in recs]
        ax.bar(x + (i - 1.5) * width, vals, width, label=key)
    ax.axhline(0.99, color="grey", linestyle="--", linewidth=0.8, label="0.99")
    ax.set_xlabel("input idx (sorted)")
    ax.set_ylabel("cos(ablation C*, mode-A C*)")
    ax.set_title("Position-embedding ablation: per-input cosine to original mode-A fixed point")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(-0.1, 1.05)
    fig.tight_layout()
    fig.savefig(fig_dir / "cos_to_modeA_baseline_per_input.png", dpi=140)
    plt.close(fig)

    # Plot: similarity heatmaps for trained variants
    for kind in ("trained", "random"):
        for variant in ("C1_cancelpos", "C2_posfree"):
            sim = np.load(processed_dir / f"similarity_{kind}_{variant}.npz")["sim"]
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
            ax.set_title(f"C* pairwise cos similarity — {kind} / {variant}")
            ax.set_xlabel("input idx"); ax.set_ylabel("input idx")
            plt.colorbar(im, ax=ax, label="cos sim")
            fig.tight_layout()
            fig.savefig(fig_dir / f"similarity_{kind}_{variant}.png", dpi=140)
            plt.close(fig)

    print(f"[{RUN_ID}] done in {elapsed:.1f}s")
    print(f"  see summary: {processed_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
