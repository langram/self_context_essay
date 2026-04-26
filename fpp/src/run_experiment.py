"""Phase 1 orchestrator per FPP plan §3.

Loads a config, runs hidden-state iteration on the input zoo for the trained model
(and optionally a randomly-initialised twin), saves raw traces and per-trace metrics,
runs robustness checks, and emits the §5.2 figures plus a stub markdown report.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.inputs import InputItem, build_input_zoo
from src.iterate import IterationResult, initial_hidden, iterate_hidden, load_model
from src.metrics import (
    cosine_similarity_matrix,
    hierarchical_linkage,
    project_to_vocab,
    trace_metrics,
)
from src.visualize import (
    plot_convergence_histograms,
    plot_dendrogram,
    plot_similarity_heatmap,
    plot_trace_examples,
    write_top5_token_table,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TraceRecord:
    idx: int
    category: str
    text: str
    converged: bool
    diverged: bool
    n_steps: int
    final_norm: float
    effective_rank: float
    per_token_norms: list[float]
    deltas: list[float]
    norms: list[float]


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def _run_one(
    model,
    input_ids: torch.Tensor,
    cfg: dict,
) -> tuple[IterationResult, torch.Tensor]:
    h0 = initial_hidden(model, input_ids)
    res = iterate_hidden(
        model,
        h0,
        max_iter=cfg["max_iter"],
        convergence_threshold=cfg["convergence_threshold"],
        divergence_factor=cfg["divergence_factor"],
    )
    return res, h0.detach().to("cpu", dtype=torch.float32).squeeze(0)


def _run_set(
    label: str,
    model,
    tokenizer,
    items: list[InputItem],
    cfg: dict,
    raw_root: Path,
) -> tuple[list[TraceRecord], np.ndarray, list[dict]]:
    raw_dir = raw_root / label
    raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[TraceRecord] = []
    final_states: list[np.ndarray] = []
    top5_table: list[dict] = []

    for item in tqdm(items, desc=f"iterate[{label}]"):
        res, _h0 = _run_one(model, item.input_ids, cfg)
        m = trace_metrics(res.deltas, res.norms, res.converged, res.diverged, res.final_hidden)
        records.append(
            TraceRecord(
                idx=item.idx,
                category=item.category,
                text=item.text,
                converged=m.converged,
                diverged=m.diverged,
                n_steps=m.n_steps,
                final_norm=m.final_norm,
                effective_rank=m.effective_rank,
                per_token_norms=m.per_token_norms.tolist(),
                deltas=m.deltas.tolist(),
                norms=m.norms.tolist(),
            )
        )
        final_states.append(res.final_hidden.numpy())
        torch.save(
            {"trace": res.trace, "deltas": res.deltas, "norms": res.norms,
             "converged": res.converged, "diverged": res.diverged,
             "category": item.category, "text": item.text, "input_ids": item.input_ids},
            raw_dir / f"trace_{item.idx:03d}.pt",
        )
        proj = project_to_vocab(res.final_hidden, model, tokenizer, top_k=5)
        top5_table.append({
            "idx": item.idx,
            "category": item.category,
            "text": item.text,
            "top5_per_position": proj.top5_token_strs,
        })

    return records, np.stack(final_states, axis=0), top5_table


def _per_category_arrays(records: list[TraceRecord]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    n_steps_by_cat: dict[str, list[int]] = {}
    conv_by_cat: dict[str, list[bool]] = {}
    for r in records:
        n_steps_by_cat.setdefault(r.category, []).append(r.n_steps)
        conv_by_cat.setdefault(r.category, []).append(r.converged)
    return (
        {k: np.asarray(v) for k, v in n_steps_by_cat.items()},
        {k: np.asarray(v) for k, v in conv_by_cat.items()},
    )


def _robustness_perturbation(
    model,
    items: list[InputItem],
    cfg: dict,
    n_inputs: int,
    repeats: int,
    sigma_rel: float,
    out_path: Path,
) -> None:
    chosen = items[: min(n_inputs, len(items))]
    results = []
    for item in tqdm(chosen, desc="robust:perturb"):
        h0 = initial_hidden(model, item.input_ids)
        h0_norm = h0.norm().item()
        sigma = sigma_rel * h0_norm / (h0.numel() ** 0.5)
        c_stars = []
        for r in range(repeats):
            torch.manual_seed(1000 * item.idx + r)
            noise = torch.randn_like(h0) * sigma
            res = iterate_hidden(
                model,
                h0 + noise,
                max_iter=cfg["max_iter"],
                convergence_threshold=cfg["convergence_threshold"],
                divergence_factor=cfg["divergence_factor"],
            )
            c_stars.append(res.final_hidden.numpy().reshape(-1))
        c_arr = np.stack(c_stars, axis=0)
        norms = np.linalg.norm(c_arr, axis=1, keepdims=True) + 1e-12
        sim = (c_arr / norms) @ (c_arr / norms).T
        offdiag = sim[~np.eye(repeats, dtype=bool)]
        results.append({
            "idx": item.idx,
            "category": item.category,
            "text": item.text,
            "mean_offdiag_cosine": float(offdiag.mean()),
            "min_offdiag_cosine": float(offdiag.min()),
            "max_offdiag_cosine": float(offdiag.max()),
        })
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def _robustness_long_trace(
    model,
    items: list[InputItem],
    cfg: dict,
    n_inputs: int,
    long_max_iter: int,
    out_path: Path,
) -> None:
    chosen = items[: min(n_inputs, len(items))]
    results = []
    for item in tqdm(chosen, desc="robust:long"):
        h0 = initial_hidden(model, item.input_ids)
        res = iterate_hidden(
            model,
            h0,
            max_iter=long_max_iter,
            convergence_threshold=cfg["convergence_threshold"],
            divergence_factor=cfg["divergence_factor"],
        )
        results.append({
            "idx": item.idx,
            "category": item.category,
            "text": item.text,
            "n_steps": res.n_steps,
            "converged": res.converged,
            "diverged": res.diverged,
            "deltas_tail": res.deltas[-50:],
            "norms_tail": res.norms[-50:],
        })
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def _write_stub_report(run_dir: Path, cfg: dict, summary: dict) -> None:
    md = [
        f"# FPP run report — `{cfg['run_id']}`",
        "",
        f"- timestamp: {summary['timestamp']}",
        f"- git commit: `{summary['git_commit']}`",
        f"- model: `{cfg['model_name']}` (dtype={cfg['dtype']}, device={cfg['device']})",
        f"- max_iter: {cfg['max_iter']}, convergence_threshold: {cfg['convergence_threshold']}",
        f"- inputs: {len(cfg['input_categories'])} categories × {cfg['inputs_per_category']}",
        "",
        "## Aggregate convergence",
        "",
        "| variant | n_inputs | %converged | %diverged | mean steps (converged) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for variant, agg in summary["aggregates"].items():
        md.append(
            f"| {variant} | {agg['n']} | {agg['pct_converged']:.1f}% | {agg['pct_diverged']:.1f}% | "
            f"{agg['mean_steps_converged']:.1f} |"
        )
    md += [
        "",
        "## Pointers",
        "",
        "- raw per-trace tensors: `data/raw/<run_id>/<variant>/trace_*.pt`",
        "- per-trace metrics JSON: `data/processed/<run_id>/records_<variant>.json`",
        "- figures: `outputs/figures/<run_id>/`",
        "- top-5 vocab projection: `outputs/reports/<run_id>/top5_<variant>.md`",
        "",
        "## Open questions",
        "",
        "- Q1 existence — see convergence histograms.",
        "- Q2 diversity — see C* similarity heatmap and dendrogram.",
        "- Q3 trained-vs-random — compare both variants in the histograms / similarity grids.",
        "- Q4 semantic correspondence — see top-5 vocab projection.",
        "",
        "Fill in the §5.3 narrative answers after inspecting the figures.",
    ]
    (run_dir / "report.md").write_text("\n".join(md), encoding="utf-8")


def _aggregate(records: list[TraceRecord]) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0, "pct_converged": 0.0, "pct_diverged": 0.0, "mean_steps_converged": 0.0}
    converged = [r for r in records if r.converged]
    diverged = [r for r in records if r.diverged]
    mean_steps = float(np.mean([r.n_steps for r in converged])) if converged else 0.0
    return {
        "n": n,
        "pct_converged": 100.0 * len(converged) / n,
        "pct_diverged": 100.0 * len(diverged) / n,
        "mean_steps_converged": mean_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_id = cfg["run_id"]
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    git_commit = _git_commit()

    data_root = PROJECT_ROOT / "data"
    fig_dir = PROJECT_ROOT / "outputs" / "figures" / run_id
    report_dir = PROJECT_ROOT / "outputs" / "reports" / run_id
    processed_dir = data_root / "processed" / run_id
    raw_root = data_root / "raw" / run_id
    for d in (fig_dir, report_dir, processed_dir, raw_root):
        d.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True

    print(f"[{run_id}] loading trained model {cfg['model_name']} ({cfg['dtype']}) on {cfg['device']}")
    trained_model, tokenizer = load_model(
        cfg["model_name"], cfg["dtype"], cfg["device"], random_init=False, seed=cfg["seed"],
    )

    items = build_input_zoo(
        tokenizer,
        cfg["input_categories"],
        cfg["inputs_per_category"],
        cfg["seq_len"],
        cfg["seed"],
    )
    print(f"[{run_id}] built {len(items)} inputs across {len(cfg['input_categories'])} categories")

    variants_summary: dict[str, dict] = {}
    final_states_by_variant: dict[str, np.ndarray] = {}
    labels_by_variant: dict[str, list[str]] = {}

    t0 = time.time()
    trained_records, trained_finals, trained_top5 = _run_set(
        "trained", trained_model, tokenizer, items, cfg, raw_root,
    )
    final_states_by_variant["trained"] = trained_finals
    labels_by_variant["trained"] = [it.category for it in items]
    variants_summary["trained"] = _aggregate(trained_records)
    (processed_dir / "records_trained.json").write_text(
        json.dumps([asdict(r) for r in trained_records], indent=2), encoding="utf-8"
    )
    write_top5_token_table(trained_top5, report_dir / "top5_trained.md")

    if cfg.get("include_random_init", False):
        print(f"[{run_id}] loading random-init twin")
        del trained_model
        torch.cuda.empty_cache()
        random_model, _ = load_model(
            cfg["model_name"], cfg["dtype"], cfg["device"], random_init=True, seed=cfg["seed"],
        )
        random_records, random_finals, random_top5 = _run_set(
            "random", random_model, tokenizer, items, cfg, raw_root,
        )
        final_states_by_variant["random"] = random_finals
        labels_by_variant["random"] = [it.category for it in items]
        variants_summary["random"] = _aggregate(random_records)
        (processed_dir / "records_random.json").write_text(
            json.dumps([asdict(r) for r in random_records], indent=2), encoding="utf-8"
        )
        write_top5_token_table(random_top5, report_dir / "top5_random.md")
        # Reload the trained model for robustness checks (only on trained variant).
        del random_model
        torch.cuda.empty_cache()
        trained_model, _ = load_model(
            cfg["model_name"], cfg["dtype"], cfg["device"], random_init=False, seed=cfg["seed"],
        )

    rb_cfg = cfg.get("robustness", {})
    if rb_cfg.get("enabled", False):
        print(f"[{run_id}] robustness: perturbation + long-trace")
        _robustness_perturbation(
            trained_model,
            items,
            cfg,
            n_inputs=rb_cfg["perturbation_n_inputs"],
            repeats=rb_cfg["perturbation_repeats"],
            sigma_rel=rb_cfg["perturbation_sigma_rel"],
            out_path=processed_dir / "robustness_perturbation.json",
        )
        _robustness_long_trace(
            trained_model,
            items,
            cfg,
            n_inputs=rb_cfg["long_trace_n_inputs"],
            long_max_iter=rb_cfg["long_trace_max_iter"],
            out_path=processed_dir / "robustness_long_trace.json",
        )

    print(f"[{run_id}] generating figures")
    for variant, finals in final_states_by_variant.items():
        records = trained_records if variant == "trained" else random_records
        n_steps_by_cat, conv_by_cat = _per_category_arrays(records)
        plot_convergence_histograms(
            n_steps_by_cat,
            conv_by_cat,
            fig_dir / f"convergence_steps_{variant}.png",
            title_suffix=f"({variant})",
            max_iter=cfg["max_iter"],
        )
        cat_to_first_idx: dict[str, int] = {}
        for r in records:
            cat_to_first_idx.setdefault(r.category, r.idx)
        ex_deltas = {f"{r.category} (#{r.idx})": np.asarray(r.deltas)
                     for r in records if r.idx in cat_to_first_idx.values()}
        plot_trace_examples(
            ex_deltas,
            fig_dir / f"trace_examples_{variant}.png",
            title=f"Per-step relative delta — {variant}",
            threshold=cfg["convergence_threshold"],
        )
        if finals.shape[0] >= 2:
            sim = cosine_similarity_matrix(finals)
            np.savez(processed_dir / f"similarity_{variant}.npz",
                     sim=sim, labels=np.array(labels_by_variant[variant]))
            plot_similarity_heatmap(
                sim,
                labels_by_variant[variant],
                fig_dir / f"similarity_{variant}.png",
                title=f"C* cosine similarity — {variant}",
            )
            link = hierarchical_linkage(sim)
            leaf_labels = [f"{cat[:6]}#{i}" for i, cat in enumerate(labels_by_variant[variant])]
            plot_dendrogram(
                link,
                leaf_labels,
                fig_dir / f"dendrogram_{variant}.png",
                title=f"Hierarchical clustering of C* — {variant}",
            )

    elapsed = time.time() - t0
    summary = {
        "timestamp": timestamp,
        "git_commit": git_commit,
        "elapsed_seconds": elapsed,
        "aggregates": variants_summary,
        "config": cfg,
    }
    (processed_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_stub_report(report_dir, cfg, summary)
    print(f"[{run_id}] done in {elapsed:.1f}s. Report: {report_dir / 'report.md'}")


if __name__ == "__main__":
    main()
