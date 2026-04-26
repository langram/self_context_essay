"""Plotting per FPP plan §5.2 — convergence histograms, traces, similarity heatmaps, dendrograms."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram


def plot_convergence_histograms(
    n_steps_by_category: dict[str, np.ndarray],
    converged_by_category: dict[str, np.ndarray],
    out_path: Path,
    title_suffix: str = "",
    max_iter: int = 100,
) -> None:
    cats = list(n_steps_by_category.keys())
    n = len(cats)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.2), squeeze=False)
    for ax_idx, cat in enumerate(cats):
        ax = axes[ax_idx // cols][ax_idx % cols]
        steps = n_steps_by_category[cat]
        conv = converged_by_category[cat]
        if steps.size == 0:
            ax.set_visible(False)
            continue
        bins = np.linspace(0, max_iter, 21)
        ax.hist(steps[conv.astype(bool)], bins=bins, color="tab:green", alpha=0.7, label="converged")
        ax.hist(steps[~conv.astype(bool)], bins=bins, color="tab:red", alpha=0.5, label="non-converged")
        ax.set_title(cat, fontsize=10)
        ax.set_xlabel("steps")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)
    fig.suptitle(f"Convergence steps {title_suffix}".strip(), fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_trace_examples(
    deltas_by_label: dict[str, np.ndarray],
    out_path: Path,
    title: str = "",
    threshold: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, deltas in deltas_by_label.items():
        if deltas.size == 0:
            continue
        ax.plot(np.arange(1, deltas.size + 1), deltas, label=label, linewidth=1.4)
    if threshold is not None:
        ax.axhline(threshold, color="grey", linestyle="--", linewidth=0.8, label=f"threshold={threshold}")
    ax.set_yscale("log")
    ax.set_xlabel("iteration step")
    ax.set_ylabel(r"$\|h_{n+1} - h_n\| / \|h_n\|$")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_similarity_heatmap(
    sim: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        sim,
        cmap="vlag",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        ax=ax,
        cbar_kws={"label": "cosine sim"},
        xticklabels=False,
        yticklabels=False,
    )
    boundaries = []
    last_label = None
    for i, lab in enumerate(labels):
        if lab != last_label:
            boundaries.append(i)
            last_label = lab
    boundaries.append(len(labels))
    for b in boundaries[1:-1]:
        ax.axhline(b, color="black", linewidth=0.6)
        ax.axvline(b, color="black", linewidth=0.6)
    centers = [(boundaries[i] + boundaries[i + 1]) / 2.0 for i in range(len(boundaries) - 1)]
    cat_names = [labels[boundaries[i]] for i in range(len(boundaries) - 1)]
    ax.set_xticks(centers)
    ax.set_xticklabels(cat_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(centers)
    ax.set_yticklabels(cat_names, rotation=0, fontsize=9)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=80, leaf_font_size=7, ax=ax)
    ax.set_ylabel("cosine distance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_top5_token_table(
    items_top5: list[dict],
    out_path: Path,
) -> None:
    """items_top5: each dict has keys 'idx', 'category', 'text', 'top5_per_position' (list of list[str])."""
    lines = ["# C* → vocab projection (top-5 per token position)\n"]
    for it in items_top5:
        lines.append(f"## {it['idx']:02d} — `{it['category']}`")
        lines.append(f"> input: `{it['text']}`")
        lines.append("")
        lines.append("| pos | top-1 | top-2 | top-3 | top-4 | top-5 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for pos, row in enumerate(it["top5_per_position"]):
            cells = [f"`{tok}`" for tok in row]
            lines.append(f"| {pos} | " + " | ".join(cells) + " |")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
