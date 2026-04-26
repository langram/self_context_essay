"""Metrics on hidden-state traces and fixed-point states per FPP plan §3.5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


@dataclass
class TraceMetrics:
    converged: bool
    diverged: bool
    n_steps: int
    deltas: np.ndarray
    norms: np.ndarray
    final_norm: float
    per_token_norms: np.ndarray
    effective_rank: float


@dataclass
class VocabProjection:
    top5_token_ids: np.ndarray         # [seq_len, 5]
    top5_token_strs: list[list[str]]    # [seq_len][5]
    top5_logits: np.ndarray            # [seq_len, 5]


def effective_rank(matrix: np.ndarray, eps: float = 1e-12) -> float:
    """Roy & Vetterli effective rank: exp of Shannon entropy of normalised singular values."""
    sv = np.linalg.svd(matrix, compute_uv=False)
    sv = sv[sv > eps]
    if sv.size == 0:
        return 0.0
    p = sv / sv.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def per_token_norms(hidden: np.ndarray) -> np.ndarray:
    """L2 norm of each token's hidden vector. hidden: [seq_len, hidden_dim]."""
    return np.linalg.norm(hidden, axis=-1)


def trace_metrics(
    deltas: list[float],
    norms: list[float],
    converged: bool,
    diverged: bool,
    final_hidden: torch.Tensor | np.ndarray,
) -> TraceMetrics:
    h = final_hidden.numpy() if isinstance(final_hidden, torch.Tensor) else final_hidden
    return TraceMetrics(
        converged=converged,
        diverged=diverged,
        n_steps=len(deltas),
        deltas=np.asarray(deltas, dtype=np.float32),
        norms=np.asarray(norms, dtype=np.float32),
        final_norm=float(np.linalg.norm(h)),
        per_token_norms=per_token_norms(h),
        effective_rank=effective_rank(h),
    )


@torch.no_grad()
def project_to_vocab(
    final_hidden: torch.Tensor,
    model,
    tokenizer,
    top_k: int = 5,
) -> VocabProjection:
    """Push C* through the LM head's unembedding matrix and read off top-k tokens per position."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    h = final_hidden.to(device=device, dtype=dtype)
    if h.dim() == 2:
        h = h.unsqueeze(0)
    # GPT-2 applies a final LayerNorm before lm_head; mirror that.
    h_normed = model.transformer.ln_f(h)
    logits = model.lm_head(h_normed).squeeze(0)  # [seq_len, vocab]
    top_vals, top_ids = torch.topk(logits, k=top_k, dim=-1)
    top_ids_np = top_ids.cpu().numpy()
    top_vals_np = top_vals.to(torch.float32).cpu().numpy()
    top_strs = [[tokenizer.decode([int(i)]) for i in row] for row in top_ids_np]
    return VocabProjection(top5_token_ids=top_ids_np, top5_token_strs=top_strs, top5_logits=top_vals_np)


def cosine_similarity_matrix(states: np.ndarray) -> np.ndarray:
    """states: [n, ...] flattened to [n, d]; returns n×n cosine sim."""
    flat = states.reshape(states.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    safe = np.maximum(norms, 1e-12)
    normed = flat / safe
    return normed @ normed.T


def hierarchical_linkage(similarity: np.ndarray, method: str = "average") -> np.ndarray:
    """Convert a similarity matrix to scipy linkage for dendrogram plotting."""
    distance = np.clip(1.0 - similarity, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0
    return linkage(squareform(distance, checks=False), method=method)
