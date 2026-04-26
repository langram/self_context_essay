"""Phase 1.1 Experiment C — position-embedding ablation per FPP plan v0.2 §3.4.

Two variants:
  - C1 (cancel-pos): subtract wpe before each iteration so the model's standard
    inputs_embeds path re-adds it once and only once.
  - C2 (posfree): manually run blocks + ln_f, skipping wpe entirely after h_0.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.iterate import IterationResult


@torch.no_grad()
def iterate_hidden_cancelpos(
    model,
    h0: torch.Tensor,
    max_iter: int = 100,
    convergence_threshold: float = 1e-3,
    divergence_factor: float = 100.0,
    save_every: int = 1,
) -> IterationResult:
    """Variant C1: subtract wpe[:seq_len] from h_n before feeding to the standard forward.

    The model's GPT2Model forward will then add wpe back, netting to a single wpe injection
    overall (same as h_0's first forward). Subsequent iterations are equivalent to running
    blocks(h_n) without an extra position bias.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    h = h0.to(device=device, dtype=dtype)
    if h.dim() == 2:
        h = h.unsqueeze(0)

    batch, seq_len, _ = h.shape
    wpe = model.transformer.wpe.weight[:seq_len].to(dtype=dtype)  # [seq_len, hidden_dim]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
    attention_mask = torch.ones(batch, seq_len, device=device, dtype=torch.long)

    init_norm = h.norm().item()
    saved = [h.detach().to("cpu", dtype=torch.float32).clone()]
    norms = [init_norm]
    deltas: list[float] = []
    converged = False
    diverged = False

    for step in range(max_iter):
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
        next_norm = h_next.norm().item()
        deltas.append(delta)
        norms.append(next_norm)
        if (step + 1) % save_every == 0 or step == max_iter - 1:
            saved.append(h_next.detach().to("cpu", dtype=torch.float32).clone())

        if next_norm > divergence_factor * init_norm or next_norm < init_norm / divergence_factor:
            diverged = True
            h = h_next
            break
        if delta < convergence_threshold:
            converged = True
            h = h_next
            break
        h = h_next

    trace = torch.cat(saved, dim=0)
    return IterationResult(
        deltas=deltas,
        norms=norms,
        trace=trace,
        converged=converged,
        diverged=diverged,
        n_steps=len(deltas),
        final_hidden=h.detach().to("cpu", dtype=torch.float32).squeeze(0).clone(),
    )


@torch.no_grad()
def iterate_hidden_posfree(
    model,
    h0: torch.Tensor,
    max_iter: int = 100,
    convergence_threshold: float = 1e-3,
    divergence_factor: float = 100.0,
    save_every: int = 1,
) -> IterationResult:
    """Variant C2: manually run transformer blocks + final ln_f, skipping all embeddings.

    h_0 is whatever the caller passes (typically the result of a standard forward, which
    includes wpe). Subsequent iterations call blocks directly with no positional input.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    h = h0.to(device=device, dtype=dtype)
    if h.dim() == 2:
        h = h.unsqueeze(0)

    init_norm = h.norm().item()
    saved = [h.detach().to("cpu", dtype=torch.float32).clone()]
    norms = [init_norm]
    deltas: list[float] = []
    converged = False
    diverged = False

    blocks = model.transformer.h
    ln_f = model.transformer.ln_f

    for step in range(max_iter):
        hidden = h
        for block in blocks:
            block_out = block(hidden)
            hidden = block_out[0] if isinstance(block_out, tuple) else block_out
        h_next = ln_f(hidden)
        prev_norm = h.norm().item()
        delta = ((h_next - h).norm() / (prev_norm + 1e-12)).item()
        next_norm = h_next.norm().item()
        deltas.append(delta)
        norms.append(next_norm)
        if (step + 1) % save_every == 0 or step == max_iter - 1:
            saved.append(h_next.detach().to("cpu", dtype=torch.float32).clone())

        if next_norm > divergence_factor * init_norm or next_norm < init_norm / divergence_factor:
            diverged = True
            h = h_next
            break
        if delta < convergence_threshold:
            converged = True
            h = h_next
            break
        h = h_next

    trace = torch.cat(saved, dim=0)
    return IterationResult(
        deltas=deltas,
        norms=norms,
        trace=trace,
        converged=converged,
        diverged=diverged,
        n_steps=len(deltas),
        final_hidden=h.detach().to("cpu", dtype=torch.float32).squeeze(0).clone(),
    )
