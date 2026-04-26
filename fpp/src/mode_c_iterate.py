"""Phase 1.1 Experiment D — Mode C token interface per FPP plan v0.2 §3.5.

Pipeline: input_ids → forward → logits → argmax per position → new input_ids → repeat.
Convergence: t_n == t_{n+1} as token-id sequences. Limit cycle: t_n appears in earlier history.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ModeCResult:
    token_trajectory: list[list[int]]   # each entry is the seq_len token-id list at that step (incl. t_0 = input)
    converged: bool                      # t_n == t_{n+1}
    cycle_period: int | None             # if non-None, smallest period detected
    n_steps: int                         # iterations actually run
    final_tokens: list[int]
    final_text: str


@torch.no_grad()
def iterate_mode_c(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_iter: int = 50,
) -> ModeCResult:
    device = next(model.parameters()).device
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]

    history: list[list[int]] = [input_ids[0].detach().cpu().tolist()]
    converged = False
    cycle_period: int | None = None

    cur = input_ids
    for step in range(max_iter):
        out = model(input_ids=cur, return_dict=True)
        logits = out.logits  # [1, seq_len, vocab]
        next_ids = logits.argmax(dim=-1)  # [1, seq_len]
        next_seq = next_ids[0].detach().cpu().tolist()

        # Convergence (fixed point in token space)
        if next_seq == history[-1]:
            history.append(next_seq)
            converged = True
            break

        # Limit cycle: next_seq matches some earlier element of history
        if next_seq in history:
            prev_idx = history.index(next_seq)
            cycle_period = len(history) - prev_idx
            history.append(next_seq)
            break

        history.append(next_seq)
        cur = next_ids

    final_tokens = history[-1]
    final_text = tokenizer.decode(final_tokens)
    return ModeCResult(
        token_trajectory=history,
        converged=converged,
        cycle_period=cycle_period,
        n_steps=len(history) - 1,
        final_tokens=final_tokens,
        final_text=final_text,
    )
