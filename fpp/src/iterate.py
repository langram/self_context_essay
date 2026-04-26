"""Hidden-state self-iteration per FPP plan §3.2 (mode A).

Treats a frozen Transformer as a discrete dynamical system C_{n+1} = transformer_θ(C_n)
by feeding last_hidden_state back through the transformer stack via inputs_embeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast


@dataclass
class IterationResult:
    deltas: list[float]                # ||h_{n+1} - h_n|| / ||h_n|| at each step
    norms: list[float]                 # ||h_n|| Frobenius norm at each step (incl. h_0)
    trace: torch.Tensor                # [n_steps+1, seq_len, hidden_dim] CPU tensor
    converged: bool
    diverged: bool                     # numerical instability (norm explosion or collapse)
    n_steps: int                       # number of iteration steps actually taken
    final_hidden: torch.Tensor         # [seq_len, hidden_dim] CPU tensor


def load_model(
    model_name: str,
    dtype: str,
    device: str,
    random_init: bool = False,
    seed: int = 42,
) -> tuple[GPT2LMHeadModel, GPT2TokenizerFast]:
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if random_init:
        config = GPT2Config.from_pretrained(model_name)
        torch.manual_seed(seed)
        model = GPT2LMHeadModel(config)
        # Match GPT-2's reference init: linear/embedding ~ N(0, 0.02), LayerNorm gain=1 bias=0,
        # other biases zero. Critical not to zero LayerNorm gains — that nukes the forward pass.
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, (torch.nn.Linear,)):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=gen)
                    if module.bias is not None:
                        module.bias.zero_()
                elif isinstance(module, torch.nn.Embedding):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=gen)
                elif isinstance(module, torch.nn.LayerNorm):
                    module.weight.fill_(1.0)
                    module.bias.zero_()
        model = model.to(dtype=torch_dtype)
    else:
        model = GPT2LMHeadModel.from_pretrained(model_name, dtype=torch_dtype)

    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def initial_hidden(model: GPT2LMHeadModel, input_ids: torch.Tensor) -> torch.Tensor:
    """Run a normal forward pass on the token IDs and return last_hidden_state."""
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(next(model.parameters()).device)
    out = model.transformer(input_ids=input_ids, return_dict=True)
    return out.last_hidden_state


@torch.no_grad()
def iterate_hidden(
    model: GPT2LMHeadModel,
    h0: torch.Tensor,
    max_iter: int = 100,
    convergence_threshold: float = 1e-3,
    divergence_factor: float = 100.0,
    save_every: int = 1,
) -> IterationResult:
    """Iterate h_{n+1} = transformer(inputs_embeds=h_n) until convergence/divergence/max_iter."""
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
    norms = [init_norm]
    deltas: list[float] = []
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

        if next_norm > divergence_factor * init_norm or next_norm < init_norm / divergence_factor:
            diverged = True
            h = h_next
            break

        if delta < convergence_threshold:
            converged = True
            h = h_next
            break

        h = h_next

    trace = torch.cat(saved, dim=0)  # [n_saved, seq_len, hidden_dim]
    return IterationResult(
        deltas=deltas,
        norms=norms,
        trace=trace,
        converged=converged,
        diverged=diverged,
        n_steps=len(deltas),
        final_hidden=h.detach().to("cpu", dtype=torch.float32).squeeze(0).clone(),
    )
