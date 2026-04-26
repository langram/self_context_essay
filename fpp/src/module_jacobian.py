"""Phase 1.2 Experiment F — module-level contraction decomposition per FPP plan v0.3 §3.3.

Estimates spectral norm σ_max(J) of selected sub-modules of GPT-2 small via power
iteration on J^T J using PyTorch's jvp + vjp at a single point. (The plan's
caveat against autograd was about graphs through full iteration trajectories;
single-point Jacobian-vector products are O(1) sub-module evals and fast.)

For each (model_variant, input, test_point, sub_module) we report σ_max(J).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

from src.inputs import build_input_zoo
from src.iterate import initial_hidden, iterate_hidden


def load_model_eager_attn(model_name: str, dtype: str, device: str,
                          random_init: bool = False, seed: int = 42):
    """Same as src.iterate.load_model but with attn_implementation='eager' so jvp/vjp work."""
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if random_init:
        config = GPT2Config.from_pretrained(model_name, attn_implementation="eager")
        torch.manual_seed(seed)
        model = GPT2LMHeadModel(config)
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
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
        model = GPT2LMHeadModel.from_pretrained(model_name, dtype=torch_dtype, attn_implementation="eager")

    model.to(device)
    model.eval()
    return model, tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "phase1_2_module_jacobian"

# Layer to analyse for sub-module decomposition (middle of GPT-2 small's 12-layer stack)
ANALYSE_LAYER = 6


@dataclass
class JResult:
    label: str          # "trained" or "random"
    input_idx: int
    input_category: str
    test_point: str     # "h_0", "h_1", "h_fixed"
    submodule: str      # name of operator
    sigma_max: float
    n_power_iters: int
    converged: bool


@torch.no_grad()
def _no_grad_pass(f, h):
    return f(h)


def power_iteration_sigma(
    f: Callable[[torch.Tensor], torch.Tensor],
    h: torch.Tensor,
    max_iter: int = 60,
    tol: float = 1e-4,
    seed: int = 42,
) -> tuple[float, int, bool]:
    """Estimate σ_max(J_f(h)) by power iteration on J^T J using jvp + vjp.

    h: any tensor shape; f returns a tensor of arbitrary shape (we treat it as flattened).
    """
    g = torch.Generator(device=h.device).manual_seed(seed)
    v = torch.randn(h.shape, generator=g, device=h.device, dtype=h.dtype)
    v = v / (v.norm() + 1e-12)
    sigma_prev = 0.0
    converged = False
    for it in range(max_iter):
        _, Jv = torch.autograd.functional.jvp(f, h, v=v, create_graph=False, strict=False)
        sigma = Jv.norm().item()
        _, JtJv = torch.autograd.functional.vjp(f, h, v=Jv, create_graph=False, strict=False)
        norm_JtJv = JtJv.norm().item()
        if norm_JtJv < 1e-20:
            return sigma, it + 1, True
        v = JtJv / norm_JtJv
        if it > 5 and abs(sigma - sigma_prev) < tol * max(sigma, 1e-12):
            converged = True
            return sigma, it + 1, True
        sigma_prev = sigma
    return sigma_prev, max_iter, converged


def make_submodule_fns(model) -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    """Build a dict of single-input single-output functions for each sub-module of layer ANALYSE_LAYER."""
    block = model.transformer.h[ANALYSE_LAYER]
    ln_f = model.transformer.ln_f
    blocks = model.transformer.h

    def ln1(h):
        return block.ln_1(h)

    def attn_sublayer(h):
        # x + attn(ln_1(x))
        a = block.ln_1(h)
        a = block.attn(a)[0]  # GPT2Attention returns tuple
        return h + a

    def ln2(h):
        return block.ln_2(h)

    def mlp_sublayer(h):
        # x + mlp(ln_2(x))
        m = block.ln_2(h)
        m = block.mlp(m)
        return h + m

    def full_block(h):
        out = block(h)
        return out[0] if isinstance(out, tuple) else out

    def lnf(h):
        return ln_f(h)

    def full_stack(h):
        # Full posfree block-stack + ln_f (the Experiment C2 iteration map)
        hidden = h
        for b in blocks:
            o = b(hidden)
            hidden = o[0] if isinstance(o, tuple) else o
        return ln_f(hidden)

    return {
        "ln_1": ln1,
        "attn_sublayer": attn_sublayer,
        "ln_2": ln2,
        "mlp_sublayer": mlp_sublayer,
        f"full_block_L{ANALYSE_LAYER}": full_block,
        "ln_f": lnf,
        "full_stack_posfree": full_stack,
    }


def get_test_points(model, input_ids: torch.Tensor, max_iter_for_fixed: int = 100) -> dict[str, torch.Tensor]:
    """Return h_0 (initial forward), h_1 (one mode-A iter), h_fixed (converged state)."""
    h0 = initial_hidden(model, input_ids)
    res = iterate_hidden(model, h0, max_iter=max_iter_for_fixed,
                        convergence_threshold=1e-3, divergence_factor=100.0, save_every=1)
    h1 = res.trace[1].unsqueeze(0).to(h0.device, dtype=h0.dtype)  # h_1
    h_fixed = res.trace[-1].unsqueeze(0).to(h0.device, dtype=h0.dtype)  # last saved state
    return {"h_0": h0, "h_1": h1, "h_fixed": h_fixed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-cats", nargs="+",
                        default=["random_tokens", "common", "ambiguous", "structured"],
                        help="Phase 1 categories to sample from (1 input per category)")
    parser.add_argument("--include-b-zoo", action="store_true", default=True,
                        help="Include 6 representative B inputs (one per nested category)")
    parser.add_argument("--max-iter-fixed", type=int, default=200)
    parser.add_argument("--power-iter-max", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fig_dir = PROJECT_ROOT / "outputs" / "figures" / RUN_ID
    report_dir = PROJECT_ROOT / "outputs" / "reports" / RUN_ID
    processed_dir = PROJECT_ROOT / "data" / "processed" / RUN_ID
    for d in (fig_dir, report_dir, processed_dir):
        d.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    print(f"[{RUN_ID}] loading inputs (Phase 1 reps + B reps)")

    all_results: list[JResult] = []

    for variant in ("trained", "random"):
        print(f"[{RUN_ID}] loading {variant} model")
        model, tokenizer = load_model_eager_attn("gpt2", "float32", "cuda",
                                      random_init=(variant == "random"), seed=args.seed)
        submods = make_submodule_fns(model)

        # Phase 1 representative inputs (one per requested category)
        phase1_items = build_input_zoo(tokenizer, args.phase1_cats, 1, 32, args.seed)

        # Phase 1.1 B representative inputs (idx 0,5,10,15,20,25 = one per nested category)
        b_inputs: list[tuple[int, str, torch.Tensor]] = []
        if args.include_b_zoo:
            from src.extended_inputs import build_nested_zoo
            b_items = build_nested_zoo(tokenizer, seq_len=64)
            for j in [0, 5, 10, 15, 20, 25]:
                b_inputs.append((j, b_items[j].category, b_items[j].input_ids))

        all_inputs: list[tuple[str, int, str, torch.Tensor]] = []
        for it in phase1_items:
            all_inputs.append(("phase1", it.idx, it.category, it.input_ids))
        for j, cat, ids in b_inputs:
            all_inputs.append(("phase1_1b", j, cat, ids))

        for source, idx, category, input_ids in tqdm(all_inputs, desc=f"{variant}"):
            try:
                tps = get_test_points(model, input_ids, max_iter_for_fixed=args.max_iter_fixed)
            except Exception as e:
                print(f"  skip {source}/{idx} due to {e}")
                continue
            for tp_name, h in tps.items():
                for submod_name, f in submods.items():
                    try:
                        sigma, n_iter, conv = power_iteration_sigma(
                            f, h, max_iter=args.power_iter_max, tol=1e-4, seed=args.seed
                        )
                    except Exception as e:
                        print(f"    error {variant}/{source}_{idx}/{tp_name}/{submod_name}: {e}")
                        sigma, n_iter, conv = float("nan"), 0, False
                    all_results.append(JResult(
                        label=variant,
                        input_idx=idx if source == "phase1" else 1000 + idx,
                        input_category=f"{source}:{category}",
                        test_point=tp_name,
                        submodule=submod_name,
                        sigma_max=float(sigma),
                        n_power_iters=int(n_iter),
                        converged=bool(conv),
                    ))
        del model
        torch.cuda.empty_cache()

    # Persist results
    payload = [r.__dict__ for r in all_results]
    (processed_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Aggregate: mean σ_max per (variant, test_point, submodule)
    summary: dict = {}
    for v in ("trained", "random"):
        for tp in ("h_0", "h_1", "h_fixed"):
            for sm in list(make_submodule_fns(torch.nn.Identity if False else (lambda: None)())) if False else \
                       ["ln_1", "attn_sublayer", "ln_2", "mlp_sublayer",
                        f"full_block_L{ANALYSE_LAYER}", "ln_f", "full_stack_posfree"]:
                vals = [r.sigma_max for r in all_results
                        if r.label == v and r.test_point == tp and r.submodule == sm and not np.isnan(r.sigma_max)]
                summary[f"{v}/{tp}/{sm}"] = {
                    "n": len(vals),
                    "mean": float(np.mean(vals)) if vals else None,
                    "std": float(np.std(vals)) if vals else None,
                    "min": float(np.min(vals)) if vals else None,
                    "max": float(np.max(vals)) if vals else None,
                }
    (processed_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot: heatmap of mean σ_max per (submodule, test_point) for each variant
    submods = ["ln_1", "attn_sublayer", "ln_2", "mlp_sublayer",
               f"full_block_L{ANALYSE_LAYER}", "ln_f", "full_stack_posfree"]
    test_points = ["h_0", "h_1", "h_fixed"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, variant in zip(axes, ("trained", "random")):
        mat = np.zeros((len(submods), len(test_points)))
        for i, sm in enumerate(submods):
            for j, tp in enumerate(test_points):
                v = summary[f"{variant}/{tp}/{sm}"]["mean"]
                mat[i, j] = v if v is not None else np.nan
        im = ax.imshow(mat, cmap="coolwarm", vmin=0, vmax=2.0, aspect="auto")
        ax.set_xticks(range(len(test_points))); ax.set_xticklabels(test_points)
        ax.set_yticks(range(len(submods))); ax.set_yticklabels(submods, fontsize=9)
        ax.set_title(f"σ_max — {variant}")
        for i in range(len(submods)):
            for j in range(len(test_points)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=9,
                            color="black" if 0.4 < mat[i, j] < 1.6 else "white")
        plt.colorbar(im, ax=ax, label="σ_max")
    fig.suptitle(f"Module-level spectral norm at three test points (mean over inputs) — analyse layer = {ANALYSE_LAYER}")
    fig.tight_layout()
    fig.savefig(fig_dir / "module_jacobian_heatmaps.png", dpi=140)
    plt.close(fig)

    print(f"[{RUN_ID}] done. Results saved to {processed_dir / 'results.json'}")


if __name__ == "__main__":
    main()
