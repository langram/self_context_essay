"""Phase 1.3 Experiment I — markup-type triangulation per FPP plan v0.4 §3.2.

30 inputs across 6 bracket-type categories. Tests whether the secondary
attractor ("HTML-induced metastable structure" per the v0.4 §0.1 naming
discipline) is triggered by any tag-like character or specifically by
angle-bracket BPE tokens.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.iterate import initial_hidden, iterate_hidden, load_model
from src.metrics import effective_rank, project_to_vocab


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "phase1_3_markup_triangulation"


# Six bracket-type categories x 5 inputs each = 30 inputs.
# Each input has the same content shape ("tag/text/end-tag") but different bracket characters.
ANGLE = [
    "<a>hello world</a>",
    "<tag>some content here</tag>",
    "<x>nested <y>inner</y> outer</x>",
    "<doc><body><p>text inside</p></body></doc>",
    "<root attr=value>content body here closing</root>",
]
SQUARE = [
    "[a]hello world[/a]",
    "[tag]some content here[/tag]",
    "[x]nested [y]inner[/y] outer[/x]",
    "[doc][body][p]text inside[/p][/body][/doc]",
    "[root attr=value]content body here closing[/root]",
]
CURLY = [
    "{a}hello world{/a}",
    "{tag}some content here{/tag}",
    "{x}nested {y}inner{/y} outer{/x}",
    "{doc}{body}{p}text inside{/p}{/body}{/doc}",
    "{root attr=value}content body here closing{/root}",
]
PARENS = [
    "(a)hello world(/a)",
    "(tag)some content here(/tag)",
    "(x)nested (y)inner(/y) outer(/x)",
    "(doc)(body)(p)text inside(/p)(/body)(/doc)",
    "(root attr=value)content body here closing(/root)",
]
QUOTES = [
    '"a"hello world"/a"',
    '"tag"some content here"/tag"',
    '"x"nested "y"inner"/y" outer"/x"',
    '"doc""body""p"text inside"/p""/body""/doc"',
    '"root attr=value"content body here closing"/root"',
]
ISOLATED = [
    "< only this and content here filling out the sequence to a reasonable length",
    "> only this and content here filling out the sequence to a reasonable length",
    "[ only this and content here filling out the sequence to a reasonable length",
    "] only this and content here filling out the sequence to a reasonable length",
    "/ only this and content here filling out the sequence to a reasonable length",
]

CATEGORIES = (
    ("angle", ANGLE),
    ("square", SQUARE),
    ("curly", CURLY),
    ("parens", PARENS),
    ("quotes", QUOTES),
    ("isolated", ISOLATED),
)


@dataclass
class TRecord:
    idx: int
    category: str
    text: str
    n_tokens: int
    token_ids_bracket_chars: list[int]   # which token IDs corresponded to bracket chars in this text
    converged: bool
    diverged: bool
    n_steps: int
    final_norm: float
    effective_rank: float
    cos_to_lowercase_attractor: float
    cos_to_html_metastable: float
    final_top5_per_position_first8: list[list[str]]


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def _bracket_char_token_ids(tokenizer, text: str) -> list[int]:
    bracket_chars = set("<>[]{}()\"'/\\")
    ids = tokenizer.encode(text, add_special_tokens=False)
    out = []
    for tok_id in ids:
        decoded = tokenizer.decode([tok_id])
        if any(c in bracket_chars for c in decoded):
            out.append(int(tok_id))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=64)
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

    # Reference attractor representations (mean-pooled and unit normalised)
    # 1. lowercase attractor: any converged Phase 1 trace's final state
    p1_ref = None
    for i in range(60):
        d = torch.load(PROJECT_ROOT / "data/raw/phase1_gpt2_small/trained" / f"trace_{i:03d}.pt",
                       weights_only=False)
        if d["converged"]:
            p1_ref = d["trace"][-1].numpy()
            break
    lc_ref = p1_ref.mean(axis=0)
    lc_ref /= max(np.linalg.norm(lc_ref), 1e-12)

    # 2. HTML-induced metastable structure: pick a Phase 1.1 B trace that landed in capital basin
    html_ref_full = None
    for i in [0, 1, 2, 3, 4]:  # html group from B
        d = torch.load(PROJECT_ROOT / "data/raw/phase1_1_html_outlier" / f"trace_{i:03d}.pt",
                       weights_only=False)
        # Final saved state should be in the "capital" / HTML structure
        html_ref_full = d["trace"][-1].numpy()
        break
    html_ref = html_ref_full.mean(axis=0)
    html_ref /= max(np.linalg.norm(html_ref), 1e-12)
    print(f"[{RUN_ID}] reference attractor cos(lowercase, html_metastable) = {float(lc_ref @ html_ref):.4f}")

    # Encode inputs
    items: list[tuple[int, str, str, torch.Tensor]] = []
    for cat, texts in CATEGORIES:
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            n_real = len(ids)
            if n_real >= args.seq_len:
                ids = ids[:args.seq_len]
            else:
                pad = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
                ids = ids + [pad] * (args.seq_len - n_real)
            items.append((len(items), cat, text, torch.tensor(ids, dtype=torch.long)))

    records: list[TRecord] = []
    finals: list[np.ndarray] = []

    t0 = time.time()
    for idx, cat, text, input_ids in tqdm(items, desc="markup-triangulation"):
        h0 = initial_hidden(model, input_ids)
        res = iterate_hidden(
            model, h0, max_iter=args.max_iter,
            convergence_threshold=args.threshold, divergence_factor=100.0,
            save_every=args.save_every,
        )
        fhid = res.final_hidden.numpy()
        fvec = fhid.mean(axis=0)
        fvec /= max(np.linalg.norm(fvec), 1e-12)

        proj = project_to_vocab(res.final_hidden, model, tokenizer, top_k=5)
        records.append(TRecord(
            idx=idx,
            category=cat,
            text=text,
            n_tokens=int((input_ids != tokenizer.eos_token_id).sum().item()),
            token_ids_bracket_chars=_bracket_char_token_ids(tokenizer, text),
            converged=res.converged,
            diverged=res.diverged,
            n_steps=res.n_steps,
            final_norm=float(np.linalg.norm(fhid)),
            effective_rank=effective_rank(fhid),
            cos_to_lowercase_attractor=float(fvec @ lc_ref),
            cos_to_html_metastable=float(fvec @ html_ref),
            final_top5_per_position_first8=proj.top5_token_strs[:8],
        ))
        finals.append(fhid)
        torch.save(
            {"trace": res.trace, "deltas": res.deltas, "norms": res.norms,
             "converged": res.converged, "diverged": res.diverged,
             "category": cat, "text": text, "input_ids": input_ids,
             "save_every": args.save_every},
            raw_root / f"trace_{idx:03d}.pt",
        )

    elapsed = time.time() - t0

    # Persist
    payload = []
    for r in records:
        d = asdict(r)
        d["final_top5_per_position_first8"] = [["\\n" if t == "\n" else t for t in row]
                                                for row in r.final_top5_per_position_first8]
        payload.append(d)
    (processed_dir / "records.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Aggregate per category
    summary = {
        "timestamp": dt.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "git_commit": _git_commit(),
        "elapsed_seconds": elapsed,
        "n_inputs": len(records),
        "by_category": {},
    }
    for cat, _ in CATEGORIES:
        rs = [r for r in records if r.category == cat]
        cos_lc = [r.cos_to_lowercase_attractor for r in rs]
        cos_html = [r.cos_to_html_metastable for r in rs]
        summary["by_category"][cat] = {
            "n": len(rs),
            "n_converged": sum(1 for r in rs if r.converged),
            "mean_n_steps": float(np.mean([r.n_steps for r in rs])),
            "mean_eff_rank": float(np.mean([r.effective_rank for r in rs])),
            "mean_final_norm": float(np.mean([r.final_norm for r in rs])),
            "mean_cos_to_lowercase": float(np.mean(cos_lc)),
            "mean_cos_to_html_metastable": float(np.mean(cos_html)),
            "min_cos_to_lowercase": float(np.min(cos_lc)),
            "max_cos_to_html_metastable": float(np.max(cos_html)),
        }
    (processed_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot: per-input scatter (cos to lc vs cos to html)
    fig, ax = plt.subplots(figsize=(8, 6))
    cat_colors = {c: plt.cm.tab10(i) for i, (c, _) in enumerate(CATEGORIES)}
    for r in records:
        ax.scatter(r.cos_to_lowercase_attractor, r.cos_to_html_metastable,
                   color=cat_colors[r.category], s=80, alpha=0.85,
                   edgecolors="black", linewidths=0.4, label=r.category)
    handles, labels = ax.get_legend_handles_labels()
    by_lab = dict(zip(labels, handles))
    ax.legend(by_lab.values(), by_lab.keys(), fontsize=9, loc="lower left")
    ax.set_xlabel("cos to lowercase attractor")
    ax.set_ylabel("cos to HTML-induced metastable structure")
    ax.axvline(0.999, color="grey", linestyle=":", linewidth=0.8)
    ax.axhline(0.95, color="grey", linestyle=":", linewidth=0.8)
    ax.set_title("Each input's final state vs the two reference attractors")
    fig.tight_layout()
    fig.savefig(fig_dir / "cos_scatter.png", dpi=140)
    plt.close(fig)

    print(f"[{RUN_ID}] done in {elapsed:.1f}s")
    for cat, _ in CATEGORIES:
        agg = summary["by_category"][cat]
        print(f"  {cat:>9}: n_steps={agg['mean_n_steps']:>5.1f}  rank={agg['mean_eff_rank']:.2f}  "
              f"||C*||={agg['mean_final_norm']:>5.0f}  "
              f"cos_lc={agg['mean_cos_to_lowercase']:>+.3f}  "
              f"cos_html={agg['mean_cos_to_html_metastable']:>+.3f}")


if __name__ == "__main__":
    main()
