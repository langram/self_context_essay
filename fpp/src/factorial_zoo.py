"""Phase 1.2 Experiment H — factorial input zoo per FPP plan v0.3 §3.5.

Generates a 2 × 2 × 2 × 3 = 24 input zoo across factors:
  case        ∈ {capital, lowercase}
  markup      ∈ {with, without}     (presence of <...> tags)
  punct       ∈ {high, low}         (count of single-char punctuation tokens)
  content     ∈ {natural, code, random}

For each (case, markup, punct, content) cell, one carefully constructed input.
Per §9.2, this module should produce candidate inputs for human review BEFORE
running the iteration. The runner is in `run_factorial_zoo.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase


# Single-char punctuation tokens we count for the "punct density" factor.
PUNCT_CHARS = set(",.;:()[]{}<>!?'\"/\\|+-*=&%$#@~^")


@dataclass
class FactorialItem:
    idx: int
    text: str
    case: str        # "capital" or "lowercase"
    markup: str      # "with" or "without"
    punct: str       # "high" or "low"
    content: str     # "natural" / "code" / "random"
    input_ids: torch.Tensor
    n_tokens: int
    punct_token_count: int
    starts_with_capital_letter: bool
    contains_markup: bool


# ---------- Natural text (8 variants) ----------
NATURAL = [
    # (case, markup, punct, text)
    ("capital",   "with",    "high", "<p>The cat sat. The dog barked, then ran! The bird sang; the fish swam. Wow.</p>"),
    ("capital",   "with",    "low",  "<p>The cat sat on the soft warm mat near the window watching the bright blue sky</p>"),
    ("capital",   "without", "high", "The cat sat. The dog barked, then ran! The bird sang; the fish swam. Wow. Done."),
    ("capital",   "without", "low",  "The cat sat on the soft warm mat near the window watching the bright blue sky calmly"),
    ("lowercase", "with",    "high", "<p>the cat sat. the dog barked, then ran! the bird sang; the fish swam. wow.</p>"),
    ("lowercase", "with",    "low",  "<p>the cat sat on the soft warm mat near the window watching the bright blue sky</p>"),
    ("lowercase", "without", "high", "the cat sat. the dog barked, then ran! the bird sang; the fish swam. wow. done."),
    ("lowercase", "without", "low",  "the cat sat on the soft warm mat near the window watching the bright blue sky calmly"),
]

# ---------- Code (8 variants) ----------
# "low punct" code is pseudo-code without symbols — just keywords/identifiers.
CODE = [
    ("capital",   "with",    "high", "<code>If X: Y = Z + 1; A.run(); B, C = 2, 3; D[0] = E + F;</code>"),
    ("capital",   "with",    "low",  "<code>If True Then Run Action Else Return Default Value Or End</code>"),
    ("capital",   "without", "high", "If X: Y = Z + 1; A.run(); B, C = 2, 3; D[0] = E + F; print Q"),
    ("capital",   "without", "low",  "If True Then Run Action Else Return Default Value Or End Block"),
    ("lowercase", "with",    "high", "<code>if x: y = z + 1; a.run(); b, c = 2, 3; d[0] = e + f;</code>"),
    ("lowercase", "with",    "low",  "<code>if true then run action else return default value or end</code>"),
    ("lowercase", "without", "high", "if x: y = z + 1; a.run(); b, c = 2, 3; d[0] = e + f; print q"),
    ("lowercase", "without", "low",  "if true then run action else return default value or end block"),
]

# ---------- Random tokens (8 variants) ----------
# Constructed by hand to control case/markup/punct simultaneously.
# Random here means "no semantic structure" — a soup of unrelated words.
RANDOM = [
    ("capital",   "with",    "high", "<Foo> Bar! Baz, Quux. <Tag/>; End: More? Less; <End/>"),
    ("capital",   "with",    "low",  "<Foo>Bar Baz Quux Tag End More Less Words Here Now Then Soon Later Done</Foo>"),
    ("capital",   "without", "high", "Foo, Bar! Baz; Quux. Tag: End? More; Less! Done. Try; Now."),
    ("capital",   "without", "low",  "Foo Bar Baz Quux Tag End More Less Words Here Now Then Soon Later Done Try"),
    ("lowercase", "with",    "high", "<foo> bar! baz, quux. <tag/>; end: more? less; <end/>"),
    ("lowercase", "with",    "low",  "<foo>bar baz quux tag end more less words here now then soon later done</foo>"),
    ("lowercase", "without", "high", "foo, bar! baz; quux. tag: end? more; less! done. try; now."),
    ("lowercase", "without", "low",  "foo bar baz quux tag end more less words here now then soon later done try"),
]

ALL_RAW = (
    [(c, m, p, t, "natural") for c, m, p, t in NATURAL]
    + [(c, m, p, t, "code") for c, m, p, t in CODE]
    + [(c, m, p, t, "random") for c, m, p, t in RANDOM]
)


def _starts_with_capital_letter(text: str) -> bool:
    """Return whether the first content letter (outside any markup tag) is uppercase."""
    in_tag = False
    for ch in text:
        if ch == "<":
            in_tag = True
            continue
        if ch == ">":
            in_tag = False
            continue
        if in_tag or ch.isspace():
            continue
        if ch.isalpha():
            return ch.isupper()
    return False


def _contains_markup(text: str) -> bool:
    return "<" in text and ">" in text


def _count_punct_tokens(token_strs: list[str]) -> int:
    n = 0
    for t in token_strs:
        s = t.strip()
        if s and all(ch in PUNCT_CHARS for ch in s):
            n += 1
    return n


def build_factorial_zoo(tokenizer: PreTrainedTokenizerBase, seq_len: int = 64) -> list[FactorialItem]:
    items: list[FactorialItem] = []
    for case, markup, punct, text, content in ALL_RAW:
        ids = tokenizer.encode(text, add_special_tokens=False)
        n_real = len(ids)
        if n_real >= seq_len:
            ids = ids[:seq_len]
        else:
            pad_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
            ids = ids + [pad_id] * (seq_len - n_real)
        token_strs = [tokenizer.decode([t]) for t in ids[:n_real]]
        items.append(FactorialItem(
            idx=len(items),
            text=text,
            case=case,
            markup=markup,
            punct=punct,
            content=content,
            input_ids=torch.tensor(ids, dtype=torch.long),
            n_tokens=n_real,
            punct_token_count=_count_punct_tokens(token_strs),
            starts_with_capital_letter=_starts_with_capital_letter(text),
            contains_markup=_contains_markup(text),
        ))
    return items


def write_review_table(items: list[FactorialItem], out_path: str) -> None:
    """Write a markdown table for human review of the candidate zoo."""
    lines = [
        "# Factorial input zoo — candidate review",
        "",
        "Each row is one of the 24 candidate inputs for Phase 1.2 Experiment H. ",
        "Verify that within each (content) group of 8, factor flips are clean — "
        "i.e. the 'capital' vs 'lowercase' pair within fixed (markup, punct) differs only in case, etc.",
        "",
        "Columns:",
        "- `case_check`: ✅ if `starts_with_capital_letter` matches the declared `case` factor",
        "- `markup_check`: ✅ if `contains_markup` matches the declared `markup` factor",
        "- `punct_check`: shows the actual punct-token count; verify the high/low factor declaration is reasonable",
        "",
        "| idx | content | case | markup | punct | n_tok | punct_tok | case_check | markup_check | text |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for r in items:
        case_ok = "✅" if r.starts_with_capital_letter == (r.case == "capital") else "❌"
        markup_ok = "✅" if r.contains_markup == (r.markup == "with") else "❌"
        text_short = r.text[:60].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {r.idx} | {r.content} | {r.case} | {r.markup} | {r.punct} | "
            f"{r.n_tokens} | {r.punct_token_count} | {case_ok} | {markup_ok} | `{text_short}` |"
        )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    from pathlib import Path
    from transformers import GPT2TokenizerFast

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    out_dir = PROJECT_ROOT / "outputs" / "reports" / "phase1_2_factorial_zoo"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    items = build_factorial_zoo(tokenizer, seq_len=64)
    write_review_table(items, out_dir / "candidate_zoo_for_review.md")
    print(f"Wrote candidate zoo to {out_dir / 'candidate_zoo_for_review.md'}")
    print()
    print("Quick factor-balance check:")
    from collections import Counter
    by_factor = Counter()
    for r in items:
        by_factor[r.case] += 1
        by_factor[r.markup] += 1
        by_factor[r.punct] += 1
        by_factor[r.content] += 1
    print(f"  case: {dict((k,v) for k,v in by_factor.items() if k in ('capital','lowercase'))}")
    print(f"  markup: {dict((k,v) for k,v in by_factor.items() if k in ('with','without'))}")
    print(f"  punct: {dict((k,v) for k,v in by_factor.items() if k in ('high','low'))}")
    print(f"  content: {dict((k,v) for k,v in by_factor.items() if k in ('natural','code','random'))}")
    print()
    print("Per-content punct-token counts:")
    for content in ("natural", "code", "random"):
        rs = [r for r in items if r.content == content]
        for r in rs:
            print(f"  [{r.idx:>2d}] {r.content:>7} {r.case:>9} {r.markup:>7} {r.punct:>4}  punct_toks={r.punct_token_count:>2d}  n_tok={r.n_tokens:>2d}  '{r.text[:50]}'")
        print()
