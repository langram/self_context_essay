"""Phase 1.1 Experiment B — extended nesting input zoo per FPP plan v0.2 §3.3.

30 inputs across 6 nested-structure categories (5 each), each tagged with an
explicit nesting depth so we can correlate convergence behaviour with depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class NestedItem:
    idx: int
    category: str
    text: str
    nesting_depth: int
    input_ids: torch.Tensor


HTML = [
    ("<p>Hello world.</p>", 1),
    ('<div class="x"><p>Hello <b>world</b></p></div>', 3),
    ("<div><div><div><p>deep</p></div></div></div>", 4),
    ("<html><body><h1>Title</h1><p>Para <em>one</em>.</p></body></html>", 4),
    ("<table><tr><td><div><span>cell</span></div></td></tr></table>", 5),
]

XML = [
    "<note><body>hello</body></note>",
    "<book><chapter><section><para>text</para></section></chapter></book>",
    '<config><server name="x"><port>80</port><host>a</host></server></config>',
    "<root><a><b><c><d><e>deep</e></d></c></b></a></root>",
    '<?xml version="1.0"?><doc><meta><tag>v</tag></meta><body>x</body></doc>',
]
XML_DEPTH = [2, 4, 4, 6, 4]

JSON = [
    '{"a": 1, "b": 2}',
    '{"user": {"name": "x", "age": 30}}',
    '{"a": {"b": {"c": {"d": {"e": 1}}}}}',
    '{"items": [{"id": 1, "tags": ["a", "b"]}, {"id": 2, "tags": ["c"]}]}',
    '{"x": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]}',
]
JSON_DEPTH = [1, 2, 5, 4, 4]

LATEX = [
    r"$x^2 + y^2 = z^2$",
    r"$\frac{a}{b + \frac{c}{d}}$",
    r"\begin{matrix} a & b \\ c & d \end{matrix}",
    r"\sum_{i=1}^{n} \frac{1}{i^2} = \frac{\pi^2}{6}",
    r"\int_0^1 \frac{1}{1+x^2}\,dx = \frac{\pi}{4}",
]
LATEX_DEPTH = [1, 3, 2, 3, 3]

PURE_BRACKETS = [
    "( )",
    "( ( ) )",
    "( ( ( ) ) )",
    "( ( ( ( ) ) ) )",
    "( ( ( ( ( ) ) ) ) )",
]
PURE_DEPTH = [1, 2, 3, 4, 5]

MARKDOWN_LISTS = [
    "- item one\n- item two\n- item three",
    "- a\n  - b\n- c",
    "- a\n  - b\n    - c\n      - d",
    "1. one\n   1. one-a\n      1. one-a-i\n2. two",
    "- a\n  - b\n    - c\n      - d\n        - e\n          - f",
]
MARKDOWN_DEPTH = [1, 2, 4, 3, 6]


CURATED = (
    ("html", HTML, [d for _, d in HTML]),
    ("xml", list(zip(XML, XML_DEPTH)), XML_DEPTH),
    ("json", list(zip(JSON, JSON_DEPTH)), JSON_DEPTH),
    ("latex", list(zip(LATEX, LATEX_DEPTH)), LATEX_DEPTH),
    ("pure_brackets", list(zip(PURE_BRACKETS, PURE_DEPTH)), PURE_DEPTH),
    ("markdown_list", list(zip(MARKDOWN_LISTS, MARKDOWN_DEPTH)), MARKDOWN_DEPTH),
)


def _normalise(category_data) -> list[tuple[str, int]]:
    """Each curated category may store either list of (text, depth) tuples or list of texts + depth list."""
    if isinstance(category_data[0], tuple):
        return list(category_data)
    raise ValueError("expected list of (text, depth) tuples")


def _encode_to_fixed_length(tokenizer: PreTrainedTokenizerBase, text: str, seq_len: int) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) >= seq_len:
        ids = ids[:seq_len]
    else:
        pad_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        ids = ids + [pad_id] * (seq_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def build_nested_zoo(tokenizer: PreTrainedTokenizerBase, seq_len: int = 64) -> list[NestedItem]:
    items: list[NestedItem] = []
    for cat, texts_with_depth, _ in CURATED:
        for text, depth in _normalise(texts_with_depth):
            items.append(
                NestedItem(
                    idx=len(items),
                    category=cat,
                    text=text,
                    nesting_depth=depth,
                    input_ids=_encode_to_fixed_length(tokenizer, text, seq_len),
                )
            )
    return items
