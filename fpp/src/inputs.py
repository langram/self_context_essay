"""Input zoo per FPP plan §3.3 — 6 categories × 10 inputs, padded/truncated to a fixed seq_len."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase


CATEGORIES = (
    "random_tokens",
    "grammatical_random_semantics",
    "common",
    "ambiguous",
    "nonsense_grammatical",
    "structured",
)


GRAMMATICAL_RANDOM_SEMANTICS = [
    "The blue eight runs quickly the apple yesterday under sleeping water.",
    "A purple thought ate the calendar while seventeen rivers laughed politely.",
    "Five bricks whispered the tomorrow into a cardboard ocean of green numbers.",
    "Every triangle drinks the laughter of forgotten staircases on Tuesday.",
    "The forest argued with a glass clock about silent yellow distances.",
    "Crooked seconds painted a spoon into the hungry breath of mountains.",
    "Quiet algebra danced beneath the broken roof of an honest hour.",
    "The window remembered three salty engines that flew through cold velvet.",
    "Nineteen lazy mirrors agreed to sell the wind a flavored shadow.",
    "A polite hammer dreamed about the rectangular silence of orange rain.",
]

COMMON = [
    "The cat sat on the mat and looked out the window.",
    "Yesterday I went to the store to buy milk and bread.",
    "She opened the book and began to read the first chapter.",
    "He turned off the lights before leaving the office for the night.",
    "The children were playing in the park until it started to rain.",
    "Coffee shops in this neighborhood usually open at seven in the morning.",
    "My friend sent me a long letter about her trip to Japan.",
    "The doctor told him to drink more water and get more sleep.",
    "We watched the sun set behind the hills and then drove home.",
    "Most students prefer studying in the library rather than at home.",
]

AMBIGUOUS = [
    "The trophy doesn't fit in the suitcase because it is too small.",
    "The trophy doesn't fit in the suitcase because it is too large.",
    "Anna told her sister that she had won the prize.",
    "I saw the man with the telescope on the hill last night.",
    "The lawyer questioned the witness who lied about the contract she signed.",
    "Jane gave Mary her keys before she left for the airport.",
    "The chicken is ready to eat after you finish washing the vegetables.",
    "Visiting relatives can be exhausting when they stay for a whole month.",
    "The old men and women left the auditorium as the storm began.",
    "Time flies like an arrow but fruit flies like a banana every summer.",
]

NONSENSE_GRAMMATICAL = [
    "Colorless green ideas sleep furiously near the polished memory of zero.",
    "Triangular silence devours the obedient nostalgia of a rectangular dream.",
    "Mute integers waltz politely beneath the audible hunger of forgotten salt.",
    "Invisible adjectives admire the symmetric bureaucracy of a hollow afternoon.",
    "Transparent algorithms inherit the modest curvature of yellow recursion.",
    "Indignant fractions whisper to the placid geometry of a melted question.",
    "Rectangular memories evaporate into the diagonal patience of muted thunder.",
    "Articulate vacuum politely refuses the spherical invitation of an empty axiom.",
    "Impossible vowels migrate across the modest topology of a punctual silence.",
    "Quiet contradictions inhabit the rounded edges of a thoroughly tangential proof.",
]

STRUCTURED = [
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "for i in range(10):\n    if i % 2 == 0:\n        print(i * i)",
    "class Stack:\n    def __init__(self): self.items = []\n    def push(self, x): self.items.append(x)",
    "import numpy as np\nA = np.array([[1, 2], [3, 4]])\nprint(np.linalg.inv(A))",
    "SELECT user_id, COUNT(*) FROM events WHERE created_at > '2025-01-01' GROUP BY user_id;",
    "f(x) = integral from 0 to x of (1 / (1 + t^2)) dt equals arctan(x).",
    "The matrix product (AB)^T equals B^T times A^T for any conformable A and B.",
    "lim x -> 0 of (sin(x) / x) equals 1, and lim x -> infinity of (1 + 1/x)^x equals e.",
    "<html><body><h1>Hello</h1><p>This is a test paragraph.</p></body></html>",
    "function quicksort(arr) { if (arr.length <= 1) return arr; const pivot = arr[0]; }",
]

CURATED: dict[str, list[str]] = {
    "grammatical_random_semantics": GRAMMATICAL_RANDOM_SEMANTICS,
    "common": COMMON,
    "ambiguous": AMBIGUOUS,
    "nonsense_grammatical": NONSENSE_GRAMMATICAL,
    "structured": STRUCTURED,
}


@dataclass
class InputItem:
    idx: int
    category: str
    text: str
    input_ids: torch.Tensor


def _generate_random_token_texts(tokenizer: PreTrainedTokenizerBase, n: int, seq_len: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    vocab_size = tokenizer.vocab_size
    texts = []
    for _ in range(n):
        ids = [rng.randrange(vocab_size) for _ in range(seq_len)]
        texts.append(tokenizer.decode(ids))
    return texts


def _encode_to_fixed_length(tokenizer: PreTrainedTokenizerBase, text: str, seq_len: int) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) >= seq_len:
        ids = ids[:seq_len]
    else:
        pad_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        ids = ids + [pad_id] * (seq_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def build_input_zoo(
    tokenizer: PreTrainedTokenizerBase,
    categories: list[str],
    inputs_per_category: int,
    seq_len: int,
    seed: int,
) -> list[InputItem]:
    items: list[InputItem] = []
    for cat in categories:
        if cat == "random_tokens":
            texts = _generate_random_token_texts(tokenizer, inputs_per_category, seq_len, seed)
        elif cat in CURATED:
            texts = CURATED[cat][:inputs_per_category]
            if len(texts) < inputs_per_category:
                raise ValueError(f"Category {cat!r} only has {len(texts)} curated items, requested {inputs_per_category}.")
        else:
            raise ValueError(f"Unknown category: {cat!r}. Known: {CATEGORIES}")
        for text in texts:
            items.append(
                InputItem(
                    idx=len(items),
                    category=cat,
                    text=text,
                    input_ids=_encode_to_fixed_length(tokenizer, text, seq_len),
                )
            )
    return items
