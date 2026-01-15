"""
Docstring for utils.format_suits

Utility program for formatting poker cards as latex

Cards come in 2,n array
row 0: values in 2...14
row 1: suit in 1...4 as S,H,D,C
"""

from __future__ import annotations
from typing import Iterable, List, Sequence
import numpy as np



VALUE_MAP = {
    10: "T",
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
}

UNICODE_SUIT_MAP = {
    1: "♠",
    2: "♥",
    3: "♦",
    4: "♣",
}


def value_to_rank(value: int) -> str:
    """Map 2..14 to '2'..'9','T','J','Q','K','A'."""
    if 2 <= value <= 9:
        return str(value)
    if value in VALUE_MAP:
        return VALUE_MAP[value]
    raise ValueError(f"Card value must be in 2..14; got {value}")


def suit_to_unicode(suit: int) -> str:
    """Map suit 1..4 to a Unicode suit symbol: ♠ ♥ ♦ ♣."""
    if suit not in UNICODE_SUIT_MAP:
        raise ValueError(f"Suit must be in 1..4; got {suit}")
    return UNICODE_SUIT_MAP[suit]


def card_to_unicode(value: int, suit: int) -> str:
    """Format a single card as a Unicode string, e.g. 'A♠', 'T♥'."""
    rank = value_to_rank(int(value))
    sym = suit_to_unicode(int(suit))
    return f"{rank}{sym}"


def cards_to_unicode_list(cards: np.ndarray) -> List[str]:
    """Convert a (2, n) card matrix to a list of Unicode strings (one per card)."""
    if not isinstance(cards, np.ndarray):
        cards = np.asarray(cards)

    if cards.ndim != 2 or cards.shape[0] != 2:
        raise ValueError(f"cards must have shape (2, n); got {cards.shape}")

    out: List[str] = []
    for j in range(cards.shape[1]):
        v = int(cards[0, j])
        s = int(cards[1, j])
        out.append(card_to_unicode(v, s))
    return out


if __name__ == "__main__":
    # Quick test
    example = np.array(
        [
            [14, 13, 12, 11, 10, 9, 2],
            [1,  2,  3,  4,  1, 2, 4],
        ]
    )
    print(cards_to_unicode_list(example))
