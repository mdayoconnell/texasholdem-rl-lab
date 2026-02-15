
"""Formatting utilities for poker hand evaluations.

This module converts the evaluation tuple returned by `evaluate_hand` into
human-readable text.

Expected evaluation tuple format:
  (hand_rank: int, topcard: int, kickers: array-like[int])

Hand rank convention (1 best):
  1  Royal Flush
  2  Straight Flush
  3  Four of a Kind
  4  Full House
  5  Flush
  6  Straight
  7  Three of a Kind
  8  Two Pair
  9  One Pair
  10 High Card

`topcard` semantics:
  - straight / straight flush: high card of the straight (wheel A-2-3-4-5 counts as 5)
  - quads / trips / pairs: value of the made hand (high pair for two-pair)
  - flush / high card: highest card among the chosen 5

`tiebreak` controls whether we include kickers / secondary tie-break values in the string.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


_RANK_TO_NAME = {
    1: "royal flush",
    2: "straight flush",
    3: "four of a kind",
    4: "full house",
    5: "flush",
    6: "straight",
    7: "three of a kind",
    8: "two pair",
    9: "pair",
    10: "high card",
}


def value_to_rank(v: int) -> str:
    """Map 2..14 to '2'..'9','T','J','Q','K','A'."""
    v = int(v)
    if v == 14:
        return "A"
    if v == 13:
        return "K"
    if v == 12:
        return "Q"
    if v == 11:
        return "J"
    if v == 10:
        return "T"
    if 2 <= v <= 9:
        return str(v)
    raise ValueError(f"Card value must be in 2..14; got {v}")


def value_to_word(v: int, *, plural: bool = False) -> str:
    """Map 2..14 to English rank names (ace, king, ...)."""
    v = int(v)
    base = {
        14: "ace",
        13: "king",
        12: "queen",
        11: "jack",
        10: "ten",
        9: "nine",
        8: "eight",
        7: "seven",
        6: "six",
        5: "five",
        4: "four",
        3: "three",
        2: "two",
    }[v]

    if not plural:
        return base

    # Common plurals.
    if base == "six":
        return "sixes"
    return base + "s"


def _as_int_array(x) -> np.ndarray:
    if x is None:
        return np.array([], dtype=int)
    arr = np.asarray(x, dtype=int).ravel()
    return arr


def format_hand(ev: Tuple[int, int, Iterable[int]], *, tiebreak: bool = False) -> str:
    """Format an evaluation tuple into readable text.

    Examples:
      (9, 14, [13, 12, 9]) -> "pair of aces"
      (8, 13, [7, 5])      -> "two pair, kings and sevens"
      (6, 7, [])           -> "straight to seven"

    If `tiebreak=True`, include the tie-break detail (kickers / secondary values).
    """

    hand_rank, topcard, kickers = ev
    hand_rank = int(hand_rank)
    topcard = int(topcard)
    k = _as_int_array(kickers)

    if hand_rank not in _RANK_TO_NAME:
        raise ValueError(f"Unknown hand_rank {hand_rank}; expected 1..10")

    # 1) Royal flush
    if hand_rank == 1:
        return "royal flush"

    # 2) Straight flush
    if hand_rank == 2:
        out = f"straight flush to {value_to_word(topcard)}"
        return out

    # 3) Quads
    if hand_rank == 3:
        out = f"four of a kind, {value_to_word(topcard, plural=True)}"
        if tiebreak and len(k) >= 1:
            out += f" (kicker {value_to_word(k[0])})"
        return out

    # 4) Full house
    if hand_rank == 4:
        # convention: kickers[0] is the pair value
        out = f"full house, {value_to_word(topcard, plural=True)} over {value_to_word(k[0], plural=True)}" if len(k) >= 1 else f"full house, {value_to_word(topcard, plural=True)}"
        return out

    # 5) Flush
    if hand_rank == 5:
        out = f"flush, {value_to_word(topcard)} high"
        if tiebreak and len(k) > 0:
            ks = ", ".join(value_to_word(x) for x in k)
            out += f" (then {ks})"
        return out

    # 6) Straight
    if hand_rank == 6:
        return f"straight to {value_to_word(topcard)}"

    # 7) Trips
    if hand_rank == 7:
        out = f"three of a kind, {value_to_word(topcard, plural=True)}"
        if tiebreak and len(k) > 0:
            ks = ", ".join(value_to_word(x) for x in k)
            out += f" (kickers {ks})"
        return out

    # 8) Two pair
    if hand_rank == 8:
        # convention: topcard = high pair; kickers = [lowpair, kicker]
        lowpair = int(k[0]) if len(k) >= 1 else None
        out = (
            f"two pair, {value_to_word(topcard, plural=True)} and {value_to_word(lowpair, plural=True)}"
            if lowpair is not None
            else f"two pair, {value_to_word(topcard, plural=True)}"
        )
        if tiebreak and len(k) >= 2:
            out += f" (kicker {value_to_word(k[1])})"
        return out

    # 9) Pair
    if hand_rank == 9:
        out = f"pair of {value_to_word(topcard, plural=True)}"
        if tiebreak and len(k) > 0:
            ks = ", ".join(value_to_word(x) for x in k)
            out += f" (kickers {ks})"
        return out

    # 10) High card
    if hand_rank == 10:
        out = f"high card {value_to_word(topcard)}"
        if tiebreak and len(k) > 0:
            ks = ", ".join(value_to_word(x) for x in k)
            out += f" (then {ks})"
        return out

    # Unreachable
    return _RANK_TO_NAME[hand_rank]


if __name__ == "__main__":
    # Smoke tests
    tests = [
        (1, 14, []),
        (2, 9, []),
        (3, 12, [7]),
        (4, 11, [9]),
        (5, 14, [13, 11, 9, 6]),
        (6, 5, []),
        (7, 8, [14, 12]),
        (8, 13, [7, 5]),
        (9, 14, [13, 12, 9]),
        (10, 14, [13, 11, 9, 6]),
    ]

    for ev in tests:
        print(ev, "->", format_hand(ev, tiebreak=False))
    print("\nWith tiebreak detail:\n")
    for ev in tests:
        print(ev, "->", format_hand(ev, tiebreak=True))
