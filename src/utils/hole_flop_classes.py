"""Generate canonical hole+flop classes, quotienting by suit equivalence and rank ties."""

import argparse
import csv
import sys
from itertools import combinations, permutations
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

# Ensure `src/` is on sys.path so sibling packages can be imported
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

Card = Tuple[int, int]  # (rank, suit) with suit in 0..3
Key = Tuple[int, int, int, int, int, int, int, int, int, int]

def _build_deck() -> List[Card]:
    # Match init_deck.create_deck ordering: suits first, then ranks 2..14.
    return [(rank, suit) for suit in range(4) for rank in range(2, 15)]

RANK_TO_CHAR = {
    14: "A",
    13: "K",
    12: "Q",
    11: "J",
    10: "T",
    9: "9",
    8: "8",
    7: "7",
    6: "6",
    5: "5",
    4: "4",
    3: "3",
    2: "2",
}

def _encode_suits(suits_seq: Sequence[int]) -> Tuple[int, ...]:
    mapping = {}
    next_id = 0
    encoded: List[int] = []
    for s in suits_seq:
        if s not in mapping:
            mapping[s] = next_id
            next_id += 1
        encoded.append(mapping[s])
    return tuple(encoded)


def _hole_orders(hole_cards: Sequence[Card]) -> List[List[Card]]:
    r1, r2 = hole_cards[0][0], hole_cards[1][0]
    if r1 != r2:
        return [sorted(hole_cards, key=lambda c: c[0], reverse=True)]
    # Pair: order shouldn't depend on suit labels, so we allow both and take min later.
    return [list(hole_cards), [hole_cards[1], hole_cards[0]]]


def _flop_orders(flop_cards: Sequence[Card]) -> List[List[Card]]:
    ranks = [c[0] for c in flop_cards]
    unique = set(ranks)
    if len(unique) == 3:
        return [sorted(flop_cards, key=lambda c: c[0], reverse=True)]
    if len(unique) == 1:
        # Trips: try all permutations (3! = 6).
        return [list(p) for p in permutations(flop_cards, 3)]

    # One pair + kicker
    # Identify pair rank and kicker rank.
    r1, r2, r3 = ranks
    if r1 == r2:
        pair_rank = r1
        kicker_rank = r3
    elif r1 == r3:
        pair_rank = r1
        kicker_rank = r2
    else:
        pair_rank = r2
        kicker_rank = r1

    pair_cards = [c for c in flop_cards if c[0] == pair_rank]
    kicker_card = [c for c in flop_cards if c[0] == kicker_rank][0]
    pair_perms = [pair_cards, [pair_cards[1], pair_cards[0]]]

    orders: List[List[Card]] = []
    if pair_rank > kicker_rank:
        for p in pair_perms:
            orders.append(p + [kicker_card])
    else:
        for p in pair_perms:
            orders.append([kicker_card] + p)
    return orders


def _canonical_key(hole_cards: Sequence[Card], flop_cards: Sequence[Card]) -> Key:
    hole_opts = _hole_orders(hole_cards)
    flop_opts = _flop_orders(flop_cards)

    best: Optional[Key] = None
    for h in hole_opts:
        for f in flop_opts:
            seq = h + f
            ranks_seq = tuple(c[0] for c in seq)
            suits_seq = _encode_suits([c[1] for c in seq])
            key = ranks_seq + suits_seq  # type: ignore[operator]
            if best is None or key < best:
                best = key

    if best is None:
        raise RuntimeError("Failed to compute canonical key.")
    return best


def _key_to_row(key: Key) -> List[str]:
    ranks = key[:5]
    suits = key[5:]
    rank_chars = [RANK_TO_CHAR[r] for r in ranks]
    return [
        rank_chars[0],
        rank_chars[1],
        rank_chars[2],
        rank_chars[3],
        rank_chars[4],
        str(suits[0]),
        str(suits[1]),
        str(suits[2]),
        str(suits[3]),
        str(suits[4]),
        "".join(rank_chars),
        "".join(str(s) for s in suits),
    ]


def generate_classes(
    max_holes: Optional[int] = None,
    max_flops: Optional[int] = None,
    progress_every: int = 50,
) -> Set[Key]:
    cards = _build_deck()
    all_cards = list(range(len(cards)))
    classes: Set[Key] = set()

    holes = list(combinations(all_cards, 2))
    if max_holes is not None:
        holes = holes[: max_holes]

    for h_idx, (h1, h2) in enumerate(holes, start=1):
        hole_cards = [cards[h1], cards[h2]]
        remaining = [c for c in all_cards if c != h1 and c != h2]

        flops_iter: Iterable[Tuple[int, int, int]] = combinations(remaining, 3)
        if max_flops is not None:
            flops_iter = list(flops_iter)[: max_flops]

        for f1, f2, f3 in flops_iter:
            flop_cards = [cards[f1], cards[f2], cards[f3]]
            key = _canonical_key(hole_cards, flop_cards)
            classes.add(key)

        if progress_every > 0 and (h_idx % progress_every == 0 or h_idx == len(holes)):
            print(f"Processed holes {h_idx}/{len(holes)} | classes={len(classes):,}")

    return classes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate suit-texture + rank-tie canonical hole+flop classes."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "reports" / "hole_flop_classes.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--max-holes", type=int, default=None, help="Limit number of hole combos.")
    parser.add_argument("--max-flops", type=int, default=None, help="Limit number of flops per hole.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N holes (0 to disable).",
    )
    parser.add_argument("--no-sort", action="store_true", help="Do not sort classes before writing.")
    args = parser.parse_args()

    classes = generate_classes(
        max_holes=args.max_holes,
        max_flops=args.max_flops,
        progress_every=args.progress_every,
    )

    rows = list(classes)
    if not args.no_sort:
        rows.sort()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "h1_rank",
                "h2_rank",
                "f1_rank",
                "f2_rank",
                "f3_rank",
                "h1_suit",
                "h2_suit",
                "f1_suit",
                "f2_suit",
                "f3_suit",
                "rank_key",
                "suit_key",
            ]
        )
        for key in rows:
            writer.writerow(_key_to_row(key))

    print(f"Wrote {len(rows):,} classes to {args.out}")


if __name__ == "__main__":
    main()
