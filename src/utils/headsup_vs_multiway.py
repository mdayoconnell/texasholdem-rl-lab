"""

Reads the wide preflop CSV produced by `probability_engine/preflop.py` (e.g.
`preflop_calculated_p2to6_20k.csv`) and prints:
  - Top-N hands by equity for each player count present (p2..p6)
  - Biggest risers/fallers from heads-up (p2) to multiway (p6), in terms of both rank and equity

"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class HandRow:
    hand: str
    equity_by_p: Dict[int, float]


def read_preflop_csv(csv_path: Path) -> List[HandRow]:
    """Read a wide preflop CSV produced by probability_engine/preflop.py"""
    rows: List[HandRow] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")

        # Detect player-count equity columns
        ps: List[int] = []
        for name in reader.fieldnames:
            if name.startswith("p") and name.endswith("_equity"):
                # e.g. p2_equity
                try:
                    p = int(name[1 : name.index("_")])
                    ps.append(p)
                except Exception:
                    pass
        ps = sorted(set(ps))
        if not ps:
            raise ValueError("No columns of the form pX_equity found")

        for r in reader:
            hand = (r.get("hand") or "").strip()
            if not hand:
                continue
            equity_by_p: Dict[int, float] = {}
            for p in ps:
                val = r.get(f"p{p}_equity")
                if val is None or val == "":
                    continue
                equity_by_p[p] = float(val)
            rows.append(HandRow(hand=hand, equity_by_p=equity_by_p))

    return rows


def rank_by_equity(rows: List[HandRow], p: int) -> List[Tuple[str, float]]: # Convention to rank by p2 equity 
    # rankings are not consistent across p2-p6, as shown by heatmap
    # Return [(hand, equity)] sorted descending for a given player-count p.
    pairs = [(hr.hand, float(hr.equity_by_p[p])) for hr in rows if p in hr.equity_by_p]
    return sorted(pairs, key=lambda t: t[1], reverse=True)


def print_top(rows: List[HandRow], p: int, top_n: int) -> None:
    ranked = rank_by_equity(rows, p)
    print(f"\nTop {top_n} hands by equity (p{p})")
    for i, (h, e) in enumerate(ranked[:top_n], start=1):
        print(f"{i:2d}. {h:>4s}  {e:.6f}")


def rank_map(rows: List[HandRow], p: int) -> Dict[str, int]:
    ranked = rank_by_equity(rows, p)
    return {h: i for i, (h, _e) in enumerate(ranked, start=1)}


def print_biggest_movers(rows: List[HandRow], p_from: int, p_to: int, k: int) -> None:
    # Rank movers: positive delta means improved rank when going p_from -> p_to.
    r_from = rank_map(rows, p_from)
    r_to = rank_map(rows, p_to)

    common = [h for h in r_from.keys() if h in r_to]
    deltas = [(h, r_from[h] - r_to[h]) for h in common]

    risers = sorted(deltas, key=lambda t: t[1], reverse=True)[:k]
    fallers = sorted(deltas, key=lambda t: t[1])[:k]

    print(f"\nBiggest rank risers from p{p_from} -> p{p_to} (top {k})")
    for h, d in risers:
        print(f"{h:>4s}  delta_rank={d:+d}   (p{p_from} rank {r_from[h]} -> p{p_to} rank {r_to[h]})")

    print(f"\nBiggest rank fallers from p{p_from} -> p{p_to} (top {k})")
    for h, d in fallers:
        print(f"{h:>4s}  delta_rank={d:+d}   (p{p_from} rank {r_from[h]} -> p{p_to} rank {r_to[h]})")


def print_smallest_equity_drop(
    rows: List[HandRow],
    p_from: int,
    p_to: int,
    k: int,
) -> None:
    #---
    # Print hands with smallest equity drop from p_from to p_to.

    #drop = equity(p_from) - equity(p_to). Smaller drop => more multiway-resilient.
    #---

    deltas: List[Tuple[str, float, float, float]] = []  # (hand, drop, eq_from, eq_to)
    for hr in rows:
        if p_from in hr.equity_by_p and p_to in hr.equity_by_p:
            eq_from = float(hr.equity_by_p[p_from])
            eq_to = float(hr.equity_by_p[p_to])
            drop = eq_from - eq_to
            deltas.append((hr.hand, drop, eq_from, eq_to))

    deltas_sorted = sorted(deltas, key=lambda t: t[1])

    print(f"\nSmallest equity drop from p{p_from} -> p{p_to} (top {k})")
    for i, (h, d, e_from, e_to) in enumerate(deltas_sorted[:k], start=1):
        print(f"{i:2d}. {h:>4s}  drop={d:.6f}   p{p_from}={e_from:.6f}  p{p_to}={e_to:.6f}")


def print_largest_equity_drop(
    rows: List[HandRow],
    p_from: int,
    p_to: int,
    k: int,
) -> None:
    #---
    #Print hands with largest equity drop from p_from to p_to.
    #---
    deltas: List[Tuple[str, float, float, float]] = []
    for hr in rows:
        if p_from in hr.equity_by_p and p_to in hr.equity_by_p:
            eq_from = float(hr.equity_by_p[p_from])
            eq_to = float(hr.equity_by_p[p_to])
            drop = eq_from - eq_to
            deltas.append((hr.hand, drop, eq_from, eq_to))

    deltas_sorted = sorted(deltas, key=lambda t: t[1], reverse=True)

    print(f"\nLargest equity drop from p{p_from} -> p{p_to} (top {k})")
    for i, (h, d, e_from, e_to) in enumerate(deltas_sorted[:k], start=1):
        print(f"{i:2d}. {h:>4s}  drop={d:.6f}   p{p_from}={e_from:.6f}  p{p_to}={e_to:.6f}")


def main() -> None:
    default_csv = (
        Path(__file__).resolve().parents[1]
        / "probability_engine"
        / "preflop_calculated_p2to6_20k.csv"
    )

    parser = argparse.ArgumentParser(description="Heads-up vs multiway preflop analysis")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(default_csv),
        help="Path to preflop_calculated_*.csv",
    )
    parser.add_argument("--top", type=int, default=20, help="Top-N to print")
    parser.add_argument("--movers", type=int, default=20, help="How many rank movers to print")
    parser.add_argument(
        "--drop",
        type=int,
        default=20,
        help="How many hands to show for smallest/largest equity drop",
    )
    parser.add_argument(
        "--drop-from",
        type=int,
        default=2,
        help="Player count to measure drop from (default 2)",
    )
    parser.add_argument(
        "--drop-to",
        type=int,
        default=6,
        help="Player count to measure drop to (default 6)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    rows = read_preflop_csv(csv_path)

    ps = sorted({p for hr in rows for p in hr.equity_by_p.keys()})
    print(f"Read {len(rows)} hands from {csv_path}")
    print(f"Detected equity columns for players: {ps}")

    for p in ps:
        print_top(rows, p, args.top)

    p_from = 2 if 2 in ps else min(ps)
    p_to = 6 if 6 in ps else max(ps)
    if p_from != p_to:
        print_biggest_movers(rows, p_from, p_to, args.movers)

    # Equity drop analysis (defaults p2->p6; if those columns don't exist, fall back to min->max)
    d_from = args.drop_from if args.drop_from in ps else min(ps)
    d_to = args.drop_to if args.drop_to in ps else max(ps)
    if d_from != d_to:
        print_smallest_equity_drop(rows, d_from, d_to, args.drop)
        print_largest_equity_drop(rows, d_from, d_to, args.drop)


if __name__ == "__main__":
    main()
