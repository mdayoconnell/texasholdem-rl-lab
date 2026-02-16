"""Monte Carlo equity for canonical hole+flop classes (cloud worker script)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure `src/` is on sys.path so sibling packages can be imported
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from game_engine.init_deck import create_deck
from game_engine.evaluate_hands import evaluate_hand, evaluate_winner

RANK_CHAR_TO_INT: Dict[str, int] = {
    "A": 14,
    "K": 13,
    "Q": 12,
    "J": 11,
    "T": 10,
    "9": 9,
    "8": 8,
    "7": 7,
    "6": 6,
    "5": 5,
    "4": 4,
    "3": 3,
    "2": 2,
}


def _z_for_ci(ci: float) -> float:
    z_table = {
        0.90: 1.644854,
        0.95: 1.959964,
        0.98: 2.326348,
        0.99: 2.575829,
    }
    z = z_table.get(round(ci, 2))
    if z is None:
        raise ValueError(f"Unsupported ci={ci}. Supported: {sorted(z_table.keys())}")
    return z


def _remove_cards_from_deck(deck: np.ndarray, cards: np.ndarray) -> np.ndarray:
    if deck.ndim != 2 or deck.shape[0] != 2:
        raise ValueError(f"deck must have shape (2,N); got {deck.shape}")
    if cards.ndim != 2 or cards.shape[0] != 2:
        raise ValueError(f"cards must have shape (2,k); got {cards.shape}")

    keep = np.ones(deck.shape[1], dtype=bool)
    for j in range(cards.shape[1]):
        v = int(cards[0, j])
        s = int(cards[1, j])
        matches = np.where((deck[0] == v) & (deck[1] == s) & keep)[0]
        if matches.size == 0:
            raise ValueError(f"Card ({v},{s}) not found in deck.")
        keep[matches[0]] = False

    return deck[:, keep]


def _parse_rank(s: str) -> int:
    s = s.strip().upper()
    if s.isdigit():
        return int(s)
    if s in RANK_CHAR_TO_INT:
        return RANK_CHAR_TO_INT[s]
    raise ValueError(f"Invalid rank value: {s!r}")


def _row_to_cards(row: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    ranks = [
        _parse_rank(row["h1_rank"]),
        _parse_rank(row["h2_rank"]),
        _parse_rank(row["f1_rank"]),
        _parse_rank(row["f2_rank"]),
        _parse_rank(row["f3_rank"]),
    ]
    suits_enc = [
        int(row["h1_suit"]),
        int(row["h2_suit"]),
        int(row["f1_suit"]),
        int(row["f2_suit"]),
        int(row["f3_suit"]),
    ]

    # Map encoded suits 0..3 to actual suits 1..4.
    suit_map = {0: 1, 1: 2, 2: 3, 3: 4}
    suits = [suit_map[s] for s in suits_enc]

    cards = np.asarray([ranks, suits], dtype=int)  # shape (2,5)
    hole = cards[:, :2]
    flop = cards[:, 2:]

    # Sanity: no duplicate exact cards.
    if len({(int(cards[0, i]), int(cards[1, i])) for i in range(5)}) != 5:
        raise ValueError(f"Row maps to duplicate cards: ranks={ranks} suits={suits}")

    return hole, flop


def _default_progress_path(output_csv: Path) -> Path:
    # e.g. results.csv -> results.progress.json
    return output_csv.with_suffix(output_csv.suffix + ".progress.json")


def _read_rows_done(progress_path: Path) -> int:
    if not progress_path.exists():
        return 0
    try:
        with progress_path.open("r") as f:
            data = json.load(f)
        rows_done = int(data.get("rows_done", 0))
        return max(rows_done, 0)
    except Exception:
        # If progress file is corrupted, fall back to 0 and let CSV row-count resume handle it.
        return 0


def _count_output_rows_done(output_csv: Path) -> int:
    # Counts completed data rows in output CSV (excluding header).
    if not output_csv.exists():
        return 0
    with output_csv.open("r", newline="") as f:
        # Count lines; subtract 1 for header if file has content.
        lines = sum(1 for _ in f)
    return max(lines - 1, 0)


def _write_progress_atomic(progress_path: Path, rows_done: int) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    payload = {
        "rows_done": int(rows_done),
    }
    with tmp_path.open("w") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(progress_path)


def simulate_flop_equity(
    hole: np.ndarray,
    flop: np.ndarray,
    n_players: int,
    rng: np.random.Generator,
    max_sims: int,
    min_sims: int,
    z: float,
    target_half_width: float,
) -> Dict[str, float]:
    if n_players < 2:
        raise ValueError("n_players must be >= 2")
    if max_sims <= 0:
        raise ValueError("max_sims must be > 0")
    if min_sims < 1:
        raise ValueError("min_sims must be >= 1")

    deck = create_deck()
    known = np.hstack([hole, flop])
    remaining = _remove_cards_from_deck(deck, known)

    opp_cards = 2 * (n_players - 1)
    draw_n = opp_cards + 2  # opponents + turn/river

    n = 0
    mean = 0.0
    m2 = 0.0
    win = 0
    tie = 0
    loss = 0

    while n < max_sims:
        idx = rng.choice(remaining.shape[1], size=draw_n, replace=False)
        drawn = remaining[:, idx]
        opp_holes = drawn[:, :opp_cards]
        turn_river = drawn[:, opp_cards:]
        community = np.hstack([flop, turn_river])

        hero_eval = evaluate_hand(np.hstack([hole, community]))
        evals = [hero_eval]
        for i in range(n_players - 1):
            opp = opp_holes[:, 2 * i : 2 * i + 2]
            evals.append(evaluate_hand(np.hstack([opp, community])))

        winners, _ = evaluate_winner(evals)
        if 0 in winners:
            if len(winners) == 1:
                win += 1
            else:
                tie += 1
            share = 1.0 / len(winners)
        else:
            loss += 1
            share = 0.0

        n += 1
        delta = share - mean
        mean += delta / n
        m2 += delta * (share - mean)

        if n >= min_sims and n >= 2:
            var = m2 / (n - 1)
            half_width = z * math.sqrt(var / n)
            if half_width <= target_half_width:
                break

    if n >= 2:
        var = m2 / (n - 1)
        half_width = z * math.sqrt(var / n)
    else:
        half_width = float("inf")

    win_rate = win / n
    tie_rate = tie / n
    loss_rate = loss / n

    return {
        "n_sims": n,
        "equity": mean,
        "win_rate": win_rate,
        "tie_rate": tie_rate,
        "loss_rate": loss_rate,
        "ci_half_width": half_width,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MC equity on hole+flop classes CSV.")
    parser.add_argument("--input", type=Path, required=True, help="Input classes CSV (one part).")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV with equities.")
    parser.add_argument("--players", type=int, default=2, help="Total players at table (>=2).")
    parser.add_argument("--max-sims", type=int, default=20_000, help="Max sims per class.")
    parser.add_argument("--min-sims", type=int, default=1_000, help="Min sims before CI stop.")
    parser.add_argument("--ci", type=float, default=0.98, help="CI level (0.90, 0.95, 0.98, 0.99).")
    parser.add_argument("--ci-half-width", type=float, default=0.01, help="Target half-width for equity.")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed.")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows.")
    parser.add_argument("--progress-every", type=int, default=1000, help="Print progress every N rows.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=200,
        help="Flush output and write progress JSON every N rows.",
    )
    parser.add_argument(
        "--progress-json",
        type=Path,
        default=None,
        help="Optional progress JSON path (default: <output>.progress.json).",
    )
    args = parser.parse_args()

    z = _z_for_ci(args.ci)
    rng = np.random.default_rng(args.seed)

    if not args.input.exists():
        raise SystemExit(f"Missing input CSV: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    progress_path = args.progress_json or _default_progress_path(args.output)

    # Determine resume point.
    rows_done_json = _read_rows_done(progress_path)
    rows_done_csv = _count_output_rows_done(args.output)
    rows_done = max(rows_done_json, rows_done_csv)

    if rows_done > 0:
        print(f"Resuming: {rows_done} rows already completed (progress={progress_path})")

    # If --limit is set, interpret it as a TOTAL row limit (including already-completed rows).
    stop_after_total: Optional[int]
    if args.limit is None:
        stop_after_total = None
    else:
        stop_after_total = int(args.limit)
        if stop_after_total < 0:
            raise SystemExit("--limit must be >= 0")

    out_exists = args.output.exists()
    out_mode = "a" if out_exists else "w"

    with args.input.open("r", newline="") as f_in, args.output.open(out_mode, newline="") as f_out:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise SystemExit("Input CSV missing header.")

        base_cols = list(reader.fieldnames)
        out_cols = base_cols + [
            "n_sims",
            "equity",
            "win_rate",
            "tie_rate",
            "loss_rate",
            "ci_half_width",
        ]
        writer = csv.DictWriter(f_out, fieldnames=out_cols)

        # If we're starting a new file, write header.
        if not out_exists:
            writer.writeheader()
            f_out.flush()
            os.fsync(f_out.fileno())

        # Skip already-completed input rows.
        skipped = 0
        while skipped < rows_done:
            try:
                next(reader)
            except StopIteration:
                break
            skipped += 1

        if skipped > 0:
            print(f"Skipped {skipped} input rows (already completed)")

        processed_total = skipped

        if stop_after_total is not None and processed_total >= stop_after_total:
            # Nothing to do; still ensure progress reflects current state.
            f_out.flush()
            os.fsync(f_out.fileno())
            _write_progress_atomic(progress_path, processed_total)
            print(f"Limit reached ({stop_after_total}); nothing to process.")
            return

        for local_idx, row in enumerate(reader, start=1):
            processed_total += 1

            hole, flop = _row_to_cards(row)
            stats = simulate_flop_equity(
                hole=hole,
                flop=flop,
                n_players=args.players,
                rng=rng,
                max_sims=args.max_sims,
                min_sims=args.min_sims,
                z=z,
                target_half_width=args.ci_half_width,
            )

            out_row = dict(row)
            out_row["n_sims"] = str(stats["n_sims"])
            out_row["equity"] = f"{stats['equity']:.6f}"
            out_row["win_rate"] = f"{stats['win_rate']:.6f}"
            out_row["tie_rate"] = f"{stats['tie_rate']:.6f}"
            out_row["loss_rate"] = f"{stats['loss_rate']:.6f}"
            out_row["ci_half_width"] = f"{stats['ci_half_width']:.6f}"
            writer.writerow(out_row)

            # Periodic progress prints.
            if args.progress_every > 0 and (processed_total % args.progress_every == 0):
                print(f"Processed {processed_total} rows")

            # Periodic checkpointing: flush output + write progress JSON.
            if args.checkpoint_every > 0 and (processed_total % args.checkpoint_every == 0):
                f_out.flush()
                os.fsync(f_out.fileno())
                _write_progress_atomic(progress_path, processed_total)

            if stop_after_total is not None and processed_total >= stop_after_total:
                break

        # Final checkpoint
        f_out.flush()
        os.fsync(f_out.fileno())
        _write_progress_atomic(progress_path, processed_total)

    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()

