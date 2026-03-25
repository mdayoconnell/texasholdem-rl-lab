from pathlib import Path
from typing import Dict, List, Optional
import time
import numpy as np

from ..preflop import (
    PREFLOP_HANDS,
    simulate_preflop_equity,
    _write_preflop_matrix_csv,
    _plot_equity_heatmap,
    HAVE_MPL,
)
"""Flop probability engine (placeholder).

This module will hold flop/turn estimators. Not implemented yet.
"""


def main() -> None:
    print("flop.py: not implemented yet")


if __name__ == "__main__":
    main()

def run_preflop_bulk_single_players(
    players: int,
    n_sims: int,
    seed: int,
    out_csv: Path,
    out_png: Optional[Path] = None,
) -> None:
    """Compute preflop stats for ALL 169 hands with a fixed player count."""

    total = len(PREFLOP_HANDS)
    stats_by_hand: Dict[str, Dict[str, float]] = {}

    for idx, key in enumerate(PREFLOP_HANDS):
        print(f"Now computing hand {idx+1}/{total}: {key}")
        res = simulate_preflop_equity(key, n_players=players, n_sims=n_sims, seed=seed)
        stats_by_hand[key] = res
        if (idx + 1) % 5 == 0:
            print(f"  ...computed {idx+1}/{total} hands")

    # Sort by equity
    hands_sorted = sorted(
        PREFLOP_HANDS,
        key=lambda h: float(stats_by_hand[h]["equity"]),
        reverse=True,
    )

    # Write CSV
    _write_preflop_matrix_csv(out_csv, hands_sorted, [players], {h: {players: stats_by_hand[h]} for h in hands_sorted})

    # Heatmap
    if out_png is not None:
        if not HAVE_MPL:
            raise RuntimeError("matplotlib is not installed in the active environment")

        equity_matrix = np.asarray(
            [[float(stats_by_hand[h]["equity"])] for h in hands_sorted],
            dtype=float,
        )
        _plot_equity_heatmap(
            out_png,
            hands_sorted,
            [players],
            equity_matrix,
            title=f"Preflop equity heatmap (players={players}, sims={n_sims})",
        )

    # Ranked text output
    ranked_txt = out_csv.with_suffix(".ranked.txt")
    with ranked_txt.open("w", encoding="utf-8") as g:
        for i, h in enumerate(hands_sorted, start=1):
            base = stats_by_hand[h]
            g.write(
                f"{i:3d}. {h}: equity={float(base['equity']):.6f} "
                f"(win={float(base['win_rate']):.6f}, tie={float(base['tie_rate']):.6f})\n"
            )


def run_preflop_bulk_multi_players(
    players_list: List[int],
    n_sims: int,
    seed: int,
    out_csv: Path,
    out_png: Optional[Path] = None,
    sort_by_players: int = 2,
) -> None:
    """Compute preflop stats for ALL 169 hands across multiple player-counts.

    - Computes for each hand and each `p in players_list`.
    - Sorts hands by equity at `sort_by_players` (default p=2) for ranking/plotting.
    - Writes a wide CSV with 4 stats per player-count.
    - Optionally saves a (169 x len(players_list)) equity heatmap.
    """

    if any(p < 2 for p in players_list):
        raise ValueError("All entries in players_list must be >= 2")
    if sort_by_players not in players_list:
        raise ValueError("sort_by_players must be one of players_list")

    total = len(PREFLOP_HANDS)
    stats_by_hand: Dict[str, Dict[int, Dict[str, float]]] = {}

    for idx, key in enumerate(PREFLOP_HANDS):
        print(f"Now computing hand {idx+1}/{total}: {key}")
        stats_by_hand[key] = {}
        for p in players_list:
            res = simulate_preflop_equity(key, n_players=p, n_sims=n_sims, seed=seed)
            stats_by_hand[key][p] = res
        if (idx + 1) % 5 == 0:
            print(f"  ...computed {idx+1}/{total} hands")

    # Sort by equity at the chosen player-count
    hands_sorted = sorted(
        PREFLOP_HANDS,
        key=lambda h: float(stats_by_hand[h][sort_by_players]["equity"]),
        reverse=True,
    )

    # Write CSV (sorted)
    _write_preflop_matrix_csv(out_csv, hands_sorted, players_list, stats_by_hand)

    # Heatmap (sorted)
    if out_png is not None:
        if not HAVE_MPL:
            raise RuntimeError("matplotlib is not installed in the active environment")

        equity_matrix = np.asarray(
            [[float(stats_by_hand[h][p]["equity"]) for p in players_list] for h in hands_sorted],
            dtype=float,
        )
        _plot_equity_heatmap(
            out_png,
            hands_sorted,
            players_list,
            equity_matrix,
            title=f"Preflop equity heatmap (sims={n_sims}, sorted by p{sort_by_players})",
        )

    # Ranked text output (by the sort column)
    ranked_txt = out_csv.with_suffix(".ranked.txt")
    with ranked_txt.open("w", encoding="utf-8") as g:
        for i, h in enumerate(hands_sorted, start=1):
            base = stats_by_hand[h][sort_by_players]
            g.write(
                f"{i:3d}. {h}: p{sort_by_players} equity={float(base['equity']):.6f} "
                f"(win={float(base['win_rate']):.6f}, tie={float(base['tie_rate']):.6f})\n"
            )


def main() -> None:
    # Bulk run: all 169 hands, opponents=1..5  (i.e. players=2..6)
    OPPONENTS_LIST = [1, 2, 3, 4, 5]
    PLAYERS_LIST = [o + 1 for o in OPPONENTS_LIST]
    N_SIMS = 20_000
    SEED = 10

    t0 = time.perf_counter()

    csv_path = Path(__file__).resolve().with_name("preflop_calculated_p2to6_20k.csv")
    png_path = Path(__file__).resolve().with_name("preflop_equity_heatmap_p2to6_20k.png")

    print(
        f"Running bulk preflop: players={PLAYERS_LIST} | sims={N_SIMS} per hand | hands={len(PREFLOP_HANDS)}"
    )

    run_preflop_bulk_multi_players(
        players_list=PLAYERS_LIST,
        n_sims=N_SIMS,
        seed=SEED,
        out_csv=csv_path,
        out_png=png_path,
        sort_by_players=2,
    )

    t1 = time.perf_counter()
    print("Wrote:", csv_path)
    print("Wrote:", csv_path.with_suffix(".ranked.txt"))
    print("Wrote:", png_path)
    print(f"Done in {t1 - t0:.2f}s")


if __name__ == "__main__":
    main()
