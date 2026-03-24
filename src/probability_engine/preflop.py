from __future__ import annotations

import numpy as np
import csv
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

# Matplotlib is optional; if it's not installed in the active env, we'll skip plotting.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ModuleNotFoundError:
    HAVE_MPL = False

# Ensure `src/` is on sys.path so sibling packages like `utils/` can be imported
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from game_engine.init_deck import create_deck
from game_engine.evaluate_hands import hand_plus_community, evaluate_hand, evaluate_winner



# -----------------------------------------------------------------------------
# Preflop indexing (169 hand classes) + Monte Carlo equity
# -----------------------------------------------------------------------------

RANKS_DESC: List[int] = list(range(14, 1, -1))  # A .. 2
RANK_TO_CHAR: Dict[int, str] = {
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
CHAR_TO_RANK: Dict[str, int] = {v: k for k, v in RANK_TO_CHAR.items()}


def build_preflop_hand_table() -> List[str]:
    """Return the canonical list of 169 preflop hand classes.

    Convention (common in poker tools):
      - Pairs: AA, KK, ..., 22
      - Suited: AKs, AQs, ..., 32s (high card descending, then low card)
      - Offsuit: AKo, AQo, ..., 32o

    This gives 13 + 78 + 78 = 169.
    """
    hands: List[str] = []

    # 13 pairs
    for r in RANKS_DESC:
        c = RANK_TO_CHAR[r]
        hands.append(f"{c}{c}")

    # 78 suited (r1 > r2)
    for r1 in RANKS_DESC:
        for r2 in RANKS_DESC:
            if r1 <= r2:
                continue
            c1, c2 = RANK_TO_CHAR[r1], RANK_TO_CHAR[r2]
            hands.append(f"{c1}{c2}s")

    # 78 offsuit (r1 > r2)
    for r1 in RANKS_DESC:
        for r2 in RANKS_DESC:
            if r1 <= r2:
                continue
            c1, c2 = RANK_TO_CHAR[r1], RANK_TO_CHAR[r2]
            hands.append(f"{c1}{c2}o")

    return hands


# Module-level caches
PREFLOP_HANDS: List[str] = build_preflop_hand_table()
PREFLOP_HAND_TO_ID: Dict[str, int] = {h: i for i, h in enumerate(PREFLOP_HANDS)}


def hole_to_preflop_key(hole_cards: np.ndarray) -> str:
    """Convert a 2-card (2,2) array into a canonical preflop key.

    Expected card encoding:
      - hole_cards[0] are values (2..14)
      - hole_cards[1] are suits (1..4)

    Returns keys like: "AA", "AKs", "AKo", "T9s", etc.
    """
    if hole_cards.shape != (2, 2):
        raise ValueError(f"hole_cards must have shape (2,2); got {hole_cards.shape}")

    v1, v2 = int(hole_cards[0, 0]), int(hole_cards[0, 1])
    s1, s2 = int(hole_cards[1, 0]), int(hole_cards[1, 1])

    # Order by rank (high first)
    if v2 > v1:
        v1, v2 = v2, v1
        s1, s2 = s2, s1

    if v1 == v2:
        c = RANK_TO_CHAR[v1]
        return f"{c}{c}"

    suited = (s1 == s2)
    c1, c2 = RANK_TO_CHAR[v1], RANK_TO_CHAR[v2]
    return f"{c1}{c2}{'s' if suited else 'o'}"


def preflop_id_from_hole(hole_cards: np.ndarray) -> int:
    """Map a 2-card (2,2) array to its 0..168 class id."""
    key = hole_to_preflop_key(hole_cards)
    try:
        return PREFLOP_HAND_TO_ID[key]
    except KeyError as e:
        raise KeyError(f"Unknown preflop key '{key}'.") from e


def representative_hole_from_key(key: str) -> np.ndarray:
    """Return a representative (2,2) hole-cards array for a given preflop key.

    Suits convention (consistent, avoids duplicates):
      - pairs: use suits (1,2)
      - suited: both suit 1
      - offsuit: suits (1,2)

    Assumes suits are encoded as integers 1..4.
    """
    key = key.strip().upper()
    if len(key) == 2:  # pair, e.g. "AA"
        r = CHAR_TO_RANK[key[0]]
        return np.asarray([[r, r], [1, 2]], dtype=int)

    if len(key) != 3 or key[2] not in ("S", "O"):
        raise ValueError(f"Invalid preflop key '{key}'. Expected like 'AKs', 'AKo', or 'QQ'.")

    r1 = CHAR_TO_RANK[key[0]]
    r2 = CHAR_TO_RANK[key[1]]
    if r2 > r1:
        r1, r2 = r2, r1

    if key[2] == "S":
        return np.asarray([[r1, r2], [1, 1]], dtype=int)

    # offsuit
    return np.asarray([[r1, r2], [1, 2]], dtype=int)


def _remove_cards_from_deck(deck: np.ndarray, cards: np.ndarray) -> np.ndarray:
    """Remove specific cards (2,k) from a deck (2,N) by exact column match."""
    if deck.ndim != 2 or deck.shape[0] != 2:
        raise ValueError(f"deck must have shape (2,N); got {deck.shape}")
    if cards.ndim != 2 or cards.shape[0] != 2:
        raise ValueError(f"cards must have shape (2,k); got {cards.shape}")

    keep = np.ones(deck.shape[1], dtype=bool)
    # For each card to remove, drop the first matching column
    for j in range(cards.shape[1]):
        v = int(cards[0, j])
        s = int(cards[1, j])
        matches = np.where((deck[0] == v) & (deck[1] == s) & keep)[0]
        if matches.size == 0:
            raise ValueError(f"Card ({v},{s}) not found in deck.")
        keep[matches[0]] = False

    return deck[:, keep]


def simulate_preflop_equity(
    hero: Union[str, int, np.ndarray],
    n_players: int = 2,
    n_sims: int = 20_000,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Monte Carlo preflop equity for a hero hand class.

    Parameters
    ----------
    hero:
      - preflop key (e.g. "AKs", "QQ", "T9o")
      - preflop id (0..168)
      - explicit (2,2) hole-cards array
    n_players:
      Total players at the table (hero is player 0). Must be >= 2.
    n_sims:
      Number of Monte Carlo simulations.
    seed:
      Optional RNG seed.

    Returns
    -------
    dict with keys:
      - equity: average pot share (win=1, split=1/k, lose=0)
      - win_rate: fraction hero is sole winner
      - tie_rate: fraction hero ties for best hand (including multi-way)
      - loss_rate: fraction hero loses
    """
    if n_players < 2:
        raise ValueError("n_players must be >= 2")
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")

    rng = np.random.default_rng(seed)

    # Resolve hero hole cards
    if isinstance(hero, np.ndarray):
        hero_hole = hero
    elif isinstance(hero, int):
        if hero < 0 or hero >= len(PREFLOP_HANDS):
            raise ValueError(f"preflop id must be in [0,{len(PREFLOP_HANDS)-1}]")
        hero_hole = representative_hole_from_key(PREFLOP_HANDS[hero])
    elif isinstance(hero, str):
        hero_hole = representative_hole_from_key(hero)
    else:
        raise TypeError("hero must be a preflop key (str), id (int), or (2,2) numpy array")

    deck = create_deck()  # expected shape (2,52)
    deck_rem = _remove_cards_from_deck(deck, hero_hole)

    need = 2 * (n_players - 1) + 5
    if deck_rem.shape[1] < need:
        raise ValueError("Not enough cards remaining in deck for requested n_players")

    wins = 0
    ties = 0
    losses = 0
    equity_sum = 0.0

    for _ in range(n_sims):
        pick_idx = rng.choice(deck_rem.shape[1], size=need, replace=False)
        picked = deck_rem[:, pick_idx]

        opp = picked[:, : 2 * (n_players - 1)]
        community = picked[:, 2 * (n_players - 1) :]

        # Build drawn matrix in the format your engine expects:
        # [P0 hole | P1 hole | ... | P(n-1) hole | 5 community]
        drawn = np.hstack([hero_hole, opp, community])

        evals = []
        for i in range(n_players):
            seven = hand_plus_community(i, drawn)
            evals.append(evaluate_hand(seven))

        winners, _best_ev = evaluate_winner(evals)

        if 0 in winners:
            if len(winners) == 1:
                wins += 1
                equity_sum += 1.0
            else:
                ties += 1
                equity_sum += 1.0 / len(winners)
        else:
            losses += 1

    return {
        "equity": float(equity_sum / n_sims),
        "win_rate": float(wins / n_sims),
        "tie_rate": float(ties / n_sims),
        "loss_rate": float(losses / n_sims),
    }


def _write_preflop_matrix_csv(
    out_path: Path,
    hand_keys: List[str],
    players_list: List[int],
    stats_by_hand: Dict[str, Dict[int, Dict[str, float]]],
) -> None:
    """Write a wide CSV with one row per hand and 4 stats per player-count.

    Columns:
      hand,
      p2_equity,p2_win,p2_tie,p2_loss,
      ...,
      p6_equity,p6_win,p6_tie,p6_loss
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header: List[str] = ["hand"]
    for p in players_list:
        header.extend([
            f"p{p}_equity",
            f"p{p}_win",
            f"p{p}_tie",
            f"p{p}_loss",
        ])

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for key in hand_keys:
            row: List[Union[str, float]] = [key]
            for p in players_list:
                d = stats_by_hand[key][p]
                row.extend([
                    float(d["equity"]),
                    float(d["win_rate"]),
                    float(d["tie_rate"]),
                    float(d["loss_rate"]),
                ])
            w.writerow(row)


def _plot_equity_heatmap(
    out_path: Path,
    hand_keys: List[str],
    players_list: List[int],
    equity_matrix: np.ndarray,
    title: str,
) -> None:
    """Save a heatmap image (hands x players) for equity."""
    if not HAVE_MPL:
        raise RuntimeError("matplotlib is not installed in the active environment")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, max(2.5, 0.18 * len(hand_keys))))
    ax = fig.add_subplot(111)

    im = ax.imshow(equity_matrix, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Players")
    ax.set_ylabel("Hand (sorted by p2 equity)")

    ax.set_xticks(list(range(len(players_list))))
    ax.set_xticklabels([str(p) for p in players_list])

    # For large runs (169 rows) this is readable-ish; for small test, it’s fine.
    ax.set_yticks(list(range(len(hand_keys))))
    ax.set_yticklabels(hand_keys)

    fig.colorbar(im, ax=ax, label="Equity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_preflop_bulk_single_players(
    n_players: int,
    n_sims: int,
    seed: int,
    out_csv: Path,
    out_png: Optional[Path] = None,
) -> None:
    """Compute preflop stats for ALL 169 hands at a fixed player count.

    Writes a CSV with columns:
      hand, equity, win, tie, loss

    Also optionally saves a 169x1 equity heatmap sorted by equity.
    """

    if n_players < 2:
        raise ValueError("n_players must be >= 2")

    # Compute + stream-write
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Union[str, float]]] = []

    total = len(PREFLOP_HANDS)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hand", "equity", "win", "tie", "loss"])  # header

        for idx, key in enumerate(PREFLOP_HANDS):
            # Progress report
            print(f"Now computing hand {idx+1}/{total}: {key}")
            res = simulate_preflop_equity(key, n_players=n_players, n_sims=n_sims, seed=seed)
            row = {
                "hand": key,
                "equity": float(res["equity"]),
                "win": float(res["win_rate"]),
                "tie": float(res["tie_rate"]),
                "loss": float(res["loss_rate"]),
            }
            results.append(row)

            w.writerow([row["hand"], row["equity"], row["win"], row["tie"], row["loss"]])
            if (idx + 1) % 10 == 0:
                print(f"  ...saved {idx+1}/{total} hands to CSV")
            # Periodic flush so partial progress is saved even if interrupted
            if (idx + 1) % 10 == 0:
                f.flush()

    # Sort by equity descending for plotting/ranking
    results_sorted = sorted(results, key=lambda d: float(d["equity"]), reverse=True)

    if out_png is not None:
        if HAVE_MPL:
            keys_sorted = [str(d["hand"]) for d in results_sorted]
            eq_col = np.asarray([[float(d["equity"]) ] for d in results_sorted], dtype=float)  # (169,1)

            # Plot: suppress y tick labels (too dense), but keep ordering.
            fig = plt.figure(figsize=(6, 10))
            ax = fig.add_subplot(111)
            im = ax.imshow(eq_col, aspect="auto")
            ax.set_title(f"Preflop equity heatmap (players={n_players}, sims={n_sims})")
            ax.set_xlabel("Players")
            ax.set_ylabel("Hands (sorted by equity)")
            ax.set_xticks([0])
            ax.set_xticklabels([str(n_players)])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, label="Equity")
            fig.tight_layout()
            out_png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_png, dpi=200)
            plt.close(fig)
        else:
            raise RuntimeError("matplotlib is not installed in the active environment")

    # Also write a ranked list (human-readable) next to the CSV
    ranked_txt = out_csv.with_suffix(".ranked.txt")
    with ranked_txt.open("w", encoding="utf-8") as g:
        for i, d in enumerate(results_sorted, start=1):
            g.write(f"{i:3d}. {d['hand']}: {float(d['equity']):.6f} (win={float(d['win']):.6f}, tie={float(d['tie']):.6f})\n")

def run_preflop_bulk_multi_players(
    players_list: List[int],
    n_sims: int,
    seed: int,
    out_csv: Path,
    out_png: Optional[Path] = None,
    sort_by_players: int = 2,
) -> None:
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
            stats_by_hand[key][p] = simulate_preflop_equity(key, n_players=p, n_sims=n_sims, seed=seed)
        if (idx + 1) % 5 == 0:
            print(f"  ...computed {idx+1}/{total} hands")

    hands_sorted = sorted(
        PREFLOP_HANDS,
        key=lambda h: float(stats_by_hand[h][sort_by_players]["equity"]),
        reverse=True,
    )

    _write_preflop_matrix_csv(out_csv, hands_sorted, players_list, stats_by_hand)

    if out_png is not None:
        if not HAVE_MPL:
            raise RuntimeError("matplotlib is not installed in the active environment")
        equity_matrix = np.asarray(
            [[float(stats_by_hand[h][p]["equity"]) for p in players_list] for h in hands_sorted],
            dtype=float,
        )
        _plot_equity_heatmap(
            out_png, hands_sorted, players_list, equity_matrix,
            title=f"Preflop equity heatmap (sims={n_sims}, sorted by p{sort_by_players})",
        )

    ranked_txt = out_csv.with_suffix(".ranked.txt")
    with ranked_txt.open("w", encoding="utf-8") as g:
        for i, h in enumerate(hands_sorted, start=1):
            base = stats_by_hand[h][sort_by_players]
            g.write(
                f"{i:3d}. {h}: p{sort_by_players} equity={float(base['equity']):.6f} "
                f"(win={float(base['win_rate']):.6f}, tie={float(base['tie_rate']):.6f})\n"
            )


def main() -> None:
    OPPONENTS_LIST = [1, 2, 3, 4, 5]
    PLAYERS_LIST = [o + 1 for o in OPPONENTS_LIST]
    N_SIMS = 20_000
    SEED = 10

    t0 = time.perf_counter()

    csv_path = Path(__file__).resolve().with_name("preflop_calculated_p2to6_20k.csv")
    png_path = Path(__file__).resolve().with_name("preflop_equity_heatmap_p2to6_20k.png")

    print(f"Running bulk preflop: players={PLAYERS_LIST} | sims={N_SIMS} per hand | hands={len(PREFLOP_HANDS)}")

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
