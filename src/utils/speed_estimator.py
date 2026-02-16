import argparse
import sys
from pathlib import Path
from time import perf_counter

# Ensure `src/` is on sys.path so sibling packages can be imported
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "numpy is required for this benchmark. Install it in your env and rerun."
    ) from exc

from game_engine.init_deck import create_deck
from game_engine.evaluate_hands import evaluate_hand, hand_plus_community, evaluate_winner


def _sample_7_cards(rng: np.random.Generator, deck: np.ndarray) -> np.ndarray:
    idx = rng.choice(deck.shape[1], size=7, replace=False)
    return deck[:, idx]


def _sample_drawn(rng: np.random.Generator, deck: np.ndarray, n_players: int) -> np.ndarray:
    k = 2 * n_players + 5
    idx = rng.choice(deck.shape[1], size=k, replace=False)
    return deck[:, idx]


def _sample_hole_and_flop(rng: np.random.Generator, deck: np.ndarray):
    idx = rng.choice(deck.shape[1], size=5, replace=False)
    hole = deck[:, idx[:2]]
    flop = deck[:, idx[2:]]
    return hole, flop, idx


def bench_evaluate_hand(n: int, warmup: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    deck = create_deck()

    for _ in range(warmup):
        evaluate_hand(_sample_7_cards(rng, deck))

    start = perf_counter()
    for _ in range(n):
        evaluate_hand(_sample_7_cards(rng, deck))
    dt = perf_counter() - start

    return n / dt


def bench_showdown(n: int, warmup: int, seed: int, n_players: int) -> float:
    rng = np.random.default_rng(seed)
    deck = create_deck()

    for _ in range(warmup):
        drawn = _sample_drawn(rng, deck, n_players)
        evals = [evaluate_hand(hand_plus_community(i, drawn)) for i in range(n_players)]
        evaluate_winner(evals)

    start = perf_counter()
    for _ in range(n):
        drawn = _sample_drawn(rng, deck, n_players)
        evals = [evaluate_hand(hand_plus_community(i, drawn)) for i in range(n_players)]
        evaluate_winner(evals)
    dt = perf_counter() - start

    return n / dt


def bench_multiway(n: int, warmup: int, seed: int, min_players: int, max_players: int):
    rng = np.random.default_rng(seed)
    deck = create_deck()

    hole, flop, used_idx = _sample_hole_and_flop(rng, deck)
    used_mask = np.zeros(deck.shape[1], dtype=bool)
    used_mask[used_idx] = True
    remaining = deck[:, ~used_mask]

    results = {}
    for n_players in range(min_players, max_players + 1):
        for _ in range(warmup):
            perm = rng.permutation(remaining.shape[1])
            rem = remaining[:, perm]
            opp = rem[:, : 2 * (n_players - 1)]
            turn_river = rem[:, 2 * (n_players - 1) : 2 * (n_players - 1) + 2]
            community = np.hstack([flop, turn_river])
            drawn = np.hstack([hole, opp, community])
            evals = [evaluate_hand(hand_plus_community(i, drawn)) for i in range(n_players)]
            evaluate_winner(evals)

        start = perf_counter()
        for _ in range(n):
            perm = rng.permutation(remaining.shape[1])
            rem = remaining[:, perm]
            opp = rem[:, : 2 * (n_players - 1)]
            turn_river = rem[:, 2 * (n_players - 1) : 2 * (n_players - 1) + 2]
            community = np.hstack([flop, turn_river])
            drawn = np.hstack([hole, opp, community])
            evals = [evaluate_hand(hand_plus_community(i, drawn)) for i in range(n_players)]
            evaluate_winner(evals)
        dt = perf_counter() - start

        results[n_players] = (n / dt, dt)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate poker eval throughput and project total time.")
    parser.add_argument("--mode", choices=["hand", "showdown", "multiway"], default="hand")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--min-players", type=int, default=2)
    parser.add_argument("--max-players", type=int, default=6)
    parser.add_argument("--target-evals", type=float, default=4_000_000_000)
    args = parser.parse_args()

    if args.mode == "hand":
        rate = bench_evaluate_hand(args.n, args.warmup, args.seed)
        unit = "hand-evals/sec"
    elif args.mode == "showdown":
        rate = bench_showdown(args.n, args.warmup, args.seed, args.players)
        unit = f"{args.players}p-showdowns/sec"
    else:
        results = bench_multiway(args.n, args.warmup, args.seed, args.min_players, args.max_players)
        for n_players in range(args.min_players, args.max_players + 1):
            rate, dt = results[n_players]
            print(f"{n_players}p rate: {rate:,.1f} showdowns/sec | {args.n} trials in {dt:,.3f} sec")
        return

    seconds = args.target_evals / rate
    hours = seconds / 3600
    days = hours / 24

    print(f"rate: {rate:,.1f} {unit}")
    print(f"target: {args.target_evals:,.0f} evals")
    print(f"time: {seconds:,.1f} sec | {hours:,.2f} hrs | {days:,.2f} days")


if __name__ == "__main__":
    main()
