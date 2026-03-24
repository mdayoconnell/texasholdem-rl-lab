"""Quick river sanity check: exact vs Monte Carlo."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# Ensure `src/` is on sys.path so sibling packages can be imported
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print("exe:", sys.executable)
print("cwd:", os.getcwd())
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("sys.path[0:5]:", sys.path[:5])

from game_engine.evaluate_hands import _compare_evals, evaluate_hand
from game_engine.init_deck import create_deck
from probability_engine import river as river_engine
from utils.format_hands import format_hand


def _monte_carlo_p_any(
    hand: np.ndarray,
    board: np.ndarray,
    n_players: int,
    n_sims: int = 10_000,
    seed: int = 0,
) -> float:
    if n_players < 2:
        raise ValueError("n_players must be >= 2")
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")

    rng = np.random.default_rng(seed)

    deck = create_deck()
    used = np.hstack([hand, board])
    deck_rem = river_engine._remove_cards_from_deck(deck, used)

    need = 2 * (n_players - 1)
    deck_vals = deck_rem[0]
    deck_suits = deck_rem[1]
    n_rem = deck_vals.shape[0]
    if n_rem < need:
        raise ValueError("Not enough cards remaining in deck")

    hero_eval = evaluate_hand(np.hstack([hand, board]))

    seven = np.empty((2, 7), dtype=int)
    seven[:, 2:] = board

    cmp_fn = _compare_evals
    beat_count = 0

    for _ in range(n_sims):
        idx = rng.choice(n_rem, size=need, replace=False)
        any_beats = False
        for p in range(n_players - 1):
            i = 2 * p
            seven[0, 0] = int(deck_vals[idx[i]])
            seven[1, 0] = int(deck_suits[idx[i]])
            seven[0, 1] = int(deck_vals[idx[i + 1]])
            seven[1, 1] = int(deck_suits[idx[i + 1]])
            opp_eval = evaluate_hand(seven)
            if cmp_fn(opp_eval, hero_eval) < 0:
                any_beats = True
                break
        if any_beats:
            beat_count += 1

    return beat_count / float(n_sims)


def _run_case(
    name: str,
    hand_str: str,
    board_str: str,
    n_players: int = 2,
    n_sims: int = 10_000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    hand = river_engine._parse_cards(hand_str, 2)
    board = river_engine._parse_cards(board_str, 5)

    hero_eval = evaluate_hand(np.hstack([hand, board]))
    exact = river_engine.evaluate_probability(
        river_engine.checkwhatbeatsme(hand, board, hero_eval), n_players
    )["p_any"]

    mc = _monte_carlo_p_any(hand, board, n_players, n_sims=n_sims, seed=seed)
    diff = abs(exact - mc)

    print(f"{name}: exact={exact:.4f} mc={mc:.4f} diff={diff:.4f}")
    return exact, mc, diff


def _card_to_str(value: int, suit: int) -> str:
    rank_map = {
        14: "A",
        13: "K",
        12: "Q",
        11: "J",
        10: "T",
    }
    suit_map = {
        1: "c",
        2: "d",
        3: "h",
        4: "s",
    }
    v = int(value)
    s = int(suit)
    rank = rank_map.get(v, str(v))
    return f"{rank}{suit_map[s]}"


def _cards_to_str(cards: np.ndarray) -> str:
    return " ".join(_card_to_str(cards[0, i], cards[1, i]) for i in range(cards.shape[1]))


def _hand_board_str(hand: np.ndarray, board: np.ndarray) -> Tuple[str, str]:
    return _cards_to_str(hand), _cards_to_str(board)


def _parse_n_players(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("n_players list is empty")
    out: List[int] = []
    for p in parts:
        n = int(p)
        if n < 2:
            raise ValueError("n_players must be >= 2")
        out.append(n)
    return out


def _compute_exact(hand: np.ndarray, board: np.ndarray, n_players: int) -> Tuple[float, Tuple[int, int, np.ndarray]]:
    hero_eval = evaluate_hand(np.hstack([hand, board]))
    exact = river_engine.evaluate_probability(
        river_engine.checkwhatbeatsme(hand, board, hero_eval), n_players
    )["p_any"]
    return float(exact), hero_eval


def _percentile(values: np.ndarray, p: float) -> float:
    return float(np.percentile(values, p))


def _summarize_records(records: List[Dict[str, object]]) -> Dict[str, object]:
    diffs = np.asarray([float(r["diff"]) for r in records], dtype=float)
    signed = np.asarray([float(r["signed"]) for r in records], dtype=float)
    summary: Dict[str, object] = {
        "count": int(diffs.size),
        "mean": float(diffs.mean()) if diffs.size else 0.0,
        "median": float(np.median(diffs)) if diffs.size else 0.0,
        "p90": _percentile(diffs, 90) if diffs.size else 0.0,
        "p95": _percentile(diffs, 95) if diffs.size else 0.0,
        "p99": _percentile(diffs, 99) if diffs.size else 0.0,
        "max": float(diffs.max()) if diffs.size else 0.0,
        "bias": float(signed.mean()) if signed.size else 0.0,
        "gt_001": int(np.sum(diffs > 0.01)),
        "gt_002": int(np.sum(diffs > 0.02)),
        "gt_003": int(np.sum(diffs > 0.03)),
    }
    return summary


def _top_records(records: List[Dict[str, object]], n: int = 20) -> List[Dict[str, object]]:
    return sorted(records, key=lambda r: float(r["diff"]), reverse=True)[:n]


def _render_report(
    cases: int,
    n_players_list: List[int],
    n_sims: int,
    seed: int,
    results_by_players: Dict[int, List[Dict[str, object]]],
) -> str:
    lines: List[str] = []
    lines.append("# River MC vs Exact Report")
    lines.append("")
    lines.append("## Parameters")
    lines.append(f"- cases: {cases}")
    lines.append(f"- n_players: {', '.join(str(n) for n in n_players_list)}")
    lines.append(f"- n_sims: {n_sims}")
    lines.append(f"- seed: {seed}")
    lines.append(f"- generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    for n_players in n_players_list:
        records = results_by_players.get(n_players, [])
        summary = _summarize_records(records)
        lines.append(f"## n_players = {n_players}")
        lines.append("")
        lines.append("Summary:")
        lines.append(f"- count: {summary['count']}")
        lines.append(f"- mean diff: {summary['mean']:.6f}")
        lines.append(f"- median diff: {summary['median']:.6f}")
        lines.append(f"- p90 diff: {summary['p90']:.6f}")
        lines.append(f"- p95 diff: {summary['p95']:.6f}")
        lines.append(f"- p99 diff: {summary['p99']:.6f}")
        lines.append(f"- max diff: {summary['max']:.6f}")
        lines.append(f"- mean signed error (mc - exact): {summary['bias']:.6f}")
        lines.append(f"- count diff > 0.01: {summary['gt_001']}")
        lines.append(f"- count diff > 0.02: {summary['gt_002']}")
        lines.append(f"- count diff > 0.03: {summary['gt_003']}")
        lines.append("")
        lines.append("Top 20 divergences:")
        lines.append("")
        lines.append("| # | diff | signed | exact | mc | hand | board | hero_eval |")
        lines.append("| - | ---- | ------ | ----- | -- | ---- | ----- | --------- |")
        for i, rec in enumerate(_top_records(records, n=20), start=1):
            lines.append(
                "| {idx} | {diff:.6f} | {signed:.6f} | {exact:.6f} | {mc:.6f} | {hand} | {board} | {hero} |".format(
                    idx=i,
                    diff=float(rec["diff"]),
                    signed=float(rec["signed"]),
                    exact=float(rec["exact"]),
                    mc=float(rec["mc"]),
                    hand=str(rec["hand_str"]),
                    board=str(rec["board_str"]),
                    hero=str(rec["hero_desc"]),
                )
            )
        lines.append("")

    lines.append("## Discussion")
    lines.append("")
    lines.append(
        "For n_players = 2, the exact method is truly exact (it enumerates all opponent "
        "hole-card combos), so remaining differences are Monte Carlo noise. With 5,000 sims, "
        "a typical standard error near p=0.5 is about 0.007, so occasional 0.01-level diffs are expected."
    )
    lines.append("")
    lines.append(
        "For n_players > 2, the method uses an independence approximation: "
        "p_any = 1 - (1 - p_single)^(n_players-1). In reality, opponents' hole cards are drawn "
        "without replacement, which creates negative dependence across opponents. This usually makes "
        "the approximation slightly overestimate the true p_any."
    )
    lines.append("")
    lines.append(
        "This report compares p_any (strictly better hands). Ties are treated as non-beating in both "
        "the exact enumerator and MC simulation for this test. The engine now also reports tie rates "
        "and equity separately."
    )
    lines.append("")
    lines.append("### Strengths")
    lines.append(
        "- Exact for heads-up (n_players = 2) and very fast compared to Monte Carlo.\n"
        "- Uses full river enumeration for opponent hands, so p_single is exact and stable."
    )
    lines.append("")
    lines.append("### Weaknesses / Limitations")
    lines.append(
        "- For n_players > 2, independence assumption ignores card removal between opponents, "
        "introducing a small but systematic bias (usually an overestimate of risk).\n"
        "- Does not report tie equity; ties are counted as non-beating."
    )
    lines.append("")
    lines.append("### Recommendations")
    lines.append(
        "- Keep current method for fast estimates, but if you need multi-player precision, "
        "consider a correction or direct multi-opponent enumeration for small player counts.\n"
        "- If you care about split pots, add a tie-rate calculation alongside p_any."
    )
    lines.append("")
    return "\n".join(lines)


def _run_random_cases(
    n_cases: int,
    n_players_list: List[int],
    n_sims: int,
    seed: int,
    report_path: str | None,
) -> str:
    rng = np.random.default_rng(seed)
    deck = create_deck()

    results_by_players: Dict[int, List[Dict[str, object]]] = {n: [] for n in n_players_list}

    for case_idx in range(n_cases):
        idx = rng.choice(deck.shape[1], size=7, replace=False)
        cards = deck[:, idx]
        hand = cards[:, :2]
        board = cards[:, 2:]

        hero_eval = evaluate_hand(np.hstack([hand, board]))
        hero_desc = format_hand(hero_eval, tiebreak=True)
        hand_str, board_str = _hand_board_str(hand, board)

        for p_idx, n_players in enumerate(n_players_list):
            exact, _ = _compute_exact(hand, board, n_players)
            mc_seed = seed + (case_idx * 1000) + p_idx
            mc = _monte_carlo_p_any(hand, board, n_players, n_sims=n_sims, seed=mc_seed)
            diff = abs(exact - mc)
            signed = mc - exact

            results_by_players[n_players].append(
                {
                    "case_idx": case_idx,
                    "hand_str": hand_str,
                    "board_str": board_str,
                    "hero_desc": hero_desc,
                    "exact": exact,
                    "mc": mc,
                    "diff": diff,
                    "signed": signed,
                }
            )

        if (case_idx + 1) % 100 == 0:
            print(f"Completed {case_idx + 1}/{n_cases} cases")

    report = _render_report(n_cases, n_players_list, n_sims, seed, results_by_players)

    if report_path:
        report_path_obj = Path(report_path)
        report_path_obj.parent.mkdir(parents=True, exist_ok=True)
        report_path_obj.write_text(report)
        print(f"Wrote report to {report_path_obj}")

    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="River exact vs Monte Carlo checks")
    parser.add_argument(
        "--random-cases",
        type=int,
        default=0,
        help="Number of random river cases to run (0 = use static cases).",
    )
    parser.add_argument(
        "--n-players",
        type=str,
        default="2,5",
        help="Comma-separated player counts for random mode (e.g., '2,5').",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=5_000,
        help="Monte Carlo sims per case.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Optional path to write a markdown report.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.random_cases and args.random_cases > 0:
        n_players_list = _parse_n_players(args.n_players)
        report = _run_random_cases(
            n_cases=int(args.random_cases),
            n_players_list=n_players_list,
            n_sims=int(args.n_sims),
            seed=int(args.seed),
            report_path=args.report_path,
        )
        print("\n=== Report Summary ===")
        print(report)
        return

    cases: Iterable[Tuple[str, str, str, int]] = [
        ("pair_of_aces", "Ah Ad", "Ks 7d 2c 9h Jd", 5),
        ("royal_flush", "Jh Th", "Ah Kh Qh 2c 3d", 5),
        ("six_high_straight", "2h 3d", "4s 5c 6d Kd Qs", 5),
    ]

    tol = 0.03
    for i, (name, hand, board, n_players) in enumerate(cases):
        _exact, _mc, diff = _run_case(
            name,
            hand,
            board,
            n_players=n_players,
            n_sims=10_000,
            seed=1337 + i,
        )
        if diff > tol:
            raise AssertionError(f"{name}: diff {diff:.4f} exceeds tol {tol:.4f}")


if __name__ == "__main__":
    main()
