from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import math
from time import perf_counter

import numpy as np

# Ensure `src/` is on sys.path so sibling packages like `utils/` can be imported
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from game_engine.init_deck import create_deck
from game_engine.evaluate_hands import evaluate_hand, _compare_evals
from utils.format_hands import format_hand


EvalTuple = Tuple[int, int, np.ndarray]
EvalSig = Tuple[int, int, Tuple[int, ...]]

RANKS: Dict[str, int] = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}
SUITS: Dict[str, int] = {"c": 1, "d": 2, "h": 3, "s": 4}


def _eval_signature(ev: EvalTuple) -> EvalSig:
    rank, top, kickers = ev
    if kickers is None:
        ks: Tuple[int, ...] = ()
    else:
        ks = tuple(int(x) for x in np.asarray(kickers, dtype=int).ravel())
    return (int(rank), int(top), ks)


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


def checkwhatbeatsme(
    hand: np.ndarray,
    board: np.ndarray,
    self_hand_eval: EvalTuple,
) -> Dict[str, object]:
    """Return possible better hand-evals and their hole-card combos on the river."""
    if hand.shape != (2, 2):
        raise ValueError(f"hand must have shape (2,2); got {hand.shape}")
    if board.shape != (2, 5):
        raise ValueError(f"board must have shape (2,5); got {board.shape}")

    deck = create_deck()
    used = np.hstack([hand, board])
    deck_rem = _remove_cards_from_deck(deck, used)

    board_int = np.asarray(board, dtype=int)
    seven = np.empty((2, 7), dtype=int)
    seven[:, 2:] = board_int

    eval_fn = evaluate_hand
    cmp_fn = _compare_evals

    possible: Set[EvalSig] = set()
    combos_by_eval: Dict[EvalSig, List[Tuple[Tuple[int, int], Tuple[int, int]]]] = {}

    beat_count = 0
    tie_count = 0
    lose_count = 0

    n = deck_rem.shape[1]
    for i in range(n - 1):
        v1 = int(deck_rem[0, i])
        s1 = int(deck_rem[1, i])
        for j in range(i + 1, n):
            seven[0, 0] = v1
            seven[1, 0] = s1
            seven[0, 1] = int(deck_rem[0, j])
            seven[1, 1] = int(deck_rem[1, j])
            opp_eval = eval_fn(seven)
            cmp = cmp_fn(opp_eval, self_hand_eval)
            if cmp < 0:
                sig = _eval_signature(opp_eval)
                possible.add(sig)
                combos_by_eval.setdefault(sig, []).append(((v1, s1), (int(deck_rem[0, j]), int(deck_rem[1, j]))))
                beat_count += 1
            elif cmp == 0:
                tie_count += 1
            else:
                lose_count += 1

    total_combos = n * (n - 1) // 2
    beating_combos = sum(len(v) for v in combos_by_eval.values())
    if beating_combos != beat_count:
        # Defensive consistency check: they should match exactly.
        beat_count = beating_combos

    return {
        "evals": possible,
        "combos_by_eval": combos_by_eval,
        "deck_rem": deck_rem,
        "board": board_int,
        "hand": np.asarray(hand, dtype=int),
        "total_combos": total_combos,
        "beating_combos": beating_combos,
        "tie_combos": tie_count,
        "lose_combos": lose_count,
    }


def handsthatbeat(possiblehandsthatbeatme: Dict[str, object]) -> Dict[EvalSig, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """Build all hole-card combos for each beating hand-eval signature."""
    if "combos_by_eval" in possiblehandsthatbeatme:
        combos = possiblehandsthatbeatme["combos_by_eval"]
        if isinstance(combos, dict):
            return combos

    evals = possiblehandsthatbeatme.get("evals")
    deck_rem = possiblehandsthatbeatme.get("deck_rem")
    board = possiblehandsthatbeatme.get("board")
    if evals is None or deck_rem is None or board is None:
        raise ValueError("possiblehandsthatbeatme must be output from checkwhatbeatsme")

    eval_set = set(evals)  # type: ignore[arg-type]
    board_int = np.asarray(board, dtype=int)
    seven = np.empty((2, 7), dtype=int)
    seven[:, 2:] = board_int

    eval_fn = evaluate_hand

    combos_by_eval: Dict[EvalSig, List[Tuple[Tuple[int, int], Tuple[int, int]]]] = {}
    n = deck_rem.shape[1]
    for i in range(n - 1):
        v1 = int(deck_rem[0, i])
        s1 = int(deck_rem[1, i])
        for j in range(i + 1, n):
            seven[0, 0] = v1
            seven[1, 0] = s1
            seven[0, 1] = int(deck_rem[0, j])
            seven[1, 1] = int(deck_rem[1, j])
            sig = _eval_signature(eval_fn(seven))
            if sig in eval_set:
                combos_by_eval.setdefault(sig, []).append(((v1, s1), (int(deck_rem[0, j]), int(deck_rem[1, j]))))

    return combos_by_eval


def evaluate_probability(possiblehandsthatbeatme: Dict[str, object], n_players: int) -> Dict[str, float]:
    """Approximate probability that at least one opponent holds a beating hand."""
    if n_players < 2:
        raise ValueError("n_players must be >= 2")

    total_combos = possiblehandsthatbeatme.get("total_combos")
    beating_combos = possiblehandsthatbeatme.get("beating_combos")
    tie_combos = possiblehandsthatbeatme.get("tie_combos")

    if total_combos is None or beating_combos is None:
        combos_by_eval = handsthatbeat(possiblehandsthatbeatme)
        deck_rem = possiblehandsthatbeatme.get("deck_rem")
        if deck_rem is None:
            raise ValueError("possiblehandsthatbeatme must be output from checkwhatbeatsme")
        n = deck_rem.shape[1]
        total_combos = n * (n - 1) // 2
        beating_combos = sum(len(v) for v in combos_by_eval.values())
        # tie_combos unknown in this fallback path
        tie_combos = 0

    total = float(total_combos)
    beat = float(beating_combos)
    tie = float(tie_combos or 0.0)
    if total <= 0.0:
        raise ValueError("No possible opponent combos")

    p_single_beat = beat / total
    p_single_tie = tie / total
    p_single_win = max(0.0, 1.0 - p_single_beat - p_single_tie)

    k = n_players - 1
    p_loss = 1.0 - (1.0 - p_single_beat) ** k
    p_win = p_single_win ** k
    p_tie = (p_single_win + p_single_tie) ** k - p_win

    equity = 0.0
    for m in range(k + 1):
        comb = math.comb(k, m)
        equity += comb * (p_single_win ** (k - m)) * (p_single_tie ** m) * (1.0 / (m + 1))

    return {
        "p_any": float(p_loss),
        "p_single": float(p_single_beat),
        "p_single_tie": float(p_single_tie),
        "p_win": float(p_win),
        "p_tie": float(p_tie),
        "p_loss": float(p_loss),
        "equity": float(equity),
        "beating_combos": float(beat),
        "tie_combos": float(tie),
        "total_combos": float(total),
    }


def benchmark_runtime(
    hand: np.ndarray,
    board: np.ndarray,
    n_players_list: List[int] | None = None,
    repeats: int = 1,
) -> Dict[str, float]:
    """Benchmark river enumeration + p_any evaluation for multiple player counts."""
    if n_players_list is None:
        n_players_list = [2, 3, 4, 5, 6]
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    hero_eval = evaluate_hand(np.hstack([hand, board]))

    t0 = perf_counter()
    last = None
    for _ in range(repeats):
        last = checkwhatbeatsme(hand, board, hero_eval)
    t1 = perf_counter()

    if last is None:
        raise RuntimeError("benchmark failed to compute enumeration")

    out: Dict[str, float] = {
        "enumeration_sec": (t1 - t0) / repeats,
        "total_combos": float(last.get("total_combos", 0.0)),
        "beating_combos": float(last.get("beating_combos", 0.0)),
    }

    for p in n_players_list:
        t2 = perf_counter()
        for _ in range(repeats):
            _ = evaluate_probability(last, p)
        t3 = perf_counter()
        out[f"p{p}_sec"] = (t3 - t2) / repeats

    return out


def _parse_card(tok: str) -> Tuple[int, int]:
    t = tok.strip().lower()
    t = re.sub(r"[^0-9tjqka+cdhs]", "", t)
    if not t:
        raise ValueError("Empty card token")

    if t.startswith("10"):
        rank = 10
        suit = t[2:3]
    else:
        rank_char = t[0].upper()
        rank = RANKS.get(rank_char)
        suit = t[1:2]

    if rank is None or not suit:
        raise ValueError(f"Bad card token '{tok}'")

    suit_val = SUITS.get(suit)
    if suit_val is None:
        raise ValueError(f"Bad suit in card token '{tok}'")

    return rank, suit_val


def _parse_cards(raw: str, n_expected: int) -> np.ndarray:
    tokens = [t for t in re.split(r"[\s,]+", raw.strip()) if t]
    if len(tokens) != n_expected:
        raise ValueError(f"Expected {n_expected} cards, got {len(tokens)}")

    seen = set()
    vals: List[int] = []
    suits: List[int] = []
    for tok in tokens:
        v, s = _parse_card(tok)
        if (v, s) in seen:
            raise ValueError(f"Duplicate card: {tok}")
        seen.add((v, s))
        vals.append(v)
        suits.append(s)

    return np.asarray([vals, suits], dtype=int)


def _summary_rows(
    combos_by_eval: Dict[EvalSig, List[Tuple[Tuple[int, int], Tuple[int, int]]]],
    total: int,
    top_n: int = 10,
) -> List[Tuple[float, int, str]]:
    rows: List[Tuple[float, int, str]] = []
    for sig, combos in combos_by_eval.items():
        ev = (sig[0], sig[1], np.asarray(sig[2], dtype=int))
        desc = format_hand(ev, tiebreak=True)
        count = len(combos)
        pct = 100.0 * count / float(total)
        rows.append((pct, count, desc))

    rows.sort(key=lambda x: x[0], reverse=True)
    return rows[:top_n]


def main() -> None:
    print("River smoke test")
    print("Enter cards like: Ah Kd (ace hearts, king diamonds)")
    hand_raw = input("Hero hand (2 cards): ").strip()
    board_raw = input("Board (5 cards): ").strip()
    n_players_raw = input("Players still in (>=2): ").strip()

    hand = _parse_cards(hand_raw, 2)
    board = _parse_cards(board_raw, 5)
    n_players = int(n_players_raw)

    seven = np.hstack([hand, board])
    self_eval = evaluate_hand(seven)

    res = checkwhatbeatsme(hand, board, self_eval)
    prob = evaluate_probability(res, n_players)

    print(f"Your hand: {format_hand(self_eval, tiebreak=True)}")
    print(f"Total opponent combos: {int(res['total_combos'])}")
    print(f"Beating combos: {int(res['beating_combos'])}")
    if "tie_combos" in res:
        print(f"Tie combos: {int(res['tie_combos'])}")
    print(f"P(any opponent beats you): {prob['p_any'] * 100.0:.2f}%")
    print(f"P(win): {prob['p_win'] * 100.0:.2f}% | P(tie): {prob['p_tie'] * 100.0:.2f}%")
    print(f"Equity (pot share): {prob['equity'] * 100.0:.2f}%")

    combos_by_eval = res["combos_by_eval"]
    if combos_by_eval:
        print("\nMost likely beating hands:")
        for pct, count, desc in _summary_rows(combos_by_eval, int(res["total_combos"])):
            print(f"  {pct:6.2f}%  ({count:4d})  {desc}")


if __name__ == "__main__":
    main()
