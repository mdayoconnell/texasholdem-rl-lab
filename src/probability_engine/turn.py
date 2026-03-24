from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Tuple
import math

import numpy as np

# Ensure `src/` is on sys.path so sibling packages like `utils/` can be imported
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from game_engine.init_deck import create_deck
from game_engine.evaluate_hands import evaluate_hand, _compare_evals


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

_Z_TABLE: Dict[float, float] = {
    0.90: 1.644854,
    0.95: 1.959964,
    0.98: 2.326348,
    0.99: 2.575829,
}


def _z_for_ci(ci: float) -> float:
    z = _Z_TABLE.get(round(float(ci), 2))
    if z is None:
        raise ValueError(f"Unsupported ci={ci}. Supported: {sorted(_Z_TABLE.keys())}")
    return z


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(x, hi))


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


def estimate_turn_single_outcomes(
    hand: np.ndarray,
    board: np.ndarray,
    n_sims_per_river: int = 1_000,
    seed: int | None = None,
) -> Dict[str, float]:
    """Estimate single-opponent outcomes on the turn.

    - Enumerates all 46 possible river cards (exact).
    - For each river, Monte Carlo samples opponent hole cards.
    - Returns per-opponent beat/tie/win probabilities.
    """
    if hand.shape != (2, 2):
        raise ValueError(f"hand must have shape (2,2); got {hand.shape}")
    if board.shape != (2, 4):
        raise ValueError(f"board must have shape (2,4); got {board.shape}")
    if n_sims_per_river <= 0:
        raise ValueError("n_sims_per_river must be positive")

    rng = np.random.default_rng(seed)

    deck = create_deck()
    used = np.hstack([hand, board])
    deck_rem = _remove_cards_from_deck(deck, used)

    n_rivers = deck_rem.shape[1]
    if n_rivers < 1:
        raise ValueError("No river cards available")

    seven = np.empty((2, 7), dtype=int)
    seven[:, 2:6] = board

    beat_sum = 0.0
    tie_sum = 0.0

    for r_idx in range(n_rivers):
        river = deck_rem[:, r_idx : r_idx + 1]

        # Hero eval for this river
        hero_eval = evaluate_hand(np.hstack([hand, board, river]))

        # Remove river card from remaining deck
        mask = np.ones(n_rivers, dtype=bool)
        mask[r_idx] = False
        deck_after = deck_rem[:, mask]
        vals = deck_after[0]
        suits = deck_after[1]
        n_rem = vals.shape[0]
        if n_rem < 2:
            raise ValueError("Not enough cards remaining for opponent hand")

        seven[:, 6] = river[:, 0]

        beat = 0
        tie = 0
        for _ in range(n_sims_per_river):
            idx = rng.choice(n_rem, size=2, replace=False)
            seven[0, 0] = int(vals[idx[0]])
            seven[1, 0] = int(suits[idx[0]])
            seven[0, 1] = int(vals[idx[1]])
            seven[1, 1] = int(suits[idx[1]])
            opp_eval = evaluate_hand(seven)
            cmp = _compare_evals(opp_eval, hero_eval)
            if cmp < 0:
                beat += 1
            elif cmp == 0:
                tie += 1

        beat_sum += beat / float(n_sims_per_river)
        tie_sum += tie / float(n_sims_per_river)

    p_single_beat = beat_sum / float(n_rivers)
    p_single_tie = tie_sum / float(n_rivers)
    p_single_win = max(0.0, 1.0 - p_single_beat - p_single_tie)

    return {
        "p_single": float(p_single_beat),
        "p_single_tie": float(p_single_tie),
        "p_single_win": float(p_single_win),
    }


def estimate_turn_p_single(
    hand: np.ndarray,
    board: np.ndarray,
    n_sims_per_river: int = 1_000,
    seed: int | None = None,
) -> float:
    """Backward-compatible helper: return P(single opponent beats hero)."""
    return estimate_turn_single_outcomes(hand, board, n_sims_per_river, seed)["p_single"]


def evaluate_turn_probability_joint_mc(
    hand: np.ndarray,
    board: np.ndarray,
    n_players: int = 2,
    eval_budget: int = 1_800,
    ci: float = 0.98,
    ci_half_width: float = 0.02,
    seed: int | None = None,
    n_floor: int = 64,
    min_n_floor: int = 16,
    ci_check_every: int = 8,
    flop_prior_equity: float | None = None,
    flop_prior_n_sims: int | None = None,
    prior_scale: float = 0.02,
    prior_max_weight: float = 40.0,
    prior_disagreement_scale: float = 0.15,
    shrink_on_cap: bool = True,
) -> Dict[str, float]:
    """Adaptive turn estimator with joint multiway sampling and CI stopping.

    Each sampled world draws:
      - one river card
      - all opponents' hole cards jointly (without replacement)
    so multiway card-removal dependence is handled exactly by construction.

    Sampling stops when either:
      - the target confidence-interval half-width is reached, or
      - the evaluation budget cap is reached.

    A flop LUT prior (equity + sims) can be provided as a weak stabilizer when
    the cap is hit before CI convergence.
    """
    if hand.shape != (2, 2):
        raise ValueError(f"hand must have shape (2,2); got {hand.shape}")
    if board.shape != (2, 4):
        raise ValueError(f"board must have shape (2,4); got {board.shape}")
    if n_players < 2:
        raise ValueError("n_players must be >= 2")
    if eval_budget <= 0:
        raise ValueError("eval_budget must be positive")
    if ci_half_width <= 0.0:
        raise ValueError("ci_half_width must be positive")
    if n_floor < 2:
        raise ValueError("n_floor must be >= 2")
    if min_n_floor < 2:
        raise ValueError("min_n_floor must be >= 2")
    if ci_check_every < 1:
        raise ValueError("ci_check_every must be >= 1")
    if prior_scale < 0.0:
        raise ValueError("prior_scale must be >= 0")
    if prior_max_weight < 0.0:
        raise ValueError("prior_max_weight must be >= 0")
    if prior_disagreement_scale <= 0.0:
        raise ValueError("prior_disagreement_scale must be positive")

    if flop_prior_equity is not None:
        flop_prior_equity = float(flop_prior_equity)
        if flop_prior_equity < 0.0 or flop_prior_equity > 1.0:
            raise ValueError("flop_prior_equity must be in [0,1]")
    if flop_prior_n_sims is not None and flop_prior_n_sims < 0:
        raise ValueError("flop_prior_n_sims must be >= 0")

    k = n_players - 1
    evals_per_world = n_players
    worlds_max = eval_budget // evals_per_world
    if worlds_max < 2:
        raise ValueError(
            f"eval_budget={eval_budget} is too small for n_players={n_players}; "
            f"need at least {2 * evals_per_world} for CI estimation"
        )

    z = _z_for_ci(ci)
    rng = np.random.default_rng(seed)

    deck = create_deck()
    used = np.hstack([hand, board])
    deck_rem = _remove_cards_from_deck(deck, used)

    vals = deck_rem[0]
    suits = deck_rem[1]
    n_rem = vals.shape[0]
    need = 1 + 2 * k
    if n_rem < need:
        raise ValueError("Not enough cards remaining for river + opponents")

    min_floor_eff = _clamp_int(min_n_floor, 2, worlds_max)
    n_floor_eff = _clamp_int(n_floor, min_floor_eff, worlds_max)
    n_guess = worlds_max

    if flop_prior_equity is not None:
        prior_var = max(flop_prior_equity * (1.0 - flop_prior_equity), 1e-6)
        n_guess = int(math.ceil((z * z * prior_var) / (ci_half_width * ci_half_width)))
        adaptive_floor = _clamp_int(int(math.ceil(0.5 * max(2, n_guess))), min_floor_eff, worlds_max)
        n_floor_eff = min(n_floor_eff, adaptive_floor)

    hero_seven = np.empty((2, 7), dtype=int)
    hero_seven[:, :2] = hand
    hero_seven[:, 2:6] = board

    opp_seven = np.empty((2, 7), dtype=int)
    opp_seven[:, 2:6] = board

    win = 0
    tie = 0
    loss = 0
    single_beat_sum = 0.0
    single_tie_sum = 0.0
    n_worlds = 0
    mean = 0.0
    m2 = 0.0
    ci_met = False
    half_width = float("inf")

    for _ in range(worlds_max):
        idx = rng.choice(n_rem, size=need, replace=False)
        river_idx = int(idx[0])
        river_v = int(vals[river_idx])
        river_s = int(suits[river_idx])

        hero_seven[0, 6] = river_v
        hero_seven[1, 6] = river_s
        hero_eval = evaluate_hand(hero_seven)
        opp_seven[0, 6] = river_v
        opp_seven[1, 6] = river_s

        hero_beaten = False
        n_equal_best = 1
        beaters = 0
        ties_vs_hero = 0

        for p in range(k):
            c1 = int(idx[1 + 2 * p])
            c2 = int(idx[2 + 2 * p])
            opp_seven[0, 0] = int(vals[c1])
            opp_seven[1, 0] = int(suits[c1])
            opp_seven[0, 1] = int(vals[c2])
            opp_seven[1, 1] = int(suits[c2])

            opp_eval = evaluate_hand(opp_seven)
            cmp = _compare_evals(opp_eval, hero_eval)
            if cmp < 0:
                hero_beaten = True
                beaters += 1
            elif cmp == 0:
                ties_vs_hero += 1
                if not hero_beaten:
                    n_equal_best += 1

        single_beat_sum += beaters / float(k)
        single_tie_sum += ties_vs_hero / float(k)

        if hero_beaten:
            loss += 1
            share = 0.0
        else:
            if n_equal_best == 1:
                win += 1
            else:
                tie += 1
            share = 1.0 / float(n_equal_best)

        n_worlds += 1
        delta = share - mean
        mean += delta / float(n_worlds)
        m2 += delta * (share - mean)

        if n_worlds >= n_floor_eff and n_worlds >= 2 and (
            n_worlds % ci_check_every == 0 or n_worlds == worlds_max
        ):
            var = m2 / float(n_worlds - 1)
            half_width = z * math.sqrt(var / float(n_worlds))
            if half_width <= ci_half_width:
                ci_met = True
                break

    if n_worlds >= 2:
        var = m2 / float(n_worlds - 1)
        half_width = z * math.sqrt(var / float(n_worlds))

    p_win = win / float(n_worlds)
    p_tie = tie / float(n_worlds)
    p_loss = loss / float(n_worlds)
    p_single_beat = single_beat_sum / float(n_worlds)
    p_single_tie = single_tie_sum / float(n_worlds)
    p_single_win = max(0.0, 1.0 - p_single_beat - p_single_tie)

    equity_raw_mc = mean
    equity = equity_raw_mc
    prior_weight_base = 0.0
    prior_weight_eff = 0.0

    cap_hit = n_worlds >= worlds_max and not ci_met
    if (
        cap_hit
        and shrink_on_cap
        and flop_prior_equity is not None
        and flop_prior_n_sims is not None
        and flop_prior_n_sims > 0
    ):
        prior_weight_base = min(prior_max_weight, prior_scale * float(flop_prior_n_sims))
        if prior_weight_base > 0.0:
            d = abs(equity_raw_mc - flop_prior_equity)
            prior_weight_eff = prior_weight_base * math.exp(
                -((d / prior_disagreement_scale) ** 2)
            )
            equity = (
                (float(n_worlds) * equity_raw_mc) + (prior_weight_eff * flop_prior_equity)
            ) / (float(n_worlds) + prior_weight_eff)

    return {
        "p_any": float(p_loss),
        "p_single": float(p_single_beat),
        "p_single_tie": float(p_single_tie),
        "p_single_win": float(p_single_win),
        "p_win": float(p_win),
        "p_tie": float(p_tie),
        "p_loss": float(p_loss),
        "equity": float(equity),
        "equity_raw_mc": float(equity_raw_mc),
        "ci_half_width": float(half_width),
        "ci_target_half_width": float(ci_half_width),
        "ci_met": float(1.0 if ci_met else 0.0),
        "n_worlds": float(n_worlds),
        "n_evals": float(n_worlds * evals_per_world),
        "eval_budget": float(eval_budget),
        "worlds_max": float(worlds_max),
        "n_floor": float(n_floor_eff),
        "n_guess": float(min(worlds_max, n_guess)),
        "prior_equity": float(flop_prior_equity) if flop_prior_equity is not None else float("nan"),
        "prior_n_sims": float(flop_prior_n_sims) if flop_prior_n_sims is not None else 0.0,
        "prior_weight_base": float(prior_weight_base),
        "prior_weight_eff": float(prior_weight_eff),
        "used_shrinkage": float(1.0 if prior_weight_eff > 0.0 else 0.0),
        "n_players": float(n_players),
    }


def evaluate_turn_probability(
    hand: np.ndarray,
    board: np.ndarray,
    n_players: int = 2,
    n_sims_per_river: int = 1_000,
    seed: int | None = None,
    method: str = "joint_mc",
    eval_budget: int = 1_800,
    ci: float = 0.98,
    ci_half_width: float = 0.02,
    n_floor: int = 64,
    min_n_floor: int = 16,
    ci_check_every: int = 8,
    flop_prior_equity: float | None = None,
    flop_prior_n_sims: int | None = None,
    prior_scale: float = 0.02,
    prior_max_weight: float = 40.0,
    prior_disagreement_scale: float = 0.15,
    shrink_on_cap: bool = True,
) -> Dict[str, float]:
    """Estimate turn outcomes.

    Supported methods:
      - method="joint_mc" (default): adaptive joint multiway MC with CI stopping.
      - method="legacy": exact river enumeration + per-river HU MC + independence
        approximation for multiway.
    """
    if method == "joint_mc":
        return evaluate_turn_probability_joint_mc(
            hand=hand,
            board=board,
            n_players=n_players,
            eval_budget=eval_budget,
            ci=ci,
            ci_half_width=ci_half_width,
            seed=seed,
            n_floor=n_floor,
            min_n_floor=min_n_floor,
            ci_check_every=ci_check_every,
            flop_prior_equity=flop_prior_equity,
            flop_prior_n_sims=flop_prior_n_sims,
            prior_scale=prior_scale,
            prior_max_weight=prior_max_weight,
            prior_disagreement_scale=prior_disagreement_scale,
            shrink_on_cap=shrink_on_cap,
        )

    if method != "legacy":
        raise ValueError("method must be either 'joint_mc' or 'legacy'")

    if n_players < 2:
        raise ValueError("n_players must be >= 2")

    single = estimate_turn_single_outcomes(
        hand=hand,
        board=board,
        n_sims_per_river=n_sims_per_river,
        seed=seed,
    )

    p_single_beat = float(single["p_single"])
    p_single_tie = float(single["p_single_tie"])
    p_single_win = float(single["p_single_win"])

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
        "p_single_win": float(p_single_win),
        "p_win": float(p_win),
        "p_tie": float(p_tie),
        "p_loss": float(p_loss),
        "equity": float(equity),
        "n_players": float(n_players),
        "n_sims_per_river": float(n_sims_per_river),
    }


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
    vals = []
    suits = []
    for tok in tokens:
        v, s = _parse_card(tok)
        if (v, s) in seen:
            raise ValueError(f"Duplicate card: {tok}")
        seen.add((v, s))
        vals.append(v)
        suits.append(s)

    return np.asarray([vals, suits], dtype=int)


def main() -> None:
    print("Turn estimator")
    print("Enter cards like: Ah Kd (ace hearts, king diamonds)")
    hand_raw = input("Hero hand (2 cards): ").strip()
    board_raw = input("Board (4 cards): ").strip()
    n_players_raw = input("Players still in (>=2): ").strip()
    method_raw = input("Method [joint_mc/legacy] (default joint_mc): ").strip().lower()
    method = method_raw or "joint_mc"

    hand = _parse_cards(hand_raw, 2)
    board = _parse_cards(board_raw, 4)
    n_players = int(n_players_raw)
    if method == "legacy":
        sims_raw = input("MC sims per river (e.g. 1000): ").strip()
        n_sims = int(sims_raw)
        res = evaluate_turn_probability(
            hand,
            board,
            n_players=n_players,
            n_sims_per_river=n_sims,
            seed=0,
            method="legacy",
        )
    else:
        budget_raw = input("Eval budget cap (default 1800): ").strip()
        ci_raw = input("CI level [0.90/0.95/0.98/0.99] (default 0.98): ").strip()
        hw_raw = input("Target CI half-width (default 0.02): ").strip()
        prior_eq_raw = input("Optional flop prior equity [0..1] (blank=none): ").strip()
        prior_n_raw = input("Optional flop prior n_sims (blank=none): ").strip()

        eval_budget = int(budget_raw) if budget_raw else 1_800
        ci = float(ci_raw) if ci_raw else 0.98
        ci_half_width = float(hw_raw) if hw_raw else 0.02
        prior_eq = float(prior_eq_raw) if prior_eq_raw else None
        prior_n = int(prior_n_raw) if prior_n_raw else None

        res = evaluate_turn_probability(
            hand,
            board,
            n_players=n_players,
            seed=0,
            method="joint_mc",
            eval_budget=eval_budget,
            ci=ci,
            ci_half_width=ci_half_width,
            flop_prior_equity=prior_eq,
            flop_prior_n_sims=prior_n,
        )

    print("p_single (beat):", f"{res['p_single']:.6f}")
    print("p_single (tie):", f"{res['p_single_tie']:.6f}")
    print("p_any (beat):", f"{res['p_any']:.6f}")
    print("p_win:", f"{res['p_win']:.6f}")
    print("p_tie:", f"{res['p_tie']:.6f}")
    print("equity:", f"{res['equity']:.6f}")
    if method == "joint_mc":
        print("n_worlds:", int(res["n_worlds"]))
        print("n_evals:", int(res["n_evals"]))
        print("ci_half_width:", f"{res['ci_half_width']:.6f}")
        print("ci_met:", bool(res["ci_met"]))
        if res.get("used_shrinkage", 0.0) > 0.0:
            print("equity_raw_mc:", f"{res['equity_raw_mc']:.6f}")
            print("prior_weight_eff:", f"{res['prior_weight_eff']:.3f}")


if __name__ == "__main__":
    main()
