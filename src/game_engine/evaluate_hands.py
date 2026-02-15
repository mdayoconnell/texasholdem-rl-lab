import sys
from pathlib import Path

# Ensure `src/` is on sys.path so sibling packages like `utils/` can be imported
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from collections import Counter

from game_engine.init_deck import create_deck, draw_from_deck
from utils.format_suits import cards_to_unicode_list
from utils.format_hands import format_hand


def hand_plus_community(player: int, drawn: np.ndarray) -> np.ndarray:
    # """
    #   Return a (2, 7) matrix = [player's 2 cards | 5 community cards].
    # 
    #    `drawn` has shape (2, 2*n_players + 5)
    #    Player hole cards are stored in order: P0 gets columns 0:2, P1 gets 2:4, etc.
    #    Community cards are the last 5 columns.
    # """

    if drawn.ndim != 2 or drawn.shape[0] != 2:
        raise ValueError(f"drawn must have shape (2, N); got {drawn.shape}")

    n_cols = drawn.shape[1]
    if n_cols < 7:
        raise ValueError(f"drawn must contain at least 7 cards (2 hole + 5 community); got {n_cols}")

    # finds number of players from the format (2*n_players + 5)
    if (n_cols - 5) % 2 != 0:
        raise ValueError(
            f"drawn has {n_cols} columns; expected (2*n_players + 5). (N-5) must be even."
        )
    n_players = (n_cols - 5) // 2

    if not (0 <= player < n_players):
        raise ValueError(f"player must be in [0, {n_players - 1}]; got {player}")

    community = drawn[:, -5:]  # (2, 5)
    start = 2 * player
    hole = drawn[:, start : start + 2]  # (2, 2)

    return np.hstack([hole, community])  # (2, 7)

def best_straight(values):

    #'''
    # Returns the high card of a 5-card straight from np.array "values" (or values_ext if values has an Ace)
    # If no straight returns None
    # Treats A as both HIGH and LOW, note that A2345 will return 5
    #'''


    uniq = set(int(x) for x in values)
    if 14 in uniq:
        uniq.add(1)
    seq = sorted(uniq)

    topcard = None
    run_length = 1
    for i in range(1,len(seq)):
        if seq[i] == seq[i-1] + 1:
            run_length += 1
        else:
            run_length = 1
        if run_length >= 5:
            topcard = seq[i]
    return topcard
    
def evaluate_hand(seven_cards):

    #'''
    # Conventions: handle hand identification by cases. 
    #   Takes in values and suits as (1,7) array
    # Returns a tuple (hand_rank: int, card_value: int, [kickers: ints])
    #   hand_rank goes 1-10: royal flush to high card
    #   card_value goes 2-14: card in pair, Tkind, Fkind, or high card of straight
    #   kickers include remaining cards for tiebreak (or pair in the case of FHouse)
    #'''

    # Helpers

    values = seven_cards[0]
    suits = seven_cards[1]
    suits_count = Counter(suits)

    flush_suits = [suit for suit, c in suits_count.items() if c >= 5]
    flush_suit = flush_suits[0] if flush_suits else None

    # ROYAL FLUSH
    # returns (1,14,[])     -> handrank = 1, topcard = A, no kickers necessary

    if flush_suit is not None:      # if flush is achieved
        flush_vals = [value for value, s in zip(values, suits) if s == flush_suit]
        sf_high = best_straight(flush_vals)
        if sf_high is not None:     # if straight is achieved
            royalflush = {10,11,12,13,14}
            if royalflush.issubset(set(flush_vals)):
                return(1,14,[])
    
    # STRAIGHT FLUSH
    # returns (2, topcard, [])      -> no kickers necessary
            lowflush = {14,2,3,4,5}
            if lowflush.issubset(flush_vals):
                return (2, 5, [])
            return (2, sf_high, [])
        

    # for following hands, need to compute multiplicity from (values)
    values_count = Counter(values)
    fours = sorted([val for val, c in values_count.items() if c == 4])
    threes = sorted([val for val, c in values_count.items() if c == 3])
    pairs = sorted([val for val, c in values_count.items() if c == 2])

    # FOUR OF A KIND
    if fours:
        quad = fours[0]
        kicker = np.asarray([max([val for val in values if val != quad])])
        return(3, quad, kicker)
    
    # FULL HOUSE

    if threes and pairs :
        trips = threes[-1]
        kickers = np.asarray([pairs[-1]])
        return (4,trips, kickers)

    if len(threes) >= 2:
        trips = threes[-1]
        kickers = np.asarray([threes[-2]])
        return (4,trips, kickers)
    
    # FLUSH

    if flush_suit is not None:
        flush_vals = sorted([v for v, s in zip(values, suits) if s == flush_suit], reverse= True)
        top5 = flush_vals[:5]
        kickers = np.asarray(top5[1:])
        return(5, top5[0], kickers)
    
    # STRAIGHT

    straight_high_card = best_straight(values)
    if straight_high_card is not None:
        return (6, straight_high_card, [])
    
    # THREE OF A KIND

    if threes:
        trips = threes[-1]
        without_trips = np.asarray([values[i] for i in range(len(values)) if values[i] != trips])
        kickers = np.asarray([np.sort(without_trips)[-1], np.sort(without_trips)[-2]])
        return (7, trips, kickers)
        
    #TWOPAIR

    if len(pairs) >= 2:
        highpair, lowpair = pairs[-1], pairs[-2]
        without_twopair = np.asarray([values[i] for i in range(len(values)) if values[i] != highpair and values[i] != lowpair])
        kicker_item = max(without_twopair)
        return (8, highpair, np.asarray([lowpair, kicker_item]))

    # PAIR

    if len(pairs) == 1:
           pair = pairs[-1]
           without_pair = np.asarray([values[i] for i in range(len(values)) if values[i] != pair])
           ordering = np.sort(without_pair)[::-1]
           kickers = ordering[:3]
           return(9,pair, kickers)

    
    # HIGH CARD

    else:
        ordering = np.sort(values)[::-1]
        highcard = ordering[0]
        kickers = ordering[1:5]
        return (10, highcard, kickers)
    

    return ValueError("Could not match hole plus community cards with a ranked hand")


def _as_int_array(x) -> np.ndarray:
    """Normalize kickers into a 1D numpy int array."""
    if x is None:
        return np.array([], dtype=int)
    arr = np.asarray(x, dtype=int)
    return arr.ravel()


def _compare_evals(a, b) -> int:
    """Compare two hand evals.

    Each eval is (hand_rank, topcard, kickers).
    Returns:
      -1 if a is better
       0 if tie
      +1 if b is better
    """
    ra, ta, ka = a
    rb, tb, kb = b

    # 1) Hand rank: lower is better
    if ra != rb:
        return -1 if ra < rb else 1

    # 2) Primary value (topcard / made-hand value): higher is better
    ta_i = int(ta)
    tb_i = int(tb)
    if ta_i != tb_i:
        return -1 if ta_i > tb_i else 1

    # 3) Kickers: lexicographic compare, higher is better at first difference
    ka_arr = _as_int_array(ka)
    kb_arr = _as_int_array(kb)

    m = max(len(ka_arr), len(kb_arr))
    if len(ka_arr) < m:
        ka_arr = np.pad(ka_arr, (0, m - len(ka_arr)), constant_values=-1)
    if len(kb_arr) < m:
        kb_arr = np.pad(kb_arr, (0, m - len(kb_arr)), constant_values=-1)

    for x, y in zip(ka_arr, kb_arr):
        if x != y:
            return -1 if x > y else 1

    return 0


def evaluate_winner(hand_evals):
    """Determine winner(s) among a set of player hand evaluations.

    Input:
      hand_evals: list of tuples (hand_rank, topcard, kickers)
        where index i corresponds to player i.

    Output:
      (winners, best_eval)
        winners: list[int] of player indices (multiple if tie)
        best_eval: the best evaluation tuple

    Notes:
      - Two-pair kickers are expected as [lowpair, kicker].
      - For all other hand classes, kickers should already be in descending tie-break order.
    """
    if len(hand_evals) == 0:
        raise ValueError("hand_evals must be non-empty")

    best_idx = 0
    best_eval = hand_evals[0]
    winners = [0]

    for i in range(1, len(hand_evals)):
        cmp = _compare_evals(hand_evals[i], best_eval)
        if cmp < 0:
            best_eval = hand_evals[i]
            best_idx = i
            winners = [i]
        elif cmp == 0:
            winners.append(i)

    return winners, best_eval


if __name__ == "__main__":

    n_players = 2

    deck = create_deck()
    drawn = draw_from_deck(n_players,deck)[0]

    community = drawn[:, -5:]
    community_fmt = cards_to_unicode_list(community)
    print("Community cards:", community_fmt)

    evals = []
    hole_fmts = []
    for i in range(n_players):
        cards = hand_plus_community(i, drawn)
        ev = evaluate_hand(cards)
        evals.append(ev)

        hole = cards[:, :2]
        hole_fmt = cards_to_unicode_list(hole)
        hole_fmts.append(hole_fmt)
        print("Player {}'s hole cards: {}".format(i, hole_fmt))

    winners, best_ev = evaluate_winner(evals)
    best_text = format_hand(best_ev, tiebreak=True)

    if len(winners) == 1:
        w = winners[0]
        print(f"Winner: Player {w} — {best_text} — hole {hole_fmts[w]}")
    else:
        holes = {w: hole_fmts[w] for w in winners}
        print(f"Tie between players {winners} — {best_text} — holes {holes}")
