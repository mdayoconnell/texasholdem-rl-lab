


import numpy as np
from typing import Tuple, Optional


def create_deck() -> np.ndarray:

    values = np.arange(2,15)   # card values 2...14 (Ace = 14)
    suits = np.arange(1,5)    # suits 1-4

    V, S = np.meshgrid(values, suits, indexing = "xy")      # two arrays of shape (4,13)
    deck = np.vstack([V.ravel(),S.ravel()])

    return deck

def draw_from_deck(n_players: int, 
                   deck: np.ndarray,
                   rng = None):

    #'''
    # Draws 2*players + 5 cards for each round
    # Returns a (2,N) matrix. 
    # Row = [value, suit]
    # Columns = card #
    #'''

    if rng is None:
        rng = np.random.default_rng()

    if n_players < 1:
        raise ValueError("n_players must be >= 1")
    
    k = 2*n_players + 5
    n = deck.shape[1]   # should be 52
    
    idx = rng.choice(n, size=k, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True

    drawn = deck[:, mask]
    remaining = deck[:, ~mask]


    return drawn, remaining
    



if __name__ == "__main__":
    deck = create_deck()
    drawn, remaining = draw_from_deck(2,deck)
    print("drawn:\n", drawn)

