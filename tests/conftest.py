"""Shared fixtures. Synthetic data keeps tests fast and independent of the raw CSV."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_like() -> pd.DataFrame:
    """A tiny raw-schema frame with the quirks clean() must handle."""
    return pd.DataFrame({
        "id": ["a", "b", "b", "c"],                 # 'b' duplicated
        "rated": ["TRUE", "False", "False", "true"],  # mixed case
        "created_at": [1.5e12, 1.5e12, 1.5e12, 1.5e12],
        "last_move_at": [1.5e12, 1.5e12, 1.5e12, 1.5e12],
        "turns": [40, 0, 0, 25],                    # one zero-turn (degenerate)
        "victory_status": ["mate", "resign", "resign", "draw"],
        "winner": ["white", "black", "black", "draw"],
        "increment_code": ["10+0", "5+5", "5+5", "15+15"],
        "white_id": ["p1", "p2", "p2", "p3"],
        "white_rating": [1500, 1600, 1600, 1400],
        "black_id": ["q1", "q2", "q2", "q3"],
        "black_rating": [1450, 1700, 1700, 1400],
        "moves": ["e4 e5", "d4 d5", "d4 d5", "c4 c5"],
        "opening_eco": ["C50", "D02", "D02", "A10"],
        "opening_name": ["Italian", "London", "London", "English"],
        "opening_ply": [5, 3, 3, 2],
    })


@pytest.fixture
def signal_df() -> pd.DataFrame:
    """Synthetic games with real signal so a model can beat the trivial baselines.

    Outcome: the higher-rated side wins most of the time (plus noise).
    Rating: avg_rating is driven by opening_ply and turns (plus noise).
    """
    rng = np.random.default_rng(0)
    n = 600
    white = rng.integers(1000, 2000, n)
    black = rng.integers(1000, 2000, n)
    # Higher-rated usually wins; ~15% upsets.
    upset = rng.random(n) < 0.15
    higher_white = white >= black
    white_wins = higher_white ^ upset
    winner = np.where(white_wins, "white", "black")

    opening_ply = rng.integers(1, 15, n)
    turns = rng.integers(10, 120, n)
    avg_rating = (
        1200 + 30 * opening_ply + 2.0 * turns + rng.normal(0, 80, n)
    )
    # Encode avg_rating back into individual ratings so engineer() is consistent.
    white = (avg_rating + rng.normal(0, 50, n)).astype(int)
    black = (2 * avg_rating - white).astype(int)

    return pd.DataFrame({
        "rated": rng.random(n) < 0.8,
        "turns": turns,
        "victory_status": rng.choice(["mate", "resign", "outoftime"], n),
        "winner": winner,
        "increment_code": rng.choice(["10+0", "5+5", "15+15", "3+2"], n),
        "white_rating": white,
        "black_rating": black,
        "moves": ["e4 e5"] * n,
        "opening_eco": rng.choice(["C50", "D02", "A10", "B01", "C00"], n),
        "opening_name": ["x"] * n,
        "opening_ply": opening_ply,
    })
