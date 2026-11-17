"""Ranking of load-shifting windows inside one 24-hour forecast.

A window is a run of consecutive forecast hours. Its score is the mean of a forecast column
over those hours: the median (`p50`) for a neutral choice, the upper bound (`p95`) for a
cautious one. The lowest score wins; ties go to the earliest start. The dashboard extracts
and the QA agent both rank windows with these functions, so they cannot disagree.
"""

from __future__ import annotations

import numpy as np

RISK_COLUMNS: dict[str, str] = {"median": "p50", "cautious": "p95"}


def window_means(values: np.ndarray, window_hours: int) -> np.ndarray:
    """Mean of every run of `window_hours` consecutive values; entry i starts at position i."""
    series = np.asarray(values, dtype=float)
    if window_hours < 1 or window_hours > series.size:
        raise ValueError(f"window_hours must be between 1 and {series.size}, got {window_hours}")
    totals = np.cumsum(np.concatenate([[0.0], series]))
    return (totals[window_hours:] - totals[:-window_hours]) / window_hours


def best_start(scores: np.ndarray) -> int:
    """Position of the lowest score; the earliest one when several are equal."""
    if np.isnan(scores).any():
        raise ValueError("Cannot rank windows with missing scores")
    return int(np.argmin(scores))


def hindsight_ranks(realized_means: np.ndarray) -> np.ndarray:
    """Rank of each window by its realized mean: 1 is the cleanest, ties share the lower rank."""
    order = np.sort(realized_means)
    return np.searchsorted(order, realized_means, side="left") + 1
