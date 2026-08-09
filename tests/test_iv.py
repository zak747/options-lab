"""Benchmark 3."""

import pytest

pytestmark = pytest.mark.skip(reason="iv.py not yet implemented")


def test_round_trip_recovery():
    """Benchmark 3. Price with a known sigma across a grid of moneyness
    and maturity, invert, recover sigma to better than 1e-12.

    Include the deep wings, where naive Newton diverges. If the solver
    only works near the money it has not been tested.
    """
    raise NotImplementedError


def test_rejects_arbitrage_violating_prices():
    """Prices below intrinsic or above the upper bound must return NaN,
    not a number. Silent garbage here contaminates the whole surface."""
    raise NotImplementedError


def test_iteration_count_bounded():
    """Record mean and maximum iteration count across the grid. Both are
    reportable numbers, and a blown maximum is the signal that the
    bracket safeguard is not working."""
    raise NotImplementedError
