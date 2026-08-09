"""Benchmarks 1 and 2.

Each test below is skipped until the corresponding function exists.
Remove the skip as you implement — do not weaken the tolerance.
"""

import pytest

pytestmark = pytest.mark.skip(reason="bs.py not yet implemented")


def test_put_call_parity():
    """Benchmark 1. C - P = df*(F - K) to better than 1e-14.

    Sweep a grid of F, K, T, sigma including deep ITM and OTM, and short
    and long maturities. This is an algebraic identity, so the only
    permissible error is floating point.
    """
    raise NotImplementedError


def test_greeks_vs_finite_difference():
    """Benchmark 2. Every analytic Greek within 1e-6 relative error of a
    central finite difference of bs_price.

    Choose the bump size deliberately: central differences have O(h^2)
    truncation error and O(eps/h) rounding error, so h ~ eps^(1/3) times
    the scale of the variable is about right.
    """
    raise NotImplementedError


def test_gamma_vega_call_put_identical():
    """Free consistency check — gamma and vega do not depend on the
    right of the option."""
    raise NotImplementedError


def test_zero_time_and_zero_vol_limits():
    """T = 0 returns the discounted payoff; sigma = 0 returns discounted
    intrinsic on the forward. Neither should produce NaN."""
    raise NotImplementedError


def test_textbook_values():
    """A handful of published worked examples, hard-coded, as an
    independent check that the whole formula is right and not merely
    self-consistent."""
    raise NotImplementedError
