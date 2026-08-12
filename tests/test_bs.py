"""Tests for bs.py — benchmarks 1 and 2.

Tolerances are pre-registered in BENCHMARKS.md and are not weakened to
make a test pass. Reasoning for each choice is in DEVIATIONS.md; the
docstrings below reference the entry rather than restating it.

Grids are fixed rather than random so failures are reproducible.
"""

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from optionslab.bs import bs_greeks, bs_price, forward_from_spot

EPS = np.finfo(float).eps
RATE = 0.03


def _grid(strikes, moneyness, maturities, vols):
    """Cartesian product flattened into broadcast-ready 1-D arrays."""
    rows = list(product(strikes, moneyness, maturities, vols))
    K = np.array([k for k, _, _, _ in rows], dtype=float)
    F = np.array([k * m for k, m, _, _ in rows], dtype=float)
    T = np.array([t for _, _, t, _ in rows], dtype=float)
    sigma = np.array([s for _, _, _, s in rows], dtype=float)
    return F, K, T, sigma, np.exp(-RATE * T)


def wide_grid():
    """160 points, including index-scale notionals and extreme vols."""
    return _grid([100.0, 5000.0], [0.5, 0.8, 1.0, 1.25, 2.0],
                 [1e-3, 0.1, 1.0, 5.0], [0.01, 0.2, 1.0, 2.0])


def interior_grid():
    """27 points with v = sigma*sqrt(T) away from zero, for finite differences."""
    return _grid([100.0], [0.8, 1.0, 1.25], [0.1, 1.0, 3.0], [0.1, 0.3, 0.8])


def _central(f, x, h):
    """Central first difference, error O(h^2) + O(eps/h)."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


# --- Benchmark 1 -----------------------------------------------------------

def test_put_call_parity():
    """C - P = df*(F - K) to 1e-14 relative to notional (DD4).

    An algebraic identity from N(x) + N(-x) = 1, independent of sigma. It
    catches every sign error in d1, d2 and omega.
    """
    F, K, T, sigma, df = wide_grid()
    call = bs_price(F, K, T, sigma, df, is_call=True)
    put = bs_price(F, K, T, sigma, df, is_call=False)

    ratio = np.abs(call - put - df * (F - K)) / (1e-14 * df * np.maximum(F, K))
    assert ratio.max() <= 1.0, (
        f"worst {ratio.max():.2e} x tolerance at index {int(ratio.argmax())}"
    )


def test_no_arbitrage_bounds():
    """df*(F-K)+ <= C <= df*F, and the put analogue.

    iv.py rejects quotes outside these bounds, so validating them here
    means the solver inherits a tested invariant.
    """
    F, K, T, sigma, df = wide_grid()
    call = bs_price(F, K, T, sigma, df, is_call=True)
    put = bs_price(F, K, T, sigma, df, is_call=False)
    tol = 1e-12 * df * np.maximum(F, K)

    assert np.all(call >= df * np.maximum(F - K, 0.0) - tol)
    assert np.all(call <= df * F + tol)
    assert np.all(put >= df * np.maximum(K - F, 0.0) - tol)
    assert np.all(put <= df * K + tol)


# --- Benchmark 2 -----------------------------------------------------------

@pytest.mark.parametrize("is_call", [True, False])
def test_greeks_vs_finite_difference(is_call):
    """Analytic Greeks match numerical derivatives to 1e-6 relative.

    Each Greek bumps the variable its convention differentiates, holding
    fixed what DD6 says is held fixed — theta bumps T with df frozen, rho
    bumps r inside df with F frozen. Gamma differences the analytic delta
    rather than second-differencing the price (DD7). The atol floor
    excludes deep-wing Greeks where the finite difference is rounding
    noise (DD8).
    """
    F, K, T, sigma, df = interior_grid()
    g = bs_greeks(F, K, T, sigma, df, is_call=is_call)
    h = EPS ** (1 / 3)

    checks = {
        "delta": _central(lambda x: bs_price(x, K, T, sigma, df, is_call), F, h * F),
        "gamma": _central(lambda x: bs_greeks(x, K, T, sigma, df, is_call)["delta"], F, h * F),
        "vega": _central(lambda x: bs_price(F, K, T, x, df, is_call), sigma, h * sigma),
        "theta": -_central(lambda x: bs_price(F, K, x, sigma, df, is_call), T, h * T),
        "rho": _central(lambda x: bs_price(F, K, T, sigma, np.exp(-x * T), is_call), RATE, h * RATE),
    }

    for name, numeric in checks.items():
        assert_allclose(numeric, g[name], rtol=1e-6,
                        atol=1e-8 * np.max(np.abs(g[name])),
                        err_msg=f"{name} disagrees with finite difference")


def test_greek_call_put_identities():
    """Gamma, vega and theta are bit-identical for calls and puts.

    All three contain only phi(d1) and no N(.) term, so they cannot depend
    on the option's right; the implementation never applies omega to them.
    Theta's equality is specific to the forward parameterisation (DD6).

    Delta obeys parity differentiated, dC/dF - dP/dF = df — the check that
    catches an omega sign error localised to bs_greeks.
    """
    F, K, T, sigma, df = interior_grid()
    gc = bs_greeks(F, K, T, sigma, df, is_call=True)
    gp = bs_greeks(F, K, T, sigma, df, is_call=False)

    for name in ("gamma", "vega", "theta"):
        assert_array_equal(gc[name], gp[name])
    assert_allclose(gc["delta"] - gp["delta"], df, rtol=0.0, atol=1e-14)


# --- Edge cases and external reference --------------------------------------

def test_degenerate_limits():
    """v -> 0 collapses the price to discounted intrinsic (DD5).

    T = 0 and sigma = 0 are the same limit but reach it by different code
    paths, so both are tested. Gamma at F = K is excluded, where inf is
    the correct answer.
    """
    K, df = 100.0, 0.97
    F = np.array([80.0, 100.0, 120.0])

    for T, sigma in [(0.0, 0.2), (1.0, 0.0)]:
        for is_call, w in [(True, 1.0), (False, -1.0)]:
            price = bs_price(F, K, T, sigma, df, is_call=is_call)
            assert_allclose(price, df * np.maximum(w * (F - K), 0.0), atol=1e-14)

            g = bs_greeks(F, K, T, sigma, df, is_call=is_call)
            assert_allclose(g["delta"], np.where(w * (F - K) > 0, w * df, 0.0), atol=1e-14)
            assert_allclose(g["gamma"][F != K], 0.0, atol=1e-14)
            assert_allclose(g["vega"], 0.0, atol=1e-14)
            assert_allclose(g["theta"], 0.0, atol=1e-14)


def test_greeks_finite_on_wide_grid():
    """No Greek returns NaN or inf anywhere on the wide grid.

    Prices are covered implicitly — a NaN price fails parity. A stray NaN
    in a Greek is not, and is painful to trace once buried in an SVI fit.
    """
    F, K, T, sigma, df = wide_grid()
    for is_call in (True, False):
        for name, value in bs_greeks(F, K, T, sigma, df, is_call=is_call).items():
            assert np.all(np.isfinite(value)), f"{name} produced non-finite values"


def test_hull_worked_example():
    """Reproduce a published worked example.

    Every other test here is internal consistency and would pass on a
    self-consistent implementation of the wrong formula. This is the only
    external check.

    Source: Hull, Options, Futures and Other Derivatives — the worked
    example of the Black-Scholes-Merton model chapter: a European option on
    a non-dividend-paying stock with S = 42, K = 40, r = 0.10, sigma = 0.20,
    T = 0.5, for which the book quotes c = 4.76 and p = 0.81 to two decimals.
    Those are exactly the closed-form values these parameters imply, and the
    implementation returns 4.7594 and 0.8086, so the check is genuine rather
    than circular. The parameters are the standard textbook example; the
    quoted values have not been checked against a printed copy, so this test
    currently verifies the closed form rather than an external source.
    """
    S, K, r, sigma, T = 42.0, 40.0, 0.10, 0.20, 0.5
    F = forward_from_spot(S, r, 0.0, T)
    df = np.exp(-r * T)

    assert_allclose(bs_price(F, K, T, sigma, df, is_call=True), 4.76, atol=5e-3)
    assert_allclose(bs_price(F, K, T, sigma, df, is_call=False), 0.81, atol=5e-3)