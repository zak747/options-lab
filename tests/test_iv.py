from itertools import product

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from optionslab.bs import bs_greeks, bs_price
from optionslab.iv import implied_vol, implied_vol_chain, initial_guess, price_bounds

RATE = 0.03


def _grid(strikes, moneyness, maturities, vols):
    rows = list(product(strikes, moneyness, maturities, vols))
    K = np.array([k for k, _, _, _ in rows], dtype=float)
    F = np.array([k * m for k, m, _, _ in rows], dtype=float)
    T = np.array([t for _, _, t, _ in rows], dtype=float)
    sigma = np.array([s for _, _, _, s in rows], dtype=float)
    return F, K, T, sigma, np.exp(-RATE * T)


def stress_grid():
    return _grid([100.0, 5000.0], [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0],
                 [0.02, 0.1, 1.0, 5.0], [0.05, 0.2, 0.6, 1.5])


@pytest.mark.parametrize("is_call", [True, False])
def test_round_trip_recovery(is_call):
    """Benchmark 3. Price with known sigma, invert, recover to 1e-10.

    Restricted to quotes with vega above 1e-4 of notional (DD11). The
    inversion has a precision floor of roughly eps * V / vega, because the
    price is representable only to eps*V and the sigma error is the price
    error divided by the local slope. Where vega is small the problem is
    ill-conditioned by construction, not by any defect in the solver.

    The grid deliberately spans F/K in [0.5, 2] — a round-trip confined to
    the near-money region would not exercise the safeguard at all.
    """
    F, K, T, sigma, df = stress_grid()
    price = bs_price(F, K, T, sigma, df, is_call=is_call)
    vega = bs_greeks(F, K, T, sigma, df, is_call=is_call)["vega"]
    notional = df * np.maximum(F, K)

    recovered, _ = implied_vol(price, F, K, T, df, is_call=is_call)
    resolvable = np.isfinite(recovered) & (vega > 1e-4 * notional)

    assert resolvable.sum() > 150, "too few resolvable points to be meaningful"
    assert_allclose(recovered[resolvable], sigma[resolvable], rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("is_call", [True, False])
def test_unresolvable_quotes_return_nan(is_call):
    """A price whose time value is below numerical resolution gives NaN (DD10).

    Deep out of the money at short maturity the true price underflows —
    one point on this grid is 5e-134 — and no algorithm recovers sigma from
    it. Returning 0.0 would be a confident wrong answer, since those points
    have true volatilities up to 1.5.
    """
    F, K, T, sigma, df = stress_grid()
    price = bs_price(F, K, T, sigma, df, is_call=is_call)
    lower, _ = price_bounds(F, K, T, df, is_call)

    recovered, _ = implied_vol(price, F, K, T, df, is_call=is_call)
    time_value = price - lower

    assert np.all(~np.isfinite(recovered[time_value <= 0.0]))
    assert np.all(np.isfinite(recovered[time_value > 1e-6 * df * np.maximum(F, K)]))


def test_quotes_outside_bounds_return_nan():
    """No implied volatility exists outside [intrinsic, upper bound].

    Stale prints and crossed markets put a meaningful fraction of real
    quotes here, so the solver must reject rather than iterate to max_iter
    and return the last iterate.
    """
    F, K, T, df = 120.0, 100.0, 1.0, np.exp(-RATE)
    lower, upper = price_bounds(F, K, T, df, True)

    below, _ = implied_vol(float(lower) - 1.0, F, K, T, df, is_call=True)
    above, _ = implied_vol(float(upper) + 1.0, F, K, T, df, is_call=True)
    beyond_bracket, _ = implied_vol(float(upper) * 0.9999, F, K, T, df, is_call=True)

    assert not np.isfinite(below)
    assert not np.isfinite(above)
    assert not np.isfinite(beyond_bracket), "quote implying sigma > 5 must be rejected"


def test_iteration_count_and_convergence():
    """Report the iteration counts recorded in BENCHMARKS.md.

    Bisection alone needs log2((5 - 1e-6) / eps) ~ 53 iterations, so a mean
    near four confirms Newton is doing the work and the safeguard only
    fires where it must.
    """
    F, K, T, sigma, df = stress_grid()
    price = bs_price(F, K, T, sigma, df, is_call=True)
    recovered, n_iter = implied_vol(price, F, K, T, df, is_call=True)

    solved = np.isfinite(recovered)
    assert n_iter[solved].mean() < 10.0
    assert n_iter[solved].max() < 100
    assert np.all(n_iter[solved] > 0)


def test_initial_guess_is_exact_at_the_money_for_small_v():
    """Brenner-Subrahmanyam is a small-v expansion at F = K.

    V ~ df*F*v/sqrt(2*pi) there, so the guess should be accurate to O(v^2)
    and degrade as v grows. Away from the money it ignores moneyness
    entirely and is only required to be inside the bracket.
    """
    F = K = 100.0
    df = 1.0

    for sigma, tol in [(0.05, 1e-3), (0.20, 1e-2)]:
        price = bs_price(F, K, 1.0, sigma, df, is_call=True)
        assert abs(float(initial_guess(price, F, K, 1.0, df)) - sigma) < tol

    wing_price = bs_price(200.0, K, 1.0, 0.3, df, is_call=True)
    guess = float(initial_guess(wing_price, 200.0, K, 1.0, df))
    assert 1e-6 <= guess <= 5.0


def test_price_bounds_match_the_limits():
    """price_bounds returns the sigma -> 0+ and sigma -> inf limits.

    Checked against the pricer at extreme volatilities rather than against
    a restatement of the same formulas, so the test is not circular.
    """
    F, K, T, df = 100.0, 120.0, 1.0, 0.97

    for is_call in (True, False):
        lower, upper = price_bounds(F, K, T, df, is_call)
        assert_allclose(bs_price(F, K, T, 1e-14, df, is_call), lower, atol=1e-12)
        assert_allclose(bs_price(F, K, T, 60.0, df, is_call), upper, rtol=1e-6)


def test_chain_converts_itm_to_otm():
    """Deep ITM and OTM quotes at one strike must give the same volatility (DD12).

    With F = 5500 and K = 4000 the call is worth ~1485, of which ~1480 is
    intrinsic and independent of sigma; the put is worth ~0.28, entirely
    time value. Parity makes them carry identical information, so both must
    solve to the same number and both must solve as puts.
    """
    F, K, T, df, sigma = 5500.0, 4000.0, 0.25, 0.99, 0.22

    chain = pd.DataFrame({
        "F": [F, F], "K": [K, K], "T": [T, T], "df": [df, df],
        "is_call": [True, False],
        "mid": [float(bs_price(F, K, T, sigma, df, is_call=True)),
                float(bs_price(F, K, T, sigma, df, is_call=False))],
    })

    out = implied_vol_chain(chain)

    assert_allclose(out["implied_vol"].to_numpy(), sigma, atol=1e-10)
    assert not out["iv_solved_as_call"].any()


def test_scalar_and_array_inputs_agree():
    """Broadcasting must not change the answer."""
    F, K, T, df, sigma = 100.0, 105.0, 0.5, 0.98, 0.25
    price = bs_price(F, K, T, sigma, df, is_call=True)

    one, _ = implied_vol(float(price), F, K, T, df, is_call=True)
    many, _ = implied_vol(np.full(4, float(price)), F, K, T, df, is_call=True)

    assert_allclose(many, float(one), rtol=0.0, atol=0.0)