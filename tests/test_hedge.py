import numpy as np
import pytest
from numpy.testing import assert_allclose

from optionslab.bs import bs_price, forward_from_spot
from optionslab.hedge import (hedge_pnl, hedging_error_study,
                              predicted_mismatch_pnl, spot_delta,
                              vol_mismatch_study)
from optionslab.lsm import simulate_spot_paths

EPS = np.finfo(float).eps


def test_spot_delta_matches_the_derivative_of_price():
    """Spot delta is dV/dS, which is N(d1) for a call — not the forward delta.

    bs_greeks returns dV/dF = df*N(d1) under the forward parameterisation
    (DD6). Hedging trades the underlying, so the ratio needed is
    dV/dS = dV/dF * dF/dS = df*N(d1) * exp(r*tau) = N(d1). Confusing the
    two understates the hedge by the discount factor, which at r = 2% and
    one year is a 2% error in the position.
    """
    S, K, tau, sigma, r = 100.0, 105.0, 0.75, 0.22, 0.03
    step = EPS ** (1 / 3) * S

    def price_at_spot(spot):
        return float(bs_price(forward_from_spot(spot, r, 0.0, tau), K, tau, sigma,
                              np.exp(-r * tau), True))

    numeric = (price_at_spot(S + step) - price_at_spot(S - step)) / (2.0 * step)

    assert_allclose(spot_delta(S, K, tau, sigma, r, True), numeric, rtol=1e-6)


def test_spot_delta_settles_to_the_exercise_indicator():
    """At expiry delta is 1 in the money and 0 out of it.

    The final rebalance uses tau = 0, so an unguarded d1 divides by zero
    and the terminal hedge is undefined.
    """
    K = 100.0
    assert spot_delta(120.0, K, 0.0, 0.2, 0.02, True) == 1.0
    assert spot_delta(80.0, K, 0.0, 0.2, 0.02, True) == 0.0
    assert spot_delta(120.0, K, 0.0, 0.2, 0.02, False) == 0.0
    assert spot_delta(80.0, K, 0.0, 0.2, 0.02, False) == -1.0


def test_hedging_error_exponent():
    """Benchmark 7. std of hedging P&L scales as n^(-1/2).

    Boyle & Emanuel (1980). Over one period the replication error is
    approximately (1/2)*S^2*gamma*sigma^2*dt*(Z^2 - 1), whose variance is
    proportional to dt^2. Summing n such terms with dt = T/n gives a total
    variance proportional to 1/n, hence a standard deviation proportional
    to n^(-1/2).

    The measured exponent is slightly above -0.5 because the result is
    asymptotic; fitting from larger n moves it toward -0.5 monotonically
    (DD28).
    """
    _, slope, slope_error = hedging_error_study([20, 40, 80, 160, 320],
                                                n_paths=40_000, seed=3)

    assert abs(slope + 0.5) < 0.05
    assert slope_error < 0.01


def test_hedging_is_unbiased_when_the_vol_is_right():
    """Hedging at the vol the option was priced at gives zero expected P&L.

    Discreteness introduces variance, not bias. A non-zero mean would
    indicate a self-financing error — cash not carried at r, or a
    rebalance priced at the wrong node.
    """
    S0, K, T, sigma, r = 100.0, 100.0, 1.0, 0.2, 0.02
    n_steps, n_paths = 200, 100_000

    paths = simulate_spot_paths(S0, r, 0.0, T, sigma, n_paths, n_steps,
                                np.random.default_rng(11), antithetic=True)
    pnl = hedge_pnl(paths, K, T, sigma, sigma, r, T / n_steps)

    standard_error = pnl.std(ddof=1) / np.sqrt(n_paths)
    assert abs(pnl.mean()) < 3.0 * standard_error


def test_hedging_at_realised_vol_gives_a_deterministic_profit():
    """Sell at sigma_price, hedge at the vol that actually realises, and the
    P&L is known in advance.

    It equals the difference in Black-Scholes prices compounded to expiry,

        P&L = [V(sigma_price) - V(sigma_realised)] * exp(rT)

    with no path dependence at all. The residual spread is discretisation
    only and vanishes as n grows. This is the cleanest statement of what an
    options market maker is selling: not a view on direction, but the gap
    between the volatility quoted and the volatility delivered.
    """
    table = vol_mismatch_study(0.20, [0.10, 0.20, 0.30], hedge_at="realised",
                               n_steps=500, n_paths=40_000, seed=5)

    for _, row in table.iterrows():
        assert abs(row["mean_pnl"] - row["predicted_pnl"]) < 3.0 * row["mean_standard_error"] + 0.02
        assert row["std_pnl"] < 0.5


def test_hedging_at_implied_vol_has_the_same_mean_but_more_variance():
    """The choice of hedging volatility changes the risk, not the expectation.

    Hedging at the priced volatility leaves the P&L path dependent — it
    accumulates as gamma P&L against realised moves — while hedging at the
    realised volatility locks the answer in. Both have the same
    expectation, which is why the mean is not the whole story and why the
    variance ratio is reported alongside it (DD29).
    """
    realised_hedge = vol_mismatch_study(0.20, [0.10, 0.30], hedge_at="realised",
                                        n_steps=500, n_paths=40_000, seed=5)
    implied_hedge = vol_mismatch_study(0.20, [0.10, 0.30], hedge_at="priced",
                                       n_steps=500, n_paths=40_000, seed=5)

    for realised_row, implied_row in zip(realised_hedge.itertuples(),
                                         implied_hedge.itertuples()):
        combined = 3.0 * (realised_row.mean_standard_error
                          + implied_row.mean_standard_error) + 0.05
        assert abs(realised_row.mean_pnl - implied_row.mean_pnl) < combined
        assert implied_row.std_pnl > 2.0 * realised_row.std_pnl


def test_selling_expensive_vol_makes_money_and_the_reverse_loses():
    """Sign convention check on the whole experiment.

    Short the option at 20% while the world realises 10% and the position
    profits; realise 30% and it loses. If this came out backwards, every
    number in the module would be plausible and wrong.
    """
    table = vol_mismatch_study(0.20, [0.10, 0.30], hedge_at="realised",
                               n_steps=250, n_paths=20_000, seed=7)

    assert table.loc[0, "mean_pnl"] > 0.0
    assert table.loc[1, "mean_pnl"] < 0.0


def test_predicted_pnl_is_zero_when_there_is_no_mismatch():
    """A consistency check on the closed-form prediction itself."""
    assert predicted_mismatch_pnl(100.0, 100.0, 1.0, 0.2, 0.2, 0.02) == 0.0
    assert predicted_mismatch_pnl(100.0, 100.0, 1.0, 0.25, 0.2, 0.02) > 0.0


def test_zero_volatility_hedge_is_exact():
    """With no randomness the hedge replicates exactly, so P&L is zero.

    Every source of error — discreteness, gamma, rebalancing — vanishes
    when the path is deterministic, so any residual is a bookkeeping error
    in the cash account rather than a modelling one.
    """
    S0, K, T, r = 100.0, 90.0, 1.0, 0.03
    n_steps = 50

    paths = simulate_spot_paths(S0, r, 0.0, T, 0.0, 1000, n_steps,
                                np.random.default_rng(1), antithetic=True)
    pnl = hedge_pnl(paths, K, T, 1e-8, 1e-8, r, T / n_steps)

    assert np.abs(pnl).max() < 1e-6


@pytest.mark.parametrize("is_call", [True, False])
def test_put_and_call_hedges_both_converge(is_call):
    """The experiment works for either right.

    Delta differs by df between them (differentiated parity), so a sign
    error in the put branch would leave the call correct and the put
    systematically biased.
    """
    S0, K, T, sigma, r = 100.0, 100.0, 0.5, 0.25, 0.02
    n_steps, n_paths = 200, 60_000

    paths = simulate_spot_paths(S0, r, 0.0, T, sigma, n_paths, n_steps,
                                np.random.default_rng(13), antithetic=True)
    pnl = hedge_pnl(paths, K, T, sigma, sigma, r, T / n_steps, is_call=is_call)

    assert abs(pnl.mean()) < 3.0 * pnl.std(ddof=1) / np.sqrt(n_paths)