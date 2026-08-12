import numpy as np
import pytest
from numpy.testing import assert_allclose

from optionslab.bs import bs_price, forward_from_spot
from optionslab.lsm import (laguerre_basis, lsm_price, polynomial_basis,
                            price_american_put, simulate_spot_paths)


def test_american_call_equals_european_without_dividends():
    """Early exercise is never optimal for a call on a non-dividend payer.

    The strongest correctness check available, because the European value
    is known in closed form. If the regression overstates the continuation
    value the LSM price falls below the European; if it exercises
    spuriously it also falls below. Only a correct backward induction
    reproduces it.
    """
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.25, 1.0
    n_paths, n_steps = 200_000, 50

    rng = np.random.default_rng(3)
    paths = simulate_spot_paths(S0, r, 0.0, T, sigma, n_paths, n_steps, rng,
                                antithetic=True)
    american, standard_error = lsm_price(paths, K, r, T, is_call=True, degree=3)

    forward = forward_from_spot(S0, r, 0.0, T)
    european = float(bs_price(forward, K, T, sigma, np.exp(-r * T), True))

    assert abs(american - european) < 3.0 * standard_error


def test_american_put_exceeds_european():
    """Early exercise has positive value for a put, so the American price
    must strictly dominate the European.

    The gap widens with maturity, because a longer-dated put has more
    opportunity to reach the exercise boundary.
    """
    S0, K, r, sigma = 36.0, 40.0, 0.06, 0.20

    gaps = []
    for T in (1.0, 2.0):
        american, _ = price_american_put(S0, K, r, sigma, T, n_paths=100_000,
                                         seed=5, degree=3)
        forward = forward_from_spot(S0, r, 0.0, T)
        european = float(bs_price(forward, K, T, sigma, np.exp(-r * T), False))
        gaps.append(american - european)
        assert american > european

    assert gaps[1] > gaps[0]


def test_price_is_bounded_below_by_immediate_exercise():
    """An American put is worth at least its intrinsic value.

    LSM values only the exercise opportunities at t_1, ..., t_N, so where
    immediate exercise at t_0 is optimal the regression result falls below
    intrinsic — measured at 9.95 against 10.00 for S = 30. Taking the
    maximum against the t_0 payoff is what makes the returned figure a
    price rather than a continuation value (DD15).
    """
    K, r, sigma, T = 40.0, 0.06, 0.20, 1.0

    for S0 in (30.0, 34.0, 36.0):
        price, _ = price_american_put(S0, K, r, sigma, T, n_paths=50_000,
                                      seed=6, degree=3)
        assert price >= K - S0


def test_basis_richness_raises_the_price_monotonically():
    """LSM returns a low-biased estimate, so a richer basis must not lower it.

    The exercise policy is estimated, hence suboptimal, hence the price is a
    lower bound in expectation. Adding basis functions improves the policy
    and raises the estimate, converging once the continuation value is
    adequately spanned. Measured: 8.472, 8.496, 8.496, 8.504 for degrees
    2 to 5 (DD16).
    """
    S0, K, r, sigma, T = 36.0, 40.0, 0.06, 0.40, 2.0

    prices = [price_american_put(S0, K, r, sigma, T, n_paths=100_000, seed=1,
                                 degree=degree)[0] for degree in (2, 3, 4)]

    assert prices[1] > prices[0]
    assert abs(prices[2] - prices[1]) < 0.01


def test_laguerre_and_polynomial_bases_agree():
    """Two spanning sets of the same degree give the same price.

    The continuation value depends on the span of the basis, not its
    parameterisation, so a material disagreement indicates a conditioning
    problem rather than a modelling choice.
    """
    S0, K, r, sigma, T = 40.0, 40.0, 0.06, 0.40, 1.0

    laguerre, error = price_american_put(S0, K, r, sigma, T, n_paths=100_000,
                                         seed=1, degree=3, basis="laguerre")
    polynomial, _ = price_american_put(S0, K, r, sigma, T, n_paths=100_000,
                                       seed=1, degree=3, basis="polynomial")

    assert abs(laguerre - polynomial) < 5.0 * error


def test_basis_functions_are_linearly_independent():
    """Each basis returns degree + 1 well-conditioned columns.

    The Laguerre weight exp(-x/2) makes conditioning depend on the scale of
    x, which is why spot is normalised by the strike before the regression
    (DD17). Without that, x ~ 40 gives exp(-20) ~ 2e-9 and the design
    matrix collapses.
    """
    x = np.linspace(0.2, 2.5, 500)

    for builder in (laguerre_basis, polynomial_basis):
        for degree in (2, 3, 4):
            design = builder(x, degree)
            assert design.shape == (500, degree + 1)
            assert np.linalg.matrix_rank(design) == degree + 1
            assert np.linalg.cond(design) < 1e6


def test_spot_paths_have_the_correct_risk_neutral_drift():
    """E[S_t] = S_0 * exp((r - q) t) at every node.

    simulate_spot_paths reuses the martingale simulator from mc.py and
    multiplies by a deterministic growth factor, so this confirms the
    factor is applied per node rather than only at maturity.
    """
    S0, r, q, T, sigma = 100.0, 0.06, 0.02, 2.0, 0.3
    n_paths, n_steps = 200_000, 8

    paths = simulate_spot_paths(S0, r, q, T, sigma, n_paths, n_steps,
                                np.random.default_rng(9), antithetic=True)
    times = np.arange(n_steps + 1) * (T / n_steps)
    expected = S0 * np.exp((r - q) * times)

    for step in range(1, n_steps + 1):
        column = paths[:, step]
        tolerance = 4.0 * column.std(ddof=1) / np.sqrt(n_paths)
        assert abs(column.mean() - expected[step]) < tolerance


def test_longstaff_schwartz_table_1():
    """Benchmark 6. Reproduce the American put values of LS (2001) Table 1.

    Longstaff, F. and Schwartz, E. (2001), Valuing American Options by
    Simulation: A Simple Least-Squares Approach, Review of Financial
    Studies 14(1), 113-147, Table 1 (p. 127).

    K = 40, r = 0.06, S in {36, ..., 44}, sigma in {0.2, 0.4}, T in {1, 2}.
    The option is exercisable 50 times per year; price_american_put uses the
    same 50 exercise points per year (n_steps_per_year=50), so both estimates
    target the same discrete-exercise value rather than the continuous limit.

    REFERENCE is the "Simulated American" column and its standard error (the
    figure in parentheses), the LSM estimate the paper obtains from 100,000
    (50,000 + 50,000 antithetic) paths. The finite-difference column is
    deliberately not used: only the simulated column carries a standard error,
    and the tolerance is built from it.

    Agreement is required within three combined standard errors, where the
    combined figure is sqrt(se_ours^2 + se_paper^2). The paper's own standard
    errors set the tolerance; none is chosen here. Failures are collected
    across all twenty cells before asserting, so a systematic disagreement
    down a row or column is visible at once rather than stopping at the first.
    """
    K, r = 40.0, 0.06

    # (S, sigma, T) -> (simulated price, standard error), transcribed from
    # the "Simulated American (s.e.)" column of Table 1.
    REFERENCE = {
        (36.0, 0.20, 1.0): (4.472, 0.010),
        (36.0, 0.20, 2.0): (4.821, 0.012),
        (36.0, 0.40, 1.0): (7.091, 0.020),
        (36.0, 0.40, 2.0): (8.488, 0.024),
        (38.0, 0.20, 1.0): (3.244, 0.009),
        (38.0, 0.20, 2.0): (3.735, 0.011),
        (38.0, 0.40, 1.0): (6.139, 0.019),
        (38.0, 0.40, 2.0): (7.669, 0.022),
        (40.0, 0.20, 1.0): (2.313, 0.009),
        (40.0, 0.20, 2.0): (2.879, 0.010),
        (40.0, 0.40, 1.0): (5.308, 0.018),
        (40.0, 0.40, 2.0): (6.921, 0.022),
        (42.0, 0.20, 1.0): (1.617, 0.007),
        (42.0, 0.20, 2.0): (2.206, 0.010),
        (42.0, 0.40, 1.0): (4.588, 0.017),
        (42.0, 0.40, 2.0): (6.243, 0.021),
        (44.0, 0.20, 1.0): (1.118, 0.007),
        (44.0, 0.20, 2.0): (1.675, 0.009),
        (44.0, 0.40, 1.0): (3.957, 0.017),
        (44.0, 0.40, 2.0): (5.622, 0.021),
    }

    failures = []
    for (S0, sigma, T), (reference, reference_error) in REFERENCE.items():
        ours, our_error = price_american_put(S0, K, r, sigma, T,
                                             n_paths=200_000, seed=1, degree=3)
        combined = np.sqrt(our_error ** 2 + reference_error ** 2)
        if abs(ours - reference) >= 3.0 * combined:
            failures.append(
                f"S={S0:.0f} sigma={sigma:.2f} T={T:.0f}: "
                f"{ours:.3f} vs {reference:.3f}, "
                f"{abs(ours - reference) / combined:.1f} combined s.e."
            )

    assert not failures, "\n".join(failures)