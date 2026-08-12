import numpy as np
import pytest
from numpy.testing import assert_allclose

from optionslab.bs import bs_price
from optionslab.mc import (control_variate, convergence_study, delta_control,
                           mc_price, payoff, simulate_paths, simulate_terminal,
                           simulate_terminal_sobol)

RATE = 0.03


def test_terminal_is_a_martingale():
    """E[S_T] = F to within three standard errors.

    Under the T-forward measure the forward price is a martingale, so the
    -0.5*sigma^2*T drift must exactly offset the Jensen effect of
    exponentiating. Omitting it is the most common error in a GBM simulator
    and produces a price biased high but still plausible.
    """
    F, T, sigma, n_paths = 100.0, 1.0, 0.3, 500_000
    terminal = simulate_terminal(F, T, sigma, n_paths, np.random.default_rng(0))

    standard_error = terminal.std(ddof=1) / np.sqrt(n_paths)
    assert abs(terminal.mean() - F) < 3.0 * standard_error


def test_paths_are_martingale_at_every_node():
    """Each column of simulate_paths has mean F, not just the last.

    Catches a drift applied per-path rather than per-step, which the
    terminal-only check would miss for n_steps > 1. Step 0 is excluded: all
    paths start at F, so the column has zero variance.
    """
    F, T, sigma = 100.0, 2.0, 0.25
    paths = simulate_paths(F, T, sigma, 200_000, 12, np.random.default_rng(1))

    assert paths[:, 0].min() == paths[:, 0].max() == F
    for step in range(1, paths.shape[1]):
        column = paths[:, step]
        assert abs(column.mean() - F) < 4.0 * column.std(ddof=1) / np.sqrt(len(column))


@pytest.mark.parametrize("is_call", [True, False])
def test_mc_agrees_with_closed_form(is_call):
    """The MC estimate lies within three standard errors of bs_price.

    A single check that catches a missing drift term, a missing discount
    factor and a wrong payoff sign simultaneously.
    """
    F, K, T, sigma = 100.0, 105.0, 1.0, 0.2
    df = np.exp(-RATE * T)
    exact = float(bs_price(F, K, T, sigma, df, is_call))

    estimate, standard_error = mc_price(F, K, T, sigma, df, is_call, 1_000_000,
                                        np.random.default_rng(2))

    assert abs(estimate - exact) < 3.0 * standard_error


def test_convergence_exponent():
    """Benchmark 4. RMSE ~ n^b with b = -0.50 +/- 0.02.

    Stronger than it appears: the slope is distorted by a biased estimator
    or by correlated seeding across repetitions, neither of which shows up
    as an obviously wrong price at any single n.
    """
    F, K, T, sigma = 100.0, 100.0, 1.0, 0.2
    df = np.exp(-RATE * T)
    path_counts = [2 ** k for k in range(10, 19)]

    _, _, slope = convergence_study(F, K, T, sigma, df, True, path_counts,
                                    n_reps=24, seed=1)

    assert abs(slope + 0.5) < 0.02, f"exponent {slope:.4f} is not -0.5"


def test_sobol_converges_faster_than_pseudorandom():
    """Scrambled Sobol beats n^(-1/2) on a one-dimensional payoff.

    Scrambling matters: plain Sobol is deterministic and has no standard
    error at all, so independent randomisations are what make the RMSE
    estimable.
    """
    F, K, T, sigma = 100.0, 100.0, 1.0, 0.2
    df = np.exp(-RATE * T)
    path_counts = [2 ** k for k in range(10, 19)]

    _, _, slope = convergence_study(F, K, T, sigma, df, True, path_counts,
                                    n_reps=24, seed=1, method="sobol")

    assert slope < -0.75, f"QMC exponent {slope:.4f} is no better than pseudorandom"


def test_control_variate_matches_predicted_reduction():
    """Benchmark 5. Achieved VRF agrees with 1/(1-rho^2) to 5%.

    Var(V + b*(mu - X)) is minimised at b* = Cov(Y,X)/Var(X), giving a
    variance reduction factor of exactly 1/(1-rho^2). Testing the identity
    rather than the magnitude validates the b* estimation and the variance
    accounting at once — a wrong b* still reduces variance, just not
    optimally, and would pass a "did it improve" check.
    """
    F, K, T, sigma, n_paths = 100.0, 100.0, 1.0, 0.2, 200_000
    df = np.exp(-RATE * T)

    terminal = simulate_terminal(F, T, sigma, n_paths, np.random.default_rng(7))
    payoffs = df * payoff(terminal, K, True)
    plain_error = payoffs.std(ddof=1) / np.sqrt(n_paths)

    _, adjusted_error, _, rho = control_variate(payoffs, terminal, F)

    achieved = (plain_error / adjusted_error) ** 2
    predicted = 1.0 / (1.0 - rho ** 2)
    assert_allclose(achieved, predicted, rtol=0.05)


def test_delta_control_beats_terminal_control():
    """The discrete delta-hedge P&L is a far better control than S_T.

    Its expectation is zero because the forward is a martingale and the
    delta at each node is measurable at that node. It removes the component
    of the payoff explained by first-order moves, leaving only gamma P&L —
    hence correlation above 0.99 and a VRF two orders of magnitude larger.
    The same quantity is the object of study in Phase 8.
    """
    F, K, T, sigma, n_paths = 100.0, 100.0, 1.0, 0.2, 200_000
    df = np.exp(-RATE * T)

    paths = simulate_paths(F, T, sigma, n_paths, 50, np.random.default_rng(7))
    payoffs = df * payoff(paths[:, -1], K, True)
    control = delta_control(paths, K, T, sigma, True)

    plain_error = payoffs.std(ddof=1) / np.sqrt(n_paths)
    estimate, adjusted_error, _, rho = control_variate(payoffs, control, 0.0)

    exact = float(bs_price(F, K, T, sigma, df, True))
    assert abs(estimate - exact) < 3.0 * adjusted_error
    assert rho > 0.99
    assert (plain_error / adjusted_error) ** 2 > 50.0


def test_antithetic_helps_the_call_and_hurts_the_straddle():
    """Antithetic variates require a monotone payoff (DD14).

    Var((Y(Z) + Y(-Z))/2) = Var(Y)*(1 + rho_anti)/2, so at equal payoff
    evaluations the VRF is exactly 1/(1 + rho_anti). A call is monotone in
    Z, giving rho_anti < 0 and a modest gain. A straddle is very nearly
    symmetric in Z, so rho_anti is strongly positive and the technique is
    actively harmful.

    The failure case is the point. Reporting only the call would make the
    technique look unconditionally useful, which it is not.
    """
    F, K, T, sigma, n_paths = 100.0, 100.0, 1.0, 0.2, 400_000
    df = np.exp(-RATE * T)
    v = sigma * np.sqrt(T)

    for straddle, expect_gain in [(False, True), (True, False)]:
        _, plain_error = mc_price(F, K, T, sigma, df, True, n_paths,
                                  np.random.default_rng(11), antithetic=False,
                                  straddle=straddle)
        _, anti_error = mc_price(F, K, T, sigma, df, True, n_paths,
                                 np.random.default_rng(11), antithetic=True,
                                 straddle=straddle)

        z = np.random.default_rng(3).standard_normal(n_paths // 2)
        up = payoff(F * np.exp(-0.5 * v ** 2 + v * z), K, True, straddle)
        down = payoff(F * np.exp(-0.5 * v ** 2 - v * z), K, True, straddle)
        rho_anti = np.corrcoef(up, down)[0, 1]

        achieved = (plain_error / anti_error) ** 2
        assert_allclose(achieved, 1.0 / (1.0 + rho_anti), rtol=0.05)
        assert (achieved > 1.0) == expect_gain


def test_antithetic_pairs_are_exact_negatives():
    """Paired draws satisfy S_up * S_down = F^2 * exp(-v^2) exactly.

    If the pairing is broken the estimator is still unbiased and the price
    still looks right — only the variance is wrong — so this is checked
    directly rather than inferred from the VRF.

    The product identity is used rather than comparing log returns, which
    pass through zero and so admit no meaningful relative tolerance.
    """
    F, T, sigma, n_paths = 100.0, 1.0, 0.2, 1000
    terminal = simulate_terminal(F, T, sigma, n_paths, np.random.default_rng(5),
                                 antithetic=True)

    v = sigma * np.sqrt(T)
    n_half = n_paths // 2
    products = terminal[:n_half] * terminal[n_half:]
    assert_allclose(products, F ** 2 * np.exp(-v ** 2), rtol=1e-14)


def test_results_are_reproducible():
    """Two runs from the same seed agree bitwise.

    Every stochastic result in the repo must be reproducible, which
    requires an explicit Generator rather than the np.random global state.
    """
    F, K, T, sigma = 100.0, 100.0, 1.0, 0.2
    df = np.exp(-RATE * T)

    first, _ = mc_price(F, K, T, sigma, df, True, 50_000, np.random.default_rng(42))
    second, _ = mc_price(F, K, T, sigma, df, True, 50_000, np.random.default_rng(42))

    assert first == second


def test_sobol_requires_power_of_two():
    """random_base2 balances the sequence only at powers of two.

    Truncating a Sobol sequence at arbitrary n destroys its uniformity and
    silently degrades convergence back toward n^(-1/2).
    """
    terminal = simulate_terminal_sobol(100.0, 1.0, 0.2, 4096, seed=0)
    assert len(terminal) == 4096
    assert abs(terminal.mean() - 100.0) < 0.5