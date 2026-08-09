"""Monte Carlo pricing and variance reduction.

Simulating a European option is pointless as a pricing exercise — the
closed form already exists. That is exactly why it is useful: the true
answer is known, so the effect of each variance reduction technique can be
*measured* rather than asserted.

Under the risk-neutral measure with forward F,

    S_T = F * exp(-sigma^2*T/2 + sigma*sqrt(T)*Z),   Z ~ N(0,1)

For Europeans only the terminal value is needed; no time stepping.
Path simulation is required by lsm.py and hedge.py.
"""

import numpy as np

__all__ = [
    "simulate_terminal",
    "simulate_paths",
    "mc_price",
    "control_variate",
    "convergence_study",
]


def simulate_terminal(F, T, sigma, n_paths, rng, antithetic=False):
    """Draw terminal values S_T.

    With `antithetic=True`, pair each Z with -Z and return n_paths values
    in total (i.e. draw n_paths//2 normals). Antithetic helps when the
    payoff is monotone in Z; it does very little for a straddle, whose
    payoff is symmetric. Demonstrating the case where it fails is a
    better result than showing only cases where it works.
    """
    raise NotImplementedError


def simulate_paths(F, T, sigma, n_paths, n_steps, rng, antithetic=False):
    """Full GBM paths on a uniform grid. Shape (n_paths, n_steps + 1).

    Needed by lsm.py (exercise decisions at intermediate dates) and
    hedge.py (rebalancing).
    """
    raise NotImplementedError


def mc_price(payoffs, df):
    """Return (price, standard_error).

    Standard error is df * std(payoffs, ddof=1) / sqrt(n). Always return
    it — an MC price without an error bar is not a result.
    """
    raise NotImplementedError


def control_variate(payoffs, control, control_mean):
    """Optimal-coefficient control variate.

        b* = Cov(Y, X) / Var(X)
        Y_cv = Y - b*(X - E[X])

    The theoretical variance reduction factor is 1/(1 - rho^2). Return the
    empirical rho and the achieved VRF as well as the estimate, so that
    the two can be checked against each other — that agreement is
    benchmark 5, and it is a genuine internal consistency test rather
    than a restatement.

    Start with X = S_T (E[X] = F). Then try a delta-based control, which
    is substantially better.

    Returns
    -------
    (estimate, se, b_star, rho, vrf)
    """
    raise NotImplementedError


def convergence_study(pricer, analytic, n_grid, n_reps, rng):
    """RMSE against the analytic price as a function of path count.

    For each N in `n_grid`, run `n_reps` independent estimates, take the
    RMSE against `analytic`, then fit

        log(RMSE) = a + b*log(N)

    Plain Monte Carlo should give b ~ -0.5 (benchmark 4). Scrambled Sobol
    should do materially better — use scipy.stats.qmc.Sobol with
    scramble=True so that error can still be estimated by independent
    randomisations.

    Returns a DataFrame with columns: n_paths, rmse, mean_se — plus the
    fitted exponent.
    """
    raise NotImplementedError
