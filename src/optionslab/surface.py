"""SVI volatility surface fitting and no-arbitrage diagnostics.

Raw implied vols are noisy and exist only at listed strikes. SVI
(Gatheral) parameterises TOTAL VARIANCE w(k) = sigma^2 * T as a function
of log-moneyness k = ln(K/F), with five parameters per expiry:

    w(k) = a + b * [ rho*(k - m) + sqrt((k - m)^2 + s^2) ]

Fit by weighted least squares. Weight by vega, or by inverse bid-ask
spread — an unweighted fit chases noise in the illiquid wings, where the
quotes deserve the least trust.

Reference: Gatheral & Jacquier (2014), "Arbitrage-free SVI volatility
surfaces", Quantitative Finance 14(1).
"""

import numpy as np

__all__ = [
    "svi_raw",
    "fit_svi",
    "check_butterfly",
    "check_calendar",
    "surface_diagnostics",
]


def svi_raw(k, params):
    """Total variance w(k) under raw SVI. `params` = (a, b, rho, m, s)."""
    raise NotImplementedError


def fit_svi(k, w, weights=None, x0=None):
    """Weighted least squares fit of raw SVI to one expiry.

    Constraints that must be enforced, not hoped for:
        b >= 0, |rho| < 1, s > 0, a + b*s*sqrt(1 - rho^2) >= 0

    The objective is non-convex, so the starting point matters. Fit
    expiries in order and seed each from the previous one.

    Returns (params, rmse_in_vol_points, n_quotes).
    """
    raise NotImplementedError


def check_butterfly(params, k_grid):
    """Durrleman's condition: the risk-neutral density must be positive.

    Returns (n_violations, worst_value, k_at_worst). Target is zero
    violations (benchmark 9).
    """
    raise NotImplementedError


def check_calendar(params_by_expiry, k_grid):
    """Total variance must be non-decreasing in T at fixed log-moneyness.

    A violation is a calendar spread arbitrage. Returns the count and
    where it occurs.
    """
    raise NotImplementedError


def surface_diagnostics(df, params_by_expiry):
    """Summary table for the surface.

    Reported quantities: fit RMSE in vol points, percentage of market
    mids repriced inside the bid-ask spread, butterfly violations,
    calendar violations. Percentage inside the spread is the honest
    measure — RMSE alone says nothing about whether the fit is good
    relative to how tightly the market is actually quoting.
    """
    raise NotImplementedError
