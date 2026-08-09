"""Black-Scholes in forward / discount-factor form, with analytic Greeks.

This module is the reference implementation for the whole project. Monte
Carlo must converge to it, the implied vol solver must invert it, and the
surface uses it to move between price space and vol space.

Parameterisation (see DEVIATIONS.md, DD1): all functions take the forward
`F` and the discount factor `df`, never spot / rate / dividend yield.

    call = df * (F * N(d1) - K * N(d2))
    put  = df * (K * N(-d2) - F * N(-d1))

All functions are vectorised: every argument broadcasts under numpy rules,
and there are no Python-level loops.

Benchmarks produced here: 1 (put-call parity), 2 (Greeks vs finite
difference). See BENCHMARKS.md.
"""

from __future__ import annotations

import numpy as np

__all__ = ["d1_d2", "bs_price", "bs_greeks", "forward_from_spot"]


def forward_from_spot(S, r, q, T):
    """Convenience converter, F = S * exp((r - q) * T).

    Provided only for synthetic tests and textbook examples where r and q
    are given. Never used on real chain data, where the forward is inferred
    from the quotes instead (see chain.implied_forward).
    """
    raise NotImplementedError


def d1_d2(F, K, T, sigma):
    """Return (d1, d2), the two arguments of the normal CDF.

    d1 = (log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    Split into its own function because it is where every sign error in a
    Black-Scholes implementation lives, and so it can be tested directly.

    Degenerate inputs (T == 0 or sigma == 0) are handled by the callers,
    not here; this function assumes sigma * sqrt(T) > 0.
    """
    raise NotImplementedError


def bs_price(F, K, T, sigma, df, is_call=True):
    """Undiscounted-forward Black-Scholes price, discounted by `df`.

    Parameters
    ----------
    F, K, T, sigma, df : array_like
        Forward, strike, time to expiry in years, volatility, discount
        factor. All broadcast against one another.
    is_call : array_like of bool
        True for calls, False for puts. Broadcasts with the rest.

    Returns
    -------
    ndarray

    Edge cases that must be handled explicitly rather than falling out of
    the algebra:
      - T == 0        -> payoff on the forward, discounted
      - sigma == 0    -> discounted intrinsic value on the forward
      - deep wings    -> N(d) saturates; check no NaN is produced
    """
    raise NotImplementedError


def bs_greeks(F, K, T, sigma, df, is_call=True):
    """Analytic Greeks.

    Returns
    -------
    dict with keys 'delta', 'gamma', 'vega', 'theta', 'rho'.

    Conventions must be stated in the docstring once chosen and then held
    consistently across the project:
      - delta is with respect to the FORWARD, not spot (DD1). Record the
        spot-delta conversion in DEVIATIONS.md if it is ever needed.
      - vega is per 1.00 of volatility (not per vol point).
      - theta is per year (not per day).

    Note gamma and vega are identical for a call and the put of the same
    strike; that gives a free consistency test.
    """
    raise NotImplementedError
