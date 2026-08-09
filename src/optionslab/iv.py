"""Implied volatility: inverting Black-Scholes for sigma.

There is no closed form, so this is a root find. The difficulty is not the
root find itself but robustness:

  - In the deep wings vega collapses toward zero, so a raw Newton step
    (price error / vega) explodes and the iteration diverges.
  - Stale or crossed quotes routinely sit outside the no-arbitrage price
    bounds, meaning no solution exists. The solver must detect this and
    return NaN rather than a plausible-looking number.

Strategy: safeguarded Newton. Maintain a bracket [lo, hi] that is known to
contain the root. Take a Newton step; if it lands outside the bracket, take
a bisection step instead. This keeps Newton's speed where the function is
well behaved and guarantees convergence where it is not.

Convergence is tested on PRICE error, not on sigma error, with tolerance
scaled to the tick size.

Benchmark produced here: 3 (round-trip recovery). See BENCHMARKS.md.
"""

from __future__ import annotations

import numpy as np

__all__ = ["price_bounds", "initial_guess", "implied_vol", "implied_vol_chain"]


def price_bounds(F, K, T, df, is_call=True):
    """Return (lower, upper) no-arbitrage bounds on the option price.

    Lower bound is discounted intrinsic on the forward; upper bound is
    df * F for a call and df * K for a put. A quote outside these bounds
    has no implied volatility and must be rejected before any iteration
    begins.
    """
    raise NotImplementedError


def initial_guess(price, F, K, T, df):
    """Brenner-Subrahmanyam ATM approximation, sigma ~ sqrt(2*pi/T) * price / F.

    Cheap and good enough near the money. Accuracy matters little because
    the bracket guarantees convergence regardless; a good guess only saves
    iterations.
    """
    raise NotImplementedError


def implied_vol(price, F, K, T, df, is_call=True, tol=1e-10, max_iter=100):
    """Solve bs_price(F, K, T, sigma, df, is_call) == price for sigma.

    Returns NaN when the price is outside the no-arbitrage bounds.

    Returns
    -------
    (sigma, n_iter) : the solution and the iteration count. The iteration
        count is reported in the benchmark table, so it is returned rather
        than discarded.
    """
    raise NotImplementedError


def implied_vol_chain(df_chain):
    """Vectorised solve across a whole cleaned chain.

    Convert in-the-money quotes to their out-of-the-money counterpart via
    put-call parity before solving. An ITM option is mostly intrinsic
    value, so the time value being inverted is a small fraction of the
    price and the recovered vol is correspondingly imprecise. Record this
    convention in DEVIATIONS.md.
    """
    raise NotImplementedError
