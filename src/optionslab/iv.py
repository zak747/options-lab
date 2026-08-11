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
from scipy.special import ndtr
from optionslab.bs import d1_d2

__all__ = ["price_bounds", "initial_guess", "implied_vol", "implied_vol_chain"]


def price_bounds(F, K, T, df, is_call=True):
    w = np.where(is_call, 1.0, -1.0)
    lower_bound = df * np.maximum(w * (F - K), 0.0)
    upper_bound = np.where(is_call, df * F, df * K)
    return lower_bound, upper_bound


def initial_guess(price, F, K, T, df):
    sigma_0 = np.sqrt((2 * np.pi) / T) * (price / (df * F))
    low = 1e-6
    high = 5.0
    return np.clip(sigma_0, low, high)

def _price_and_vega(F, K, T, sigma, df, w):
    v = sigma * np.sqrt(T)
    d1, d2 = d1_d2(F, K, v)
    price = w * df * (F * ndtr(w * d1) - K * ndtr(w * d2))
    vega = df * F * np.exp(-0.5 * d1 ** 2) * (1.0 / np.sqrt(2.0 * np.pi)) * np.sqrt(T)
    return price, vega

def implied_vol(price, F, K, T, df, is_call=True, tol=1e-15, max_iter=100):
    price, F, K, T, df, is_call = np.broadcast_arrays(
        np.asarray(price, float), np.asarray(F, float), np.asarray(K, float),
        np.asarray(T, float), np.asarray(df, float), np.asarray(is_call))
    w = np.where(is_call, 1.0, -1.0)
    low = 1e-6
    high = 5.0

    lower_bound, upper_bound = price_bounds(F, K, T, df, is_call)
    tolerance = np.maximum(tol * np.abs(price), 1e-16 * df * np.maximum(F, K))

    sigma = np.array(initial_guess(price, F, K, T, df), dtype=float)
    lo = np.full(sigma.shape, low)
    hi = np.full(sigma.shape, high)
    n_iter = np.zeros(sigma.shape, dtype=int)
    result = np.full(sigma.shape, np.nan)

    solvable = (price > lower_bound + tolerance) & (price < upper_bound - tolerance)
    residual_lo = _price_and_vega(F, K, T, lo, df, w)[0] - price
    residual_hi = _price_and_vega(F, K, T, hi, df, w)[0] - price
    active = solvable & (residual_lo < 0.0) & (residual_hi > 0.0)

    for _ in range(max_iter):
        if not active.any():
            break

        model_price, vega = _price_and_vega(F, K, T, sigma, df, w)
        residual = model_price - price

        converged = active & (np.abs(residual) <= tolerance)
        result = np.where(converged, sigma, result)
        active = active & ~converged
        if not active.any():
            break

        n_iter = n_iter + active
        hi = np.where(active & (residual > 0.0), sigma, hi)
        lo = np.where(active & (residual <= 0.0), sigma, lo)

        step_ok = (vega > 1e-300) & np.isfinite(vega)
        vega_safe = np.where(step_ok, vega, 1.0)
        proposal = np.where(step_ok, sigma - residual / vega_safe, np.inf)

        inside = (proposal > lo) & (proposal < hi)
        new_sigma = np.where(active, np.where(inside, proposal, 0.5 * (lo + hi)), sigma)

        stalled = active & (np.abs(new_sigma - sigma) <= 1e-16 * np.maximum(sigma, 1.0))
        result = np.where(stalled, new_sigma, result)
        active = active & ~stalled
        sigma = new_sigma

    return result, n_iter


def implied_vol_chain(chain, price_col="mid", tol=1e-15, max_iter=100):
    out = chain.copy()
    F = out["F"].to_numpy(float)
    K = out["K"].to_numpy(float)
    T = out["T"].to_numpy(float)
    df = out["df"].to_numpy(float)
    price = out[price_col].to_numpy(float)
    is_call = out["is_call"].to_numpy(bool)

    w = np.where(is_call, 1.0, -1.0)
    itm = w * (F - K) > 0.0
    price = np.where(itm, price - w * df * (F - K), price)
    is_call = np.where(itm, ~is_call, is_call)

    sigma, n_iter = implied_vol(price, F, K, T, df, is_call, tol, max_iter)

    out["implied_vol"] = sigma
    out["iv_n_iter"] = n_iter
    out["iv_solved_as_call"] = is_call
    return out