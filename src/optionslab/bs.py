from __future__ import annotations

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

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

__all__ = ["d1_d2", "bs_price", "bs_greeks", "forward_from_spot"]


def forward_from_spot(S, r, q, T):
    F = S * np.exp((r - q) * T)
    return F
    
def _prepare(F, K, T, sigma, df):
    F, K, T, sigma, df = np.broadcast_arrays(F, K, T, sigma, df)
    v = sigma * np.sqrt(T)
    degenerate = v < 1e-12
    v_safe = np.where(degenerate, 1.0, v)
    return F, K, T, sigma, df, v_safe, degenerate

def d1_d2(F, K, v):
    d1 = ((np.log(F / K)) + (0.5 * (v ** 2))) / v
    d2 = d1 - v
    return d1, d2

def bs_price(F, K, T, sigma, df, is_call = True):
    F, K, T, sigma, df, v_safe, degenerate = _prepare(F, K, T, sigma, df)
    d1, d2 = d1_d2(F, K, v_safe)
    w = np.where(is_call, 1.0, -1.0)
    price = w * df * (F * ndtr(w * d1) - K * ndtr(w * d2))
    intrinsic = df * np.maximum(w * (F - K), 0.0)
    return np.where(degenerate, intrinsic, price)

def bs_greeks(F, K, T, sigma, df, is_call=True):
    F, K, T, sigma, df, v_safe, degenerate = _prepare(F, K, T, sigma, df)
    d1, d2 = d1_d2(F, K, v_safe)
    w = np.where(is_call, 1.0, -1.0)
    phi_d1 = np.exp(-0.5 * d1 ** 2) / np.sqrt(2.0 * np.pi)

    # compute
    delta = w * df * ndtr(w * d1)
    gamma = df * phi_d1 / (F * v_safe)
    vega = df * F * phi_d1 * np.sqrt(T)
    theta = -df * F * phi_d1 * sigma ** 2 / (2.0 * v_safe)
    rho = -T * bs_price(F, K, T, sigma, df, is_call)

    # then overwrite the degenerate entries
    itm = w * (F - K) > 0
    delta = np.where(degenerate, np.where(itm, w * df, 0.0), delta)
    gamma = np.where(degenerate, np.where(F == K, np.inf, 0.0), gamma)
    vega = np.where(degenerate, 0.0, vega)
    theta = np.where(degenerate, 0.0, theta)

    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho": rho}
    

