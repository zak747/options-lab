"""Discrete delta-hedging error experiment.

Short a call, delta hedge it n times over its life, accumulate the P&L.
In continuous time the hedge is exact and the P&L is zero. In discrete
time it is not, and Boyle & Emanuel (1980) show the standard deviation of
the hedging error scales as n^(-1/2).

Two experiments:

  1. Fit the exponent on simulated paths (benchmark 7). Everything is
     known here — the pricing vol, the realised vol, the rate — so the
     only source of error is discreteness itself.

  2. Hedge at a volatility different from the one the option was priced
     at, and watch a systematic P&L appear whose sign depends on whether
     realised vol exceeded implied. That is the economics of an options
     market-making book in a single plot, and it is the experiment most
     worth being able to talk through in an interview.
"""

import numpy as np

__all__ = ["hedge_pnl", "hedging_error_study", "vol_mismatch_study"]


def hedge_pnl(paths, K, T, sigma_price, sigma_hedge, r, dt, is_call=True):
    """Run the hedge over each path and return terminal P&L.

    At each rebalance: mark the option at `sigma_price`, compute delta at
    `sigma_hedge`, trade the difference in the underlying, and carry cash
    at `r`. At expiry, settle the option against its payoff.

    Set sigma_hedge = sigma_price for experiment 1.
    """
    raise NotImplementedError


def hedging_error_study(n_rebalances_grid, **kwargs):
    """Fit log(std of P&L) = a + b*log(n). Expect b ~ -0.5 (benchmark 7).

    Returns a DataFrame plus the fitted exponent and its standard error.
    """
    raise NotImplementedError


def vol_mismatch_study(sigma_price, sigma_hedge_grid, **kwargs):
    """Mean and distribution of P&L as the hedging vol is varied.

    The mean P&L should be approximately proportional to the difference in
    variance, integrated against dollar gamma over the life of the option.
    Check the simulated result against that prediction rather than just
    plotting it.
    """
    raise NotImplementedError
