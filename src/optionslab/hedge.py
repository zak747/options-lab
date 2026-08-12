import numpy as np
import pandas as pd
from scipy.special import ndtr

from optionslab.bs import bs_price, d1_d2
from optionslab.lsm import simulate_spot_paths


def spot_delta(S, K, tau, sigma, r, is_call=True):
    w = np.where(is_call, 1.0, -1.0)
    v = sigma * np.sqrt(np.maximum(tau, 0.0))
    expired = v < 1e-12
    v_safe = np.where(expired, 1.0, v)
    forward = S * np.exp(r * np.maximum(tau, 0.0))
    d1, _ = d1_d2(forward, K, v_safe)
    live_delta = w * ndtr(w * d1)
    settled_delta = np.where(w * (S - K) > 0.0, w, 0.0)
    return np.where(expired, settled_delta, live_delta)


def hedge_pnl(paths, K, T, sigma_price, sigma_hedge, r, dt, is_call=True):
    n_paths, n_nodes = paths.shape
    n_steps = n_nodes - 1
    w = 1.0 if is_call else -1.0

    initial_forward = paths[0, 0] * np.exp(r * T)
    premium = float(bs_price(initial_forward, K, T, sigma_price, np.exp(-r * T), is_call))

    times = np.arange(n_nodes) * dt
    remaining = T - times

    holding = spot_delta(paths[:, 0], K, remaining[0], sigma_hedge, r, is_call)
    cash = premium - holding * paths[:, 0]

    for step in range(1, n_steps):
        cash = cash * np.exp(r * dt)
        new_holding = spot_delta(paths[:, step], K, remaining[step], sigma_hedge, r, is_call)
        cash = cash - (new_holding - holding) * paths[:, step]
        holding = new_holding

    cash = cash * np.exp(r * dt)
    terminal = paths[:, -1]
    payoff = np.maximum(w * (terminal - K), 0.0)
    return cash + holding * terminal - payoff


def hedging_error_study(n_rebalances_grid, S0=100.0, K=100.0, T=1.0,
                        sigma_price=0.20, sigma_hedge=None, sigma_realised=None,
                        r=0.02, n_paths=100_000, seed=0, is_call=True):
    if sigma_hedge is None:
        sigma_hedge = sigma_price
    if sigma_realised is None:
        sigma_realised = sigma_price

    records = []
    for n_steps in n_rebalances_grid:
        rng = np.random.default_rng(seed + n_steps)
        paths = simulate_spot_paths(S0, r, 0.0, T, sigma_realised, n_paths, n_steps,
                                    rng, antithetic=True)
        pnl = hedge_pnl(paths, K, T, sigma_price, sigma_hedge, r, T / n_steps, is_call)
        records.append({"n_rebalances": n_steps, "mean_pnl": float(pnl.mean()),
                        "std_pnl": float(pnl.std(ddof=1)),
                        "mean_standard_error": float(pnl.std(ddof=1) / np.sqrt(n_paths))})

    table = pd.DataFrame(records)
    log_n = np.log(table["n_rebalances"].to_numpy(float))
    log_std = np.log(table["std_pnl"].to_numpy())
    slope, intercept = np.polyfit(log_n, log_std, 1)

    fitted = intercept + slope * log_n
    residual_variance = np.sum((log_std - fitted) ** 2) / max(len(log_n) - 2, 1)
    slope_error = np.sqrt(residual_variance / np.sum((log_n - log_n.mean()) ** 2))
    return table, float(slope), float(slope_error)


def predicted_mismatch_pnl(S0, K, T, sigma_price, sigma_realised, r, is_call=True):
    forward = S0 * np.exp(r * T)
    discount = np.exp(-r * T)
    priced = float(bs_price(forward, K, T, sigma_price, discount, is_call))
    realised = float(bs_price(forward, K, T, sigma_realised, discount, is_call))
    return (priced - realised) * np.exp(r * T)


def vol_mismatch_study(sigma_price, sigma_realised_grid, hedge_at="realised",
                       S0=100.0, K=100.0, T=1.0, r=0.02, n_steps=250,
                       n_paths=100_000, seed=0, is_call=True):
    records = []
    for sigma_realised in sigma_realised_grid:
        sigma_hedge = sigma_realised if hedge_at == "realised" else sigma_price
        rng = np.random.default_rng(seed)
        paths = simulate_spot_paths(S0, r, 0.0, T, sigma_realised, n_paths, n_steps,
                                    rng, antithetic=True)
        pnl = hedge_pnl(paths, K, T, sigma_price, sigma_hedge, r, T / n_steps, is_call)
        predicted = predicted_mismatch_pnl(S0, K, T, sigma_price, sigma_realised, r, is_call)

        records.append({"sigma_realised": sigma_realised, "sigma_hedge": sigma_hedge,
                        "mean_pnl": float(pnl.mean()), "std_pnl": float(pnl.std(ddof=1)),
                        "predicted_pnl": predicted,
                        "mean_standard_error": float(pnl.std(ddof=1) / np.sqrt(n_paths))})
    return pd.DataFrame(records)
