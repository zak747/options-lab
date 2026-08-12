import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from optionslab.bs import bs_price, d1_d2


def simulate_terminal(F, T, sigma, n_paths, rng, antithetic=False):
    v = sigma * np.sqrt(T)
    if antithetic:
        n_half = n_paths // 2
        z_half = rng.standard_normal(n_half)
        z = np.concatenate([z_half, -z_half])
    else:
        z = rng.standard_normal(n_paths)
    terminal = F * np.exp(-0.5 * v ** 2 + v * z)
    return terminal


def simulate_terminal_sobol(F, T, sigma, n_paths, seed=0):
    v = sigma * np.sqrt(T)
    n_bits = int(round(np.log2(n_paths)))
    engine = qmc.Sobol(d=1, scramble=True, seed=seed)
    uniforms = engine.random_base2(n_bits).ravel()
    uniforms = np.clip(uniforms, 1e-15, 1.0 - 1e-15)
    z = ndtri(uniforms)
    terminal = F * np.exp(-0.5 * v ** 2 + v * z)
    return terminal


def simulate_paths(F, T, sigma, n_paths, n_steps, rng):
    dt = T / n_steps
    v_step = sigma * np.sqrt(dt)
    z = rng.standard_normal((n_paths, n_steps))
    log_increments = -0.5 * v_step ** 2 + v_step * z
    log_paths = np.cumsum(log_increments, axis=1)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = F
    paths[:, 1:] = F * np.exp(log_paths)
    return paths


def payoff(terminal, K, is_call=True, straddle=False):
    w = np.where(is_call, 1.0, -1.0)
    if straddle:
        return np.abs(terminal - K)
    return np.maximum(w * (terminal - K), 0.0)


def mc_price(F, K, T, sigma, df, is_call=True, n_paths=100_000, rng=None,
             antithetic=False, straddle=False):
    if rng is None:
        rng = np.random.default_rng(0)
    terminal = simulate_terminal(F, T, sigma, n_paths, rng, antithetic)
    discounted = df * payoff(terminal, K, is_call, straddle)
    if antithetic:
        n_half = len(discounted) // 2
        pair_means = 0.5 * (discounted[:n_half] + discounted[n_half:])
        estimate = pair_means.mean()
        standard_error = pair_means.std(ddof=1) / np.sqrt(n_half)
    else:
        estimate = discounted.mean()
        standard_error = discounted.std(ddof=1) / np.sqrt(len(discounted))
    return estimate, standard_error


def control_variate(payoffs, control, control_mean):
    control_centred = control - control.mean()
    payoffs_centred = payoffs - payoffs.mean()
    b_star = np.sum(payoffs_centred * control_centred) / np.sum(control_centred ** 2)
    rho = np.corrcoef(payoffs, control)[0, 1]
    adjusted = payoffs + b_star * (control_mean - control)
    estimate = adjusted.mean()
    standard_error = adjusted.std(ddof=1) / np.sqrt(len(adjusted))
    return estimate, standard_error, b_star, rho


def delta_control(paths, K, T, sigma, is_call=True):
    n_paths, n_nodes = paths.shape
    n_steps = n_nodes - 1
    dt = T / n_steps
    times = np.arange(n_steps) * dt
    remaining = T - times
    w = np.where(is_call, 1.0, -1.0)

    forwards = paths[:, :-1]
    v = sigma * np.sqrt(remaining)
    d1, _ = d1_d2(forwards, K, v)
    deltas = w * ndtr(w * d1)

    increments = paths[:, 1:] - paths[:, :-1]
    control = np.sum(deltas * increments, axis=1)
    return control


def convergence_study(F, K, T, sigma, df, is_call=True, path_counts=None,
                      n_reps=32, seed=0, method="standard"):
    if path_counts is None:
        path_counts = [2 ** k for k in range(10, 21)]
    exact = float(bs_price(F, K, T, sigma, df, is_call))
    rng = np.random.default_rng(seed)

    counts = []
    rmses = []
    for n_paths in path_counts:
        errors = np.empty(n_reps)
        for rep in range(n_reps):
            if method == "sobol":
                terminal = simulate_terminal_sobol(F, T, sigma, n_paths,
                                                   seed=seed * 10_000 + rep)
                estimate = df * payoff(terminal, K, is_call).mean()
            elif method == "antithetic":
                estimate, _ = mc_price(F, K, T, sigma, df, is_call, n_paths, rng,
                                       antithetic=True)
            else:
                estimate, _ = mc_price(F, K, T, sigma, df, is_call, n_paths, rng)
            errors[rep] = estimate - exact
        counts.append(n_paths)
        rmses.append(np.sqrt(np.mean(errors ** 2)))

    counts = np.array(counts, dtype=float)
    rmses = np.array(rmses)
    slope, intercept = np.polyfit(np.log(counts), np.log(rmses), 1)
    return counts, rmses, slope
