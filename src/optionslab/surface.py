import numpy as np
import pandas as pd
from scipy.optimize import minimize


def svi_total_variance(k, params):
    a, b, rho, m, s = params
    shifted = k - m
    return a + b * (rho * shifted + np.sqrt(shifted ** 2 + s ** 2))


def svi_derivatives(k, params):
    a, b, rho, m, s = params
    shifted = k - m
    root = np.sqrt(shifted ** 2 + s ** 2)
    first = b * (rho + shifted / root)
    second = b * s ** 2 / root ** 3
    return first, second


def _fit_linear_given_shape(k, w, weights, m, s):
    scaled = (k - m) / s
    design = np.column_stack([np.ones_like(scaled), scaled, np.sqrt(scaled ** 2 + 1.0)])
    root_weights = np.sqrt(weights)[:, None]
    solution, _, _, _ = np.linalg.lstsq(design * root_weights,
                                        w * np.sqrt(weights), rcond=None)
    level, skew_term, wing_term = solution

    wing_term = max(wing_term, 1e-8)
    skew_term = np.clip(skew_term, -wing_term, wing_term)

    minimum_variance = level + np.sqrt(wing_term ** 2 - skew_term ** 2)
    if minimum_variance < 0.0:
        level = level - minimum_variance

    fitted = design @ np.array([level, skew_term, wing_term])
    objective = np.sum(weights * (w - fitted) ** 2)
    return objective, (level, skew_term, wing_term)


def fit_svi(k, w, weights=None, n_starts=6, seed=0):
    if weights is None:
        weights = np.ones_like(w)
    k = np.asarray(k, float)
    w = np.asarray(w, float)
    weights = np.asarray(weights, float)

    def shape_objective(shape):
        m, log_s = shape
        s = np.exp(log_s)
        objective, _ = _fit_linear_given_shape(k, w, weights, m, s)
        return objective

    rng = np.random.default_rng(seed)
    spread = max(k.max() - k.min(), 1e-3)
    best_objective = np.inf
    best_shape = None
    starts = [(np.median(k), np.log(0.1 * spread))]
    for _ in range(n_starts - 1):
        starts.append((rng.uniform(k.min(), k.max()),
                       np.log(spread * rng.uniform(0.02, 0.5))))

    for start in starts:
        result = minimize(shape_objective, start, method="Nelder-Mead",
                          options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 4000})
        if result.fun < best_objective:
            best_objective = result.fun
            best_shape = result.x

    m, log_s = best_shape
    s = np.exp(log_s)
    _, (level, skew_term, wing_term) = _fit_linear_given_shape(k, w, weights, m, s)

    b = wing_term / s
    rho = skew_term / wing_term if wing_term > 0 else 0.0
    return np.array([level, b, rho, m, s])


def butterfly_function(k, params):
    w = svi_total_variance(k, params)
    first, second = svi_derivatives(k, params)
    term_one = (1.0 - k * first / (2.0 * w)) ** 2
    term_two = (first ** 2 / 4.0) * (1.0 / w + 0.25)
    return term_one - term_two + second / 2.0


def risk_neutral_density(k, params):
    w = svi_total_variance(k, params)
    g = butterfly_function(k, params)
    d2 = -k / np.sqrt(w) - 0.5 * np.sqrt(w)
    return g * np.exp(-0.5 * d2 ** 2) / np.sqrt(2.0 * np.pi * w)


def check_parameter_constraints(params):
    a, b, rho, m, s = params
    minimum_variance = a + b * s * np.sqrt(max(1.0 - rho ** 2, 0.0))
    return {"b_non_negative": bool(b >= 0.0),
            "rho_in_range": bool(abs(rho) < 1.0),
            "s_positive": bool(s > 0.0),
            "minimum_variance": float(minimum_variance),
            "variance_non_negative": bool(minimum_variance >= 0.0),
            "wing_slope_below_four": bool(b * (1.0 + abs(rho)) <= 4.0)}


def check_butterfly(params, k_range=(-1.5, 1.5), n_points=1001):
    k = np.linspace(k_range[0], k_range[1], n_points)
    g = butterfly_function(k, params)
    return {"min_g": float(g.min()), "n_violations": int((g < 0.0).sum()),
            "k_at_min": float(k[np.argmin(g)])}


def check_calendar(params_by_maturity, k_range=(-1.5, 1.5), n_points=501):
    k = np.linspace(k_range[0], k_range[1], n_points)
    maturities = sorted(params_by_maturity)
    violations = 0
    worst = 0.0
    for earlier, later in zip(maturities[:-1], maturities[1:]):
        w_earlier = svi_total_variance(k, params_by_maturity[earlier])
        w_later = svi_total_variance(k, params_by_maturity[later])
        gap = w_later - w_earlier
        violations += int((gap < 0.0).sum())
        worst = min(worst, float(gap.min()))
    return {"n_violations": violations, "worst_gap": worst}


def fit_summary(k, w, params, T, bid_vol=None, ask_vol=None):
    fitted_variance = svi_total_variance(k, params)
    fitted_vol = np.sqrt(np.maximum(fitted_variance, 0.0) / T)
    market_vol = np.sqrt(np.maximum(w, 0.0) / T)
    residual = fitted_vol - market_vol

    summary = {"rmse_vol_points": float(np.sqrt(np.mean(residual ** 2))),
               "max_abs_error": float(np.abs(residual).max()),
               "n_quotes": int(len(k))}
    if bid_vol is not None and ask_vol is not None:
        inside = (fitted_vol >= bid_vol) & (fitted_vol <= ask_vol)
        summary["fraction_inside_spread"] = float(inside.mean())
    return summary


def butterfly_violations_from_quotes(strikes, prices, use_spread=False,
                                     bids=None, asks=None):
    order = np.argsort(strikes)
    strikes = np.asarray(strikes, float)[order]
    prices = np.asarray(prices, float)[order]

    records = []
    for centre in range(1, len(strikes) - 1):
        low, mid, high = strikes[centre - 1], strikes[centre], strikes[centre + 1]
        left_gap = mid - low
        right_gap = high - mid
        weight_low = right_gap / (left_gap + right_gap)
        weight_high = left_gap / (left_gap + right_gap)

        if use_spread:
            asks_sorted = np.asarray(asks, float)[order]
            bids_sorted = np.asarray(bids, float)[order]
            cost = (weight_low * asks_sorted[centre - 1]
                    + weight_high * asks_sorted[centre + 1]
                    - bids_sorted[centre])
        else:
            cost = (weight_low * prices[centre - 1]
                    + weight_high * prices[centre + 1] - prices[centre])

        if cost < 0.0:
            records.append({"K_low": low, "K_mid": mid, "K_high": high,
                            "cost": float(cost)})
    return pd.DataFrame(records)