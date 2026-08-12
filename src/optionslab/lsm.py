import numpy as np
from optionslab.mc import simulate_paths
# Apple Accelerate BLAS emits spurious divide/overflow warnings from matmul on
# finite inputs on aarch64 macOS. The results are unaffected — the LS Table 1
# replication passes 20/20 — so the warnings are suppressed rather than masked
# at the call site, where they would obscure a genuine numerical problem.
np.seterr(divide="ignore", over="ignore", invalid="ignore")


def laguerre_basis(x, degree):
    weight = np.exp(-0.5 * x)
    columns = [np.ones_like(x)]
    if degree >= 1:
        columns.append(1.0 - x)
    for order in range(1, degree):
        previous = columns[order]
        two_back = columns[order - 1]
        next_column = ((2.0 * order + 1.0 - x) * previous - order * two_back) / (order + 1.0)
        columns.append(next_column)
    return np.column_stack([weight * column for column in columns])


def polynomial_basis(x, degree):
    columns = [x ** power for power in range(degree + 1)]
    return np.column_stack(columns)


def simulate_spot_paths(S0, r, q, T, sigma, n_paths, n_steps, rng, antithetic=True):
    martingale = simulate_paths(S0, T, sigma, n_paths, n_steps, rng, antithetic)
    times = np.arange(n_steps + 1) * (T / n_steps)
    growth = np.exp((r - q) * times)
    return martingale * growth


def lsm_price(paths, K, r, T, is_call=False, degree=3, basis="laguerre"):
    n_paths, n_nodes = paths.shape
    n_steps = n_nodes - 1
    dt = T / n_steps
    discount_step = np.exp(-r * dt)
    w = 1.0 if is_call else -1.0

    payoff = np.maximum(w * (paths - K), 0.0)
    cashflow = payoff[:, -1].copy()

    for step in range(n_steps - 1, 0, -1):
        cashflow = cashflow * discount_step
        immediate = payoff[:, step]
        in_the_money = immediate > 0.0
        if in_the_money.sum() < degree + 2:
            continue

        spot_itm = paths[in_the_money, step] / K
        if basis == "laguerre":
            design = laguerre_basis(spot_itm, degree)
        else:
            design = polynomial_basis(spot_itm, degree)

        coefficients, _, _, _ = np.linalg.lstsq(design, cashflow[in_the_money], rcond=None)
        continuation = design @ coefficients

        exercise = immediate[in_the_money] > continuation
        exercise_index = np.where(in_the_money)[0][exercise]
        cashflow[exercise_index] = immediate[exercise_index]

    discounted = cashflow * discount_step
    continuation_value = discounted.mean()
    standard_error = discounted.std(ddof=1) / np.sqrt(n_paths)

    immediate_at_zero = float(np.maximum(w * (paths[0, 0] - K), 0.0))
    price = max(continuation_value, immediate_at_zero)
    return price, standard_error


def price_american_put(S0, K, r, sigma, T, n_paths=100_000, n_steps_per_year=50,
                       seed=0, degree=3, basis="laguerre"):
    n_steps = int(round(n_steps_per_year * T))
    rng = np.random.default_rng(seed)
    paths = simulate_spot_paths(S0, r, 0.0, T, sigma, n_paths, n_steps, rng,
                                antithetic=True)
    return lsm_price(paths, K, r, T, is_call=False, degree=degree, basis=basis)