import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optionslab.bs import bs_price, bs_greeks, forward_from_spot
from optionslab.hedge import hedge_pnl, hedging_error_study, vol_mismatch_study
from optionslab.lsm import price_american_put, simulate_spot_paths
from optionslab.mc import convergence_study
from optionslab.surface import fit_svi, svi_total_variance, check_butterfly

FIGURES = Path(__file__).resolve().parents[1] / "figures"
INK = "#1a1a1a"
ACCENT = "#c1440e"
MUTED = "#7a7a7a"


def apply_style():
    plt.rcParams.update({
        "figure.figsize": (7.0, 4.6), "figure.dpi": 150,
        "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "legend.frameon": False, "lines.linewidth": 1.4,
        "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    })


def figure_mc_convergence():
    F, K, T, sigma, df = 100.0, 100.0, 1.0, 0.2, np.exp(-0.03)
    path_counts = [2 ** power for power in range(10, 19)]

    fig, ax = plt.subplots()
    styles = {"standard": (ACCENT, "o"), "antithetic": ("#2b6cb0", "s"),
              "sobol": ("#276749", "^")}
    for method, (colour, marker) in styles.items():
        counts, rmses, slope = convergence_study(F, K, T, sigma, df, True,
                                                 path_counts, n_reps=24, seed=1,
                                                 method=method)
        ax.loglog(counts, rmses, marker=marker, color=colour, markersize=4,
                  label=f"{method}  (exponent {slope:+.3f})")

    reference = rmses[0] * (np.array(path_counts) / path_counts[0]) ** -0.5
    ax.loglog(path_counts, reference * 3.0, "--", color=MUTED, linewidth=1.0,
              label="slope $-1/2$")

    ax.set_xlabel("paths $N$")
    ax.set_ylabel("RMSE against closed form")
    ax.set_title("Benchmark 4 — Monte Carlo convergence, ATM European call")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()
    fig.savefig(FIGURES / "mc_convergence.png")
    plt.close(fig)


def figure_hedging_error():
    grid = [10, 20, 40, 80, 160, 320, 640]
    table, slope, slope_error = hedging_error_study(grid, n_paths=60_000, seed=0)

    counts = table["n_rebalances"].to_numpy(float)
    stds = table["std_pnl"].to_numpy()

    fig, ax = plt.subplots()
    ax.loglog(counts, stds, "o", color=ACCENT, markersize=5, label="simulated")
    fitted = np.exp(np.polyval(np.polyfit(np.log(counts), np.log(stds), 1),
                               np.log(counts)))
    ax.loglog(counts, fitted, "-", color=ACCENT, linewidth=1.2,
              label=f"fit: exponent ${slope:+.4f} \\pm {slope_error:.4f}$")
    ax.loglog(counts, stds[0] * (counts / counts[0]) ** -0.5, "--", color=MUTED,
              linewidth=1.0, label="Boyle–Emanuel $n^{-1/2}$")

    ax.set_xlabel("rebalances $n$")
    ax.set_ylabel("std of terminal P&L")
    ax.set_title("Benchmark 7 — discrete delta-hedging error")
    ax.grid(True, which="both", alpha=0.4)
    ax.legend()
    fig.savefig(FIGURES / "hedging_error.png")
    plt.close(fig)


def figure_vol_mismatch():
    sigma_price = 0.20
    realised_grid = [0.10, 0.15, 0.20, 0.25, 0.30]

    realised_hedge = vol_mismatch_study(sigma_price, realised_grid,
                                        hedge_at="realised", n_steps=400,
                                        n_paths=40_000, seed=5)
    priced_hedge = vol_mismatch_study(sigma_price, realised_grid,
                                      hedge_at="priced", n_steps=400,
                                      n_paths=40_000, seed=5)

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.2))

    left.errorbar(realised_grid, realised_hedge["mean_pnl"],
                  yerr=realised_hedge["std_pnl"], fmt="o", color=ACCENT,
                  capsize=3, markersize=5, label="hedged at realised vol")
    left.errorbar(np.array(realised_grid) + 0.004, priced_hedge["mean_pnl"],
                  yerr=priced_hedge["std_pnl"], fmt="s", color="#2b6cb0",
                  capsize=3, markersize=5, alpha=0.8,
                  label="hedged at priced vol")
    left.plot(realised_grid, realised_hedge["predicted_pnl"], "--", color=MUTED,
              linewidth=1.0, label=r"$[V(\sigma_p)-V(\sigma_r)]e^{rT}$")
    left.axhline(0.0, color=MUTED, linewidth=0.6)
    left.axvline(sigma_price, color=MUTED, linewidth=0.6, linestyle=":")
    left.set_xlabel(r"realised volatility $\sigma_r$")
    left.set_ylabel("terminal P&L")
    left.set_title(r"Short ATM call sold at $\sigma_p$ = 20%")
    left.grid(True, alpha=0.4)
    left.legend(fontsize=8)

    S0, K, T, r = 100.0, 100.0, 1.0, 0.02
    for sigma_realised, colour in [(0.10, "#276749"), (0.30, ACCENT)]:
        paths = simulate_spot_paths(S0, r, 0.0, T, sigma_realised, 40_000, 400,
                                    np.random.default_rng(5), antithetic=True)
        pnl = hedge_pnl(paths, K, T, sigma_price, sigma_price, r, T / 400)
        right.hist(pnl, bins=80, histtype="step", color=colour, density=True,
                   label=rf"$\sigma_r$ = {sigma_realised:.0%}, hedged at 20%")
    right.axvline(0.0, color=MUTED, linewidth=0.6)
    right.set_xlabel("terminal P&L")
    right.set_ylabel("density")
    right.set_title("P&L distribution when hedging at the priced vol")
    right.grid(True, alpha=0.4)
    right.legend(fontsize=8)

    fig.savefig(FIGURES / "vol_mismatch.png")
    plt.close(fig)


def figure_lsm_replication():
    path = Path("data/processed/lsm_replication.csv")
    if not path.exists():
        print("  skipped lsm_replication — run scripts/run_lsm_replication.py first")
        return

    table = pd.read_csv(path)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.2))

    limits = [table[["ours", "paper"]].min().min() - 0.3,
              table[["ours", "paper"]].max().max() + 0.3]
    left.plot(limits, limits, "--", color=MUTED, linewidth=1.0)
    left.errorbar(table["paper"], table["ours"], yerr=3.0 * table["combined_se"],
                  fmt="o", color=ACCENT, capsize=2, markersize=4)
    left.set_xlabel("Longstaff & Schwartz (2001), Table 1")
    left.set_ylabel("this implementation")
    left.set_title("Benchmark 6 — American put values")
    left.grid(True, alpha=0.4)

    right.axhspan(-3.0, 3.0, color="#e6e6e6", alpha=0.6, label=r"$\pm 3$ s.e.")
    right.axhline(0.0, color=MUTED, linewidth=0.6)
    right.plot(range(len(table)), table["z"], "o", color=ACCENT, markersize=4)
    right.set_xlabel("table cell")
    right.set_ylabel("deviation in combined standard errors")
    right.set_ylim(-4.0, 4.0)
    right.set_title(f"mean $z = {table['z'].mean():+.3f}$, "
                    f"max $|z| = {table['z'].abs().max():.2f}$")
    right.grid(True, alpha=0.4)
    right.legend(fontsize=8)

    fig.savefig(FIGURES / "lsm_replication.png")
    plt.close(fig)


def figure_svi_fit():
    forward, discount, maturity = 5487.3, 0.988, 0.25
    strikes = np.arange(3800.0, 7600.0, 25.0)
    log_moneyness = np.log(strikes / forward)
    true_vol = 0.18 - 0.35 * log_moneyness + 0.6 * log_moneyness ** 2

    rng = np.random.default_rng(1)
    market_vol = true_vol + rng.normal(0.0, 0.0015, len(strikes))
    total_variance = market_vol ** 2 * maturity
    is_call = strikes >= forward
    vega = bs_greeks(forward, strikes, maturity, market_vol, discount, is_call)["vega"]
    weights = np.maximum(vega, 1e-8) ** 2

    params = fit_svi(log_moneyness, total_variance, weights)
    fitted_vol = np.sqrt(np.maximum(svi_total_variance(log_moneyness, params), 0.0)
                         / maturity)
    residual = fitted_vol - market_vol
    butterfly = check_butterfly(params, k_range=(log_moneyness.min(),
                                                 log_moneyness.max()))

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
    top.plot(log_moneyness, market_vol * 100, ".", color=MUTED, markersize=3,
             label="market implied vol")
    top.plot(log_moneyness, fitted_vol * 100, "-", color=ACCENT,
             label="SVI fit")
    top.axvline(0.0, color=MUTED, linewidth=0.6, linestyle=":")
    top.set_ylabel("implied volatility (%)")
    top.set_title(f"Benchmark 9 — SVI fit, RMSE "
                  f"{np.sqrt(np.mean(residual ** 2)) * 100:.3f} vol points, "
                  f"{butterfly['n_violations']} arbitrage violations")
    top.grid(True, alpha=0.4)
    top.legend()

    bottom.plot(log_moneyness, residual * 100, ".", color=ACCENT, markersize=3)
    bottom.axhline(0.0, color=MUTED, linewidth=0.6)
    bottom.set_xlabel(r"log-moneyness $k = \ln(K/F)$")
    bottom.set_ylabel("residual (vol pts)")
    bottom.grid(True, alpha=0.4)

    fig.savefig(FIGURES / "svi_fit.png")
    plt.close(fig)


def figure_vix_benchmark():
    path = Path("data/processed/vix_reconstruction.csv")
    if not path.exists():
        print("  skipped vix_benchmark — no reconstruction output yet")
        return

    table = pd.read_csv(path)
    if "published_vix" not in table.columns or table["published_vix"].isna().all():
        print(f"  skipped vix_benchmark — add a published_vix column to {path}")
        return

    table = table.dropna(subset=["vix", "published_vix"])
    error = table["vix"] - table["published_vix"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    left.plot(range(len(table)), table["published_vix"], "o-", color=MUTED,
              markersize=4, label="CBOE published close")
    left.plot(range(len(table)), table["vix"], "s--", color=ACCENT, markersize=4,
              label="reconstructed")
    left.set_xlabel("snapshot")
    left.set_ylabel("VIX")
    left.set_title("Benchmark 8 — VIX reconstruction")
    left.grid(True, alpha=0.4)
    left.legend(fontsize=8)

    right.axhspan(-0.20, 0.20, color="#e6e6e6", alpha=0.6, label="target band")
    right.axhline(0.0, color=MUTED, linewidth=0.6)
    right.plot(range(len(table)), error, "o", color=ACCENT, markersize=4)
    right.set_xlabel("snapshot")
    right.set_ylabel("reconstructed minus published (vol pts)")
    right.set_title(f"mean {error.mean():+.3f}, "
                    f"mean abs {error.abs().mean():.3f}, n = {len(table)}")
    right.grid(True, alpha=0.4)
    right.legend(fontsize=8)

    fig.savefig(FIGURES / "vix_benchmark.png")
    plt.close(fig)


def main():
    apply_style()
    FIGURES.mkdir(parents=True, exist_ok=True)

    builders = [("mc_convergence", figure_mc_convergence),
                ("hedging_error", figure_hedging_error),
                ("vol_mismatch", figure_vol_mismatch),
                ("lsm_replication", figure_lsm_replication),
                ("svi_fit", figure_svi_fit),
                ("vix_benchmark", figure_vix_benchmark)]

    for name, builder in builders:
        print(f"building {name} ...")
        builder()

    print("\nfigures in", FIGURES)
    for existing in sorted(FIGURES.glob("*.png")):
        print(f"  {existing.name}  ({existing.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()