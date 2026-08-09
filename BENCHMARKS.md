# Benchmarks

This file is a **pre-registration**. Every target below was written down
before the corresponding implementation existed. Tolerances come from
published standard errors, from theory, or from machine precision — none
of them are self-assessed after the fact.

A benchmark is either met or it is not. Where one is missed, the reason is
recorded here and in `DEVIATIONS.md` rather than the tolerance being widened.

| #  | Benchmark | Source of target | Target | Achieved | Error | Produced by |
|----|-----------|------------------|--------|----------|-------|-------------|
| 1  | Put–call parity `C - P = DF*(F - K)` | Algebraic identity | abs err < 1e-14 | — | — | `tests/test_bs.py` |
| 2  | Analytic Greeks vs central finite difference | Numerical differentiation | rel err < 1e-6 | — | — | `tests/test_bs.py` |
| 3  | Implied vol round-trip recovery | Machine precision | abs err < 1e-12 | — | — | `tests/test_iv.py` |
| 4  | Monte Carlo convergence exponent | CLT: RMSE ~ N^(-1/2) | -0.50 ± 0.02 | — | — | `scripts/run_mc_study.py` |
| 5  | Control variate VRF vs 1/(1-rho^2) | Theory | agreement within 5% | — | — | `scripts/run_mc_study.py` |
| 6  | American put price table | Longstaff & Schwartz (2001), *RFS* 14(1) | within published s.e. | — | — | `scripts/run_lsm_replication.py` |
| 7  | Discrete hedging error exponent | Boyle & Emanuel (1980) | -0.50 ± 0.05 | — | — | `scripts/run_hedge_study.py` |
| 8  | **VIX reconstruction vs CBOE published close** | CBOE VIX white paper | abs err < 0.20 vol pts | — | — | `scripts/run_vix_benchmark.py` |
| 9  | SVI static arbitrage violations | Gatheral & Jacquier (2014) | 0 (butterfly and calendar) | — | — | `scripts/run_surface_fit.py` |

Benchmark 8 is the headline result. Benchmark 6 is the formal replication.

## Notes on tolerances

- **1, 3** are identities, so the only limit is floating point.
- **2** uses a central difference, whose truncation error is O(h^2); 1e-6
  relative is achievable with a sensibly chosen bump size and is not
  tight enough to be sensitive to it.
- **4, 7** are exponents fitted by OLS on log-log data; the stated bands
  are wide enough to absorb finite-sample noise at the path counts used.
- **6** uses the standard errors reported in the source table, not a
  tolerance chosen here.
- **8** is judged against the number CBOE publishes. The dominant error
  source is expected to be the timing gap between the quote snapshot and
  CBOE's own calculation window; this is quantified in `DEVIATIONS.md`.
