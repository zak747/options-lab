# options-lab

Options pricing, implied volatility surfaces and VIX reconstruction,
implemented from first principles in Python.

**Status:** in development.

## Headline results

| Result | Target | Achieved |
|--------|--------|----------|
| VIX reconstructed from raw SPX chains vs CBOE published close | < 0.20 vol points | pending data collection |
| American put table, Longstaff & Schwartz (2001) | within published s.e. | 20/20 cells within published s.e. |
| Black-Scholes against Hull's worked example | to the book's 2 d.p. | 4.7594 / 0.8086 vs 4.76 / 0.81 |
| Monte Carlo convergence exponent | -0.50 ± 0.02 | -0.5054 |
| Discrete hedging error exponent | -0.50 ± 0.05 | -0.4869 ± 0.0010 |
| SVI static arbitrage violations | 0 | 0 |
| Forward extracted from put-call parity | < 1e-4 relative | 8.5e-7 |

Full table, including the tolerances fixed in advance and the two that were
amended with reasons, in [`BENCHMARKS.md`](BENCHMARKS.md).

Selected supporting measurements: the implied volatility solver runs at
273,000 solves/sec with a mean of 3.9 iterations; antithetic sampling is
measured *increasing* variance by a factor of 1.9 on a straddle, which is
the condition under which the technique fails; and 44 butterfly arbitrage
violations found in mid quotes fall to zero when priced at executable
bid-ask levels.

## What this is

A pricing library plus a set of experiments that check it against external
ground truth. The point is not that Black-Scholes has been implemented —
it is that every component is measured against a number published by
someone else, with the tolerance written down before the code was.

- `bs.py` — Black-Scholes in forward / discount-factor form, analytic Greeks
- `iv.py` — safeguarded Newton implied volatility solver
- `mc.py` — Monte Carlo with antithetic, control variates and Sobol QMC
- `lsm.py` — Longstaff-Schwartz least-squares Monte Carlo for American options
- `chain.py` — option chain ingestion, cleaning, forward extraction by put-call parity
- `surface.py` — SVI volatility surface fitting with no-arbitrage diagnostics
- `vix.py` — CBOE VIX reconstruction from a raw SPX chain
- `hedge.py` — discrete delta-hedging error scaling experiment

No options library is used anywhere. `scipy` provides the normal
distribution, an optimiser and a Sobol sequence; everything else is
written here.

## What this is not

- Not a production pricing system. Single-threaded Python, no calibration
  to exotics, no term structure of rates beyond what the chain implies.
- Not a claim that the VIX methodology is improved on. It is reproduced
  exactly as specified in order to test the implementation.
- Not a trading strategy. No P&L is claimed anywhere except in the
  simulated hedging experiment, which is a controlled study on synthetic
  paths, not a backtest.
- The Longstaff-Schwartz price is a lower bound in expectation, because
  the exercise policy estimated by regression is suboptimal.

## Installation

```bash
git clone https://github.com/zak747/options-lab.git
cd options-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Reproducing the results

```bash
python scripts/snapshot.py                  # collect one SPX chain snapshot
python scripts/run_lsm_replication.py       # benchmark 6
python scripts/run_vix_benchmark.py         # benchmark 8
python scripts/make_figures.py              # all figures
```

`scripts/snapshot.py` must be run daily; option chain history cannot be
back-filled. See `DEVIATIONS.md` (DD3) on what is and is not committed.

## Documents

- [`BENCHMARKS.md`](BENCHMARKS.md) — pre-registered targets and results
- [`DEVIATIONS.md`](DEVIATIONS.md) — every methodological decision, numbered
- [`MEMO.md`](MEMO.md) — full write-up

## Licence

MIT.
