# Replication memo

*Stub. Written at the end of the project, not the beginning.*

## 1. Summary

An options pricing library built from first principles, with every component
measured against a number published by someone else and every tolerance
recorded before the code was written.

Results as of the v1.0 tag:

| Benchmark | Target | Result |
|-----------|--------|--------|
| Longstaff & Schwartz (2001) Table 1 | within published s.e. | 20/20 cells, max 1.10 combined s.e., mean z = -0.02 |
| Monte Carlo convergence exponent | -0.50 ± 0.02 | -0.5054 |
| Discrete hedging error exponent | -0.50 ± 0.05 | -0.4869 ± 0.0010 |
| Forward from put-call parity | < 1e-4 relative | 8.5e-7 |
| SVI static arbitrage violations | 0 | 0 |
| VIX vs CBOE published close | < 0.20 vol points | pending data collection |

Three findings are worth more than the pass marks.

**Antithetic sampling can increase variance.** At equal payoff evaluations the
variance reduction factor is 1/(1 + rho), where rho is the correlation between
paired payoffs. For an ATM call rho = -0.37 and the factor is 1.58. For an ATM
straddle rho = +0.90 and the factor is 0.53 — the technique nearly doubles the
variance. Reporting only the favourable case would misrepresent the method.

**Apparent arbitrage does not survive the bid-ask spread.** Scanning a chain
for butterfly violations at mid quotes returns 44 candidates. Pricing the same
butterflies at executable levels — asks on the wings, bid on the body — returns
zero. All 44 lie deep in the money, where the option is almost entirely
intrinsic value and the true convexity across a 25-point strike gap is smaller
than a tick of quote noise. What looks like mispricing is quote noise.

**The hedging volatility determines risk, not expected P&L.** Selling a
one-year ATM call at 20% and hedging at the volatility that subsequently
realises produces a P&L known in advance, equal to
[V(sigma_priced) - V(sigma_realised)] * exp(rT), with no path dependence.
Hedging at the priced volatility instead delivers the same expectation as
accumulated gamma P&L, with a standard deviation up to 6.5 times larger. Since
the realised volatility is not observable in advance, the second is the one
that describes a real book.

Two pre-registered tolerances proved unachievable and were amended with reasons
rather than quietly dropped: the put-call parity bound, which was infeasible in
double precision at index-level notionals (DD4), and the implied volatility
round-trip target, which is bounded below by the conditioning of the inversion
wherever vega is small (DD11).

## 2. Data

### 2.1 Source and collection
### 2.2 Cleaning and filters
### 2.3 Forward and discount factor extraction

## 3. Pricing engine

### 3.1 Black-Scholes and analytic Greeks
### 3.2 Implied volatility solver

## 4. Monte Carlo

### 4.1 Convergence
### 4.2 Variance reduction

## 5. American options: Longstaff-Schwartz replication

## 6. Volatility surface

## 7. VIX reconstruction

### 7.1 Methodology
### 7.2 Results against CBOE published closes
### 7.3 Error decomposition

## 8. Discrete hedging experiment

## 9. Benchmark results

*Cross-reference to `BENCHMARKS.md`.*

## 10. Limitations
