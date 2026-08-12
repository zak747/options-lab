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

The intended source was Cboe's delayed quotes endpoint, which serves the full
SPX chain including weeklys as JSON. Three snapshots were collected manually
during development. Automated collection was not pursued: Cboe's delayed
quotes pages prohibit extraction by automated software and state that
offending IP addresses will be blocked (DD30). A scheduled job against the
JSON endpoint behind that table falls within the prohibition regardless of
its technical route.

This is a binding constraint on benchmark 8 rather than on the
implementation. Every component of the VIX reconstruction is validated
against synthetic chains with known parameters; what is missing is the
external comparison, which requires a licensed source — OptionMetrics via
an institutional WRDS subscription, or a keyed commercial API.

### 2.2 Cleaning and filters

`clean_chain` rejects quotes with no bid, crossed markets, bids below a
minimum tick, relative spreads beyond a threshold, and expired rows. Every
rejection is labelled in a `rejected_reason` column before filtering, so the
count and composition of what was discarded is recoverable. On a synthetic
chain with realistic tick rounding and dead wings, 354 of 408 rows survive.

### 2.3 Forward and discount factor extraction

Put-call parity holds exactly at every strike:

    C(K) - P(K) = DF*F - DF*K

so a regression of the parity difference on strike has slope -DF and
intercept DF*F. Neither the interest rate nor the dividend yield is assumed;
both are implied by the quotes. This is what makes the forward
parameterisation of the pricer (DD1) usable on real data, and it is the
reason `bs_price` takes F and DF rather than S, r and q.

The fit is weighted by inverse squared quoted spread and run twice: a first
pass over all paired strikes to locate the forward, then a second restricted
to strikes within 10% of it, since deep wing quotes have one nearly
worthless leg (DD18). Measured sensitivity to that band is below 1e-4
relative across 5%, 10% and 20%.

On a synthetic chain with quotes rounded to a 0.05 tick and perturbed within
the spread, the forward is recovered to 8.5e-7 relative and the discount
factor to 1.9e-4. The two-order-of-magnitude gap is structural rather than a
defect: the forward is the regression intercept, pinned by the level of the
parity difference, while the discount factor is the slope, estimated from how
that difference varies over a strike range narrow relative to its absolute
level (DD19). It matters for the VIX strip, where the discount factor
multiplies every option price.

## 3. Pricing engine

### 3.1 Black-Scholes and analytic Greeks

Implemented in Black-76 form. The Greeks reduce cleanly because of the
identity F*phi(d1) = K*phi(d2), which cancels the derivative-of-d terms and
leaves delta as DF*N(d1) with no correction.

Two conventions are fixed and stated, because "the derivative with respect to
time" is ambiguous when both F and DF evolve: theta differentiates T with F
and DF held fixed, and rho differentiates r inside DF with F held fixed
(DD6). A consequence is that theta is identical for a call and the put of the
same strike, which is not true in the spot parameterisation and is asserted
as a test.

Both pre-registered tolerances in this section required amendment. Put-call
parity was registered at an absolute 1e-14, which is unachievable in double
precision at index-level notionals where one unit in the last place is
already ~1e-12; the tolerance is now relative to notional (DD4), and the
measured error is 1.85e-16. Gamma was to be verified by second-differencing
the price, whose best achievable relative accuracy is ~1e-6 — the tolerance
itself; it is now verified by differencing the analytic delta, which is
independently checked against the price, so the chain price -> delta -> gamma
is validated link by link at first-difference precision (DD7). Measured worst
relative error across all five Greeks is 1.33e-8.

### 3.2 Implied volatility solver

The root find is well-posed — V is strictly increasing in sigma, so the
solution is unique and bracketable — and the difficulty is entirely
numerical. Vega decays as exp(-d1^2/2), reaching ~1e-22 at d1 = -10, so a raw
Newton step of -g/vega throws the iterate to sigma ~ 1e95. The solver
maintains a bracket and takes a bisection step whenever the Newton proposal
leaves it, so every iteration either accepts a Newton step or halves the
bracket and convergence is unconditional.

Convergence is tested on price rather than on sigma: the quote is known only
to the tick, and in the wings a 1e-12 change in sigma moves the price by less
than machine epsilon, so a sigma criterion may be unsatisfiable.

Benchmark 3 was registered at a flat 1e-12 round-trip and amended (DD11). The
inversion has a precision floor of roughly eps * V / vega, because the price
is representable only to eps*V and the sigma error is the price error divided
by the local slope. Where vega is small the problem is ill-conditioned by
construction. Measured: 1.46e-12 where vega exceeds 1e-4 of notional, against
2.19e-4 unrestricted. 46 of 224 grid points have true prices below 1e-100 —
one is 5e-134 — and return NaN rather than a confident wrong answer (DD10).

Throughput is 273,000 solves per second single-threaded, at a mean of 3.9
iterations on a realistic chain.

## 4. Monte Carlo

### 4.1 Convergence

Fitting log(RMSE) against log(N) over N in [2^10, 2^19] with 32 repetitions
gives -0.5054 for pseudorandom sampling, against the CLT's -1/2. The
exponent is a stronger check than a price comparison: it is distorted by a
biased estimator or by correlated seeding across repetitions, neither of
which produces an obviously wrong number at any single N.

Scrambled Sobol gives -0.9815. Scrambling matters — plain Sobol is
deterministic and has no standard error at all, so independent randomisations
are what make the RMSE estimable.

### 4.2 Variance reduction

The optimal control variate coefficient b* = Cov(Y,X)/Var(X) implies a
variance reduction factor of exactly 1/(1-rho^2). Testing that identity
rather than the magnitude is deliberate: a wrong b* still reduces variance,
just not optimally, and would pass a "did it improve" check.

    control          rho       VRF achieved    predicted
    terminal S_T     0.8900    4.81            4.81
    delta P&L        0.9973    182.63          182.63

The delta control is the discrete delta-hedge P&L, whose expectation is zero
because the forward is a martingale and the delta at each node is measurable
at that node. It removes the component of the payoff explained by first-order
moves, leaving only gamma P&L — the same quantity that becomes the object of
study in section 8.

Antithetic sampling was implemented and found to fail on a payoff it is
routinely assumed to help. At equal payoff evaluations,

    VRF = 1 / (1 + rho_anti),   rho_anti = corr(Y(Z), Y(-Z))

so the method helps only when the payoff is monotone in Z. For an ATM call
rho_anti = -0.367 and the factor is 1.578; for an ATM straddle rho_anti =
+0.900 and the factor is 0.526, nearly doubling the variance (DD14).

Establishing this required correcting the standard error: antithetic draws
are deliberately dependent, so the iid formula is invalid and reported a
factor of 1.00 — no benefit — for the call while concealing the straddle
failure entirely at 0.997. Averaging within each pair before taking the
standard error restores a valid iid sample of n/2 observations (DD13).

## 5. American options: Longstaff-Schwartz replication

Twenty cells of Table 1 reproduced within their published standard errors,
with 200,000 paths, 50 exercise dates per year and a degree-3 basis. All 20
pass at three combined standard errors; the largest deviation is 1.10
combined s.e. and the mean is -0.02.

The mean is the informative statistic. A near-zero mean indicates no
systematic bias against the published values, which is what would be expected
if both implementations estimate the same quantity; a large mean would point
to a difference in the exercise policy rather than to sampling noise.

Two implementation points were established by test rather than assumed.

The backward induction values exercise at t_1 through t_N only, so what it
computes is the continuation value from t_0 rather than the option price. At
S = 30, K = 40 it returns 9.952 against an intrinsic value of 10.000. Taking
the maximum against the t_0 payoff converts it into a price (DD15). This does
not affect Table 1, where continuation always dominates, but it matters
wherever immediate exercise is optimal.

The basis degree was selected by measured convergence rather than by
interpretation of the paper's description. The LSM estimate is a lower bound
in expectation, since the exercise policy is estimated from a finite basis
and is therefore suboptimal, so a richer basis raises the estimate
monotonically. At S = 36, sigma = 0.40, T = 2: 8.472, 8.496, 8.496, 8.504 for
degrees 2 to 5, with a standard error of 0.023. Degree 3 is within noise of
degrees 4 and 5 (DD16).

## 6. Volatility surface

SVI fitted per expiry in total variance against log-moneyness. The raw
parameterisation is non-convex in (m, s) and a five-parameter search lands in
local minima on ordinary data. Substituting y = (k-m)/s makes the model
linear in (a, b*s*rho, b*s), so the inner problem is ordinary least squares
and only the two shape parameters require an optimiser (DD20). Verified by
exact recovery of known parameters from noiseless SVI data to 6.7e-14.

Weights are (vega / (2*sigma*T*spread))^2, which expresses a total variance
residual as a price error in units of quote uncertainty. Unweighted fitting
lets illiquid wing quotes, whose spreads can exceed their mids, pull the
surface as hard as tight near-money quotes (DD21).

Measured fit RMSE is 0.00027 volatility points across 149 quotes, with zero
butterfly and zero calendar violations.

A separate scan for arbitrage in the raw quotes produced the most
instructive result in the project. Evaluated at mid, 44 butterfly violations
appear. Evaluated at executable prices — asks on the wings bought, bid on the
body sold — zero survive. All 44 lie deep in the money, where the option is
almost entirely intrinsic value and the true convexity across a 25-point
strike gap is smaller than a tick of quote noise (DD22).

The scan also weights the wings by the opposite strike gaps. A 1-2-1
butterfly has a non-negative payoff only when strikes are equally spaced, and
SPX grids are not, so an unweighted scan reports violations that are
artefacts of the grid.

## 7. VIX reconstruction

### 7.1 Methodology

Reproduced as specified in Cboe's white paper rather than improved on:
expiries selected to bracket 30 days within a 23-to-37-day window, the
forward taken from the single strike minimising |C - P|, K0 the highest
strike at or below it, an OTM-only strip truncated at the second consecutive
zero bid, and interpolation to exactly 30 days in minutes.

The single-strike forward deliberately differs from the regression estimator
used in section 2.3 (DD2). The regression is the better estimator; the
published methodology specifies the other, and this is a replication.

Maturity is measured in minutes to the actual settlement instant — 09:30 ET
for AM-settled standard monthlies, 16:00 ET for PM-settled weeklys (DD23).
The 6.5-hour difference is 1.2% of T on a 23-day option, and the distinction
cannot be recovered from a chain that does not record the option root.

### 7.2 Validation

The comparison against a published close cannot separate an implementation
error from a data timing difference, so the internal check uses a flat
volatility surface, where the variance swap rate equals the volatility
exactly:

    input 10.00% -> 10.0094      input 20.00% -> 20.0043
    input 35.00% -> 35.0019

### 7.3 Error decomposition

Three sources are quantified, and together they set the floor on what is
achievable against a published value regardless of data quality.

Strike discretisation, since the replication is exact only in the continuum
limit: -0.0009, +0.0043, +0.0201 and +0.0849 volatility points at strike
steps of 5, 25, 50 and 100. SPX lists 5-point strikes near the money and 25
or wider in the wings.

Wing truncation, which is signed: every term in the strip is non-negative, so
an early cut biases variance down, measured at -0.005 when truncated at 1.16
times spot.

The interest rate: assuming zero when the true rate is 4% costs -0.030
volatility points, a sixth of the benchmark tolerance before any data issue
(DD26). The chain supplies a discount factor per expiry through put-call
parity, so no external curve is needed.

### 7.4 Status

Benchmark 8 is not met, and is not attempted with the three snapshots
available. Three observations taken minutes apart during a single session
would not constitute a comparison against a published close, and reporting
them as one would misrepresent the evidence. The reconstruction is complete
and internally validated; the external comparison awaits a licensed chain
source.

## 8. Discrete hedging experiment

Shorting a call and delta hedging n times over its life, the standard
deviation of the terminal P&L scales as n^(-1/2) (Boyle & Emanuel, 1980).
Over n in [10, 640] the fitted exponent is -0.4869 with a standard error of
0.0010 — inside the pre-registered band but thirteen standard errors from
-0.5. The result is asymptotic in n, and restricting the fit to larger n
moves the estimate monotonically toward the limit: -0.4879 from n = 10,
-0.4895 from n = 40, -0.4946 from n = 160 (DD28).

The hedge ratio is spot delta, dV/dS = N(d1), not the forward delta the
pricer returns. Using the latter would understate every position by the
discount factor — about 2% at r = 2% over a year — producing no exception and
no obviously wrong number, only a small systematic bias (DD27).

The second experiment is the economically substantive one. Selling a one-year
ATM call at 20% and varying the volatility that realises:

    realised   hedged at realised        hedged at priced
    vol        mean      std             mean      std
    0.10       +3.979    0.155           +3.981    1.005
    0.20       +0.001    0.312           +0.001    0.312
    0.30       -3.983    0.467           -3.980    1.869

Hedging at the volatility that realises makes the P&L deterministic up to
discretisation, equal to [V(sigma_priced) - V(sigma_realised)]*exp(rT), and
the simulated means agree with that closed form to within a standard error at
every point. Hedging at the priced volatility delivers the same expectation
as accumulated gamma P&L against realised moves, with a standard deviation up
to 6.5 times larger (DD29).

The choice of hedging volatility therefore determines risk, not expected P&L.
Since the realised volatility is not observable in advance, the second
convention is the one that describes an actual book, and the two
non-overlapping P&L distributions in `figures/vol_mismatch.png` are what a
short volatility position looks like.

## 9. Benchmark results

Ten benchmarks pre-registered, nine met, one outstanding for want of a
licensed data source. Full table in `BENCHMARKS.md`.

Two tolerances proved unachievable and were amended with stated reasons
rather than quietly dropped: put-call parity, infeasible in double precision
at index notionals (DD4), and the implied volatility round-trip, bounded
below by the conditioning of the inversion (DD11). In both cases the
amendment, the measurement that prompted it and the original wording are
recorded.

## 10. Limitations

The VIX reconstruction has no external validation. It is verified against a
flat surface, where the answer is known analytically, and its error sources
are quantified; it has not been compared to a value published by Cboe.

The Longstaff-Schwartz price is a lower bound in expectation. The exercise
policy is estimated by regression on a finite basis and is therefore
suboptimal, and the measured monotone increase in price with basis degree is
a direct observation of that bias.

Everything outside section 5 is tested against synthetic chains with known
parameters. Synthetic data reproduces tick rounding, bid-ask spreads, dead
wings and a realistic skew, but not stale prints, locked markets, corporate
actions, or the microstructure of a fast market. The cleaning rules are
therefore validated against the failure modes anticipated, not against those
encountered.

No claim of profit is made anywhere. The hedging results are controlled
simulations on synthetic paths with a known data-generating process, not a
backtest, and they carry no transaction costs, no market impact and no
discrete dividends.

Rates enter only through the discount factor implied by each expiry's own
quotes. There is no term structure model, no stochastic rates and no
calibration to instruments outside the chain.