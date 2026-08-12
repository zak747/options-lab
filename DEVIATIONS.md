# Design decisions and deviations

Every methodological choice that a reader could reasonably have made
differently is recorded here, with the alternative that was rejected and
the reason. Entries are numbered and never renumbered.

---

## DD1 — Pricer parameterised by forward and discount factor, not spot/r/q

**Decision.** `bs.py` prices in the Black-76 form,

    price = DF * [ F*N(d1) - K*N(d2) ]   (call)

taking the forward `F` and discount factor `DF` as inputs, rather than
taking spot `S`, risk-free rate `r` and dividend yield `q` and computing
`F = S*exp((r-q)T)` internally.

**Alternative rejected.** The textbook spot parameterisation.

**Reason.** On a real option chain neither `r` nor `q` is observable. Both
are inferred from the option prices themselves via put–call parity (see
`chain.implied_forward`). A pricer written in terms of `S`, `r` and `q`
forces an assumed dividend yield, which biases every implied volatility
computed from it. Taking `F` and `DF` directly means the pricer consumes
exactly what the market reveals.

---

## DD2 — Two different forward definitions, by design

**Decision.** `chain.implied_forward` estimates the forward by regressing
`C - P` on `K` across all strikes of an expiry (slope gives `-DF`,
intercept gives `DF*F`). `vix.forward_from_parity` instead uses the single
strike at which `|C - P|` is smallest.

**Alternative rejected.** Using one method throughout.

**Reason.** The regression uses the whole cross-section and is more robust
to noise in any individual quote, so it is the better estimator for surface
fitting. But the VIX benchmark is a replication of a published
specification: CBOE mandates the single-strike method, and following it is
the point of the exercise. Deviating would make benchmark 8 meaningless.
The gap between the two forwards is reported as a diagnostic.

---

## DD3 — Derived strips committed, raw chain snapshots not

**Decision.** `data/raw/` is gitignored. The cleaned, OTM-only strike strip
actually used in each day's VIX computation is written to
`data/processed/` and committed.

**Alternative rejected.** Committing raw chain snapshots, so that the
pipeline is reproducible end to end from source data.

**Reason.** Redistributing a vendor's quote feed in a public repository may
not be permitted by its terms of use, whereas the derived strip is a small
transformed extract sufficient for a third party to re-run and verify the
headline number. Reproducibility is preserved from the strip onwards; the
download step is reproducible by running `scripts/snapshot.py`.

---

## DD4 — Put–call parity tolerance is relative to notional, not absolute

**Decision.** Benchmark 1 asserts

    |C - P - DF*(F - K)|  <=  1e-14 * DF * max(F, K)

rather than an absolute bound of 1e-14.

**Alternative rejected.** The absolute tolerance originally pre-registered
in `BENCHMARKS.md`.

**Reason.** A double carries roughly 16 significant decimal digits, so at
an index-level forward of F = 5000 a single unit in the last place is
already of order 1e-12. An absolute 1e-14 bound is therefore unachievable
at SPX notionals for reasons that have nothing to do with the
implementation. Scaling by the notional makes the tolerance test what it
was intended to test — the algebra — at every scale. `BENCHMARKS.md` has
been amended and the original wording is recorded here.

---

## DD5 — Degeneracy threshold on total volatility

**Decision.** `bs._prepare` classifies an entry as degenerate when

    v = sigma * sqrt(T) < 1e-12

and both `bs_price` and `bs_greeks` overwrite those entries with their
limiting values: the discounted intrinsic value on the forward, delta of
`omega * DF` if in the money and zero otherwise, and gamma, vega and theta
of zero. At F == K with v == 0, gamma is returned as `inf`.

**Alternative rejected.** Branching separately on `T == 0` and
`sigma == 0`, or returning NaN throughout the degenerate region.

**Reason.** The price depends on the inputs only through the pair
(F/K, v), so `T = 0` and `sigma = 0` are the same limit and a single
threshold on v handles both without duplicated logic. The value 1e-12 sits
far below any volatility or maturity that appears in real data while
remaining well clear of the point where 1/v loses precision. Returning the
correct limit rather than NaN matters because expired and zero-vol rows
occur routinely in real chains and must not poison downstream aggregation.

Gamma at F == K, v == 0 is genuinely unbounded: the risk-neutral density
collapses to a point mass at the strike. `inf` is the honest answer and
propagates visibly, whereas returning a large finite number would hide the
singularity.

---

## DD6 — Theta and rho conventions

**Decision.**

  * `theta` is `-dV/dT`, holding F and DF fixed, expressed per year.
  * `rho` is `dV/dr` with `DF = exp(-r*T)` and F held fixed, giving `-T*V`.
  * `vega` is `dV/dsigma` per 1.00 of volatility, not per volatility point.
  * `delta` is `dV/dF`, a forward delta, not a spot delta.

**Alternative rejected.** The spot-parameterised definitions, in which
theta also picks up the drift of the forward and rho accounts for
`F = S*exp((r-q)T)` moving with r.

**Reason.** Under DD1 the pricer has no knowledge of S, r or q, so a spot
delta is not expressible without additional assumptions. Holding F and DF
fixed is the only convention internally consistent with that
parameterisation. A consequence worth noting is that theta is then
identical for a call and the put of the same strike, since it contains
only the density and no N(.) term — in spot terms the two differ by
rate-dependent terms. That equality is asserted as a unit test.

These conventions are also what the finite-difference test in
`tests/test_bs.py` bumps. Bumping T while letting DF move with it computes
a different quantity and fails the test correctly.

---

## DD7 — Gamma verified by differentiating analytic delta

**Decision.** Benchmark 2 checks gamma as a first central difference of
the analytic delta,

    gamma_FD = [ delta(F+h) - delta(F-h) ] / (2h),   h ~ eps^(1/3) * F

rather than as a second central difference of the price.

**Alternative rejected.** The second-difference stencil
`[V(F+h) - 2V(F) + V(F-h)] / h^2` with `h ~ eps^(1/4) * F`, which was the
original plan.

**Reason.** The second difference divides by h^2, so rounding error enters
as eps/h^2 rather than eps/h, and its best achievable relative accuracy is
around 1e-6 — the same order as the pre-registered tolerance. Measured
worst-case error on the test grid was 1.2e-6, i.e. a marginal failure
driven entirely by numerical conditioning rather than by any error in the
derivation.

Differentiating delta instead makes every check in the benchmark a first
difference, accurate to about 1e-9. This does not weaken the test: delta
is independently verified against a difference of the price, so the chain
price -> delta -> gamma is validated link by link, each link at full
first-difference precision. Loosening the tolerance would have been the
alternative, and pre-registered tolerances are not to be loosened to
accommodate a poorly conditioned estimator.

---

## DD8 — Deep-wing Greeks excluded from the relative-error metric

**Decision.** The achieved figure reported for benchmark 2 is the worst
relative error over points where the Greek exceeds 1e-6 of its maximum on
the grid. The unit test itself uses `assert_allclose` with `rtol = 1e-6`
and an absolute floor of `1e-8 * max|Greek|`.

**Alternative rejected.** Reporting an unrestricted relative error across
every grid point.

**Reason.** Deep out of the money at short maturity, vega and theta are of
order 1e-10 while the option price itself is of order 1e-12. The finite
difference of a quantity that small is dominated by floating-point
rounding, so the comparison measures the conditioning of the numerical
estimator and not the correctness of the analytic formula. Including those
points produces a headline error of 2.4e-2, which would misrepresent the
implementation as inaccurate when the analytic values there are exact and
the numerical reference is noise. The restriction is stated explicitly so
that the reported figure cannot be read as a tighter claim than it is.

---

## DD9 — Bracket endpoints for the implied volatility solver

**Decision.** `SIGMA_LO = 1e-6`, `SIGMA_HI = 5.0`, checked by sign at both
ends before iterating.

**Alternative rejected.** Relying on the asymptotic no-arbitrage bounds
alone to guarantee the root is bracketed.

**Reason.** 5.0 is 500% annualised, above anything observed; SPX peaked
near 0.8 in March 2020. The lower end sits above the v = 0 degeneracy
threshold of DD5 so the pricer never takes its degenerate branch during
iteration. Cost of a wide bracket is logarithmic in width — roughly ten
extra bisection steps for a hundredfold widening — so erring wide is cheap.

The sign check matters because `price_bounds` returns the limits as sigma
tends to 0 and infinity, while SIGMA_HI is finite. A quote at 99.99% of
df*F implies a volatility above 5 and would otherwise be iterated inside a
bracket that does not contain its root. Evaluating g at both endpoints
costs two extra pricer calls and converts an assumption into a checked
precondition.

---

## DD10 — Quotes with unresolvable time value return NaN

**Decision.** `implied_vol` returns NaN when the price is within `tol_abs`
of its lower bound, rather than returning sigma = 0.

**Alternative rejected.** Treating a quote at intrinsic as sigma = 0, which
is the correct mathematical limit.

**Reason.** Time value is the only part of the price that depends on sigma.
On the round-trip grid, 46 of 224 points have a true price below 1e-100 —
one is 5e-134 — because they are deep out of the money at short maturity.
At that magnitude the price is numerically indistinguishable from intrinsic
and no algorithm can recover sigma from it. Returning 0.0 would report a
confident wrong answer: those points have true volatilities up to 1.5.
NaN propagates visibly and is filtered downstream.

---

## DD11 — Convergence tolerance tracks price magnitude, and benchmark 3 is amended

**Decision.** Convergence is tested on `|g| <= max(1e-15 * |price|,
1e-16 * df * max(F, K))`, plus a stall check that stops when the iterate
ceases to move at double precision. Benchmark 3's target is amended from
1e-12 to **1e-10, restricted to quotes with vega above 1e-4 of notional**.

**Alternative rejected.** The flat 1e-12 round-trip target pre-registered
in BENCHMARKS.md, and a tolerance scaled to notional alone.

**Reason.** Inverting a price for volatility has a precision floor set by
the conditioning of the inversion,

    |delta sigma|  >~  eps * V / vega

since V is representable only to eps*V and the sigma error is the price
error divided by the local slope. Where vega is small the problem is
ill-conditioned by construction, not by any defect in the solver.

Measured on a 224-point grid: worst error 1.46e-12 where vega exceeds 1e-4
of notional (160 points), 3.48e-10 where time value exceeds 1e-10 of
notional (164 points), and 2.19e-4 over all resolvable points. Observed
errors track the bound above. A flat 1e-12 target across an arbitrary grid
was therefore not achievable, and the amended figure is stated with the
restriction attached rather than reporting only the subset on which the
original target happens to hold.

---

## DD12 — In-the-money quotes converted to out-of-the-money before solving

**Decision.** `implied_vol_chain` replaces every ITM quote with its OTM
counterpart via put-call parity and flips the right before solving.

**Alternative rejected.** Solving each quote as given.

**Reason.** With F = 5500 and K = 4000 the call is worth roughly 1500, of
which about 1480 is intrinsic and independent of sigma. A one-tick error is
0.07% of the price but perhaps 5% of the recovered volatility. The OTM put
at the same strike is worth about 20, entirely time value, and inverts
cleanly. Parity makes the two quotes carry identical information, so the
conversion is free and strictly better conditioned.

---

## DD13 — Antithetic standard error computed from pair means

**Decision.** With antithetic sampling the standard error is the sample
standard deviation of the n/2 pair means divided by sqrt(n/2), not the
standard deviation of all n payoffs divided by sqrt(n).

**Alternative rejected.** Treating all n payoffs as an iid sample.

**Reason.** Antithetic draws are deliberately dependent, so the iid formula
is not a valid estimator of the estimator's variance. Using it reported a
variance reduction factor of 1.00 for the call — no benefit — when the true
figure is 1.58, and it concealed the straddle failure entirely by reporting
0.997 instead of 0.526. Averaging within each pair first restores an iid
sample of n/2 observations.

---

## DD14 — Antithetic variates reported on a payoff where they fail

**Decision.** The variance reduction study reports antithetic results for a
straddle alongside a call, and the unit test asserts the straddle VRF is
below 1.

**Alternative rejected.** Reporting only the call, where the technique
works.

**Reason.** At equal payoff evaluations,

    VRF = 1 / (1 + rho_anti),   rho_anti = corr(Y(Z), Y(-Z))

so the technique helps only when the payoff is monotone in Z. Measured:
rho_anti = -0.367 for an ATM call, VRF 1.578; rho_anti = +0.900 for an ATM
straddle, VRF 0.526 — nearly doubling the variance. Reporting only the
favourable case would misrepresent antithetic sampling as unconditionally
useful. The failure is also the more informative result, since it
demonstrates the condition under which the method works rather than that
the function runs.

## DD15 — Price taken as the maximum of continuation value and t_0 intrinsic

**Decision.** `lsm_price` returns `max(mean discounted cashflow, payoff at
t_0)`.

**Alternative rejected.** Returning the regression result directly.

**Reason.** The backward induction values exercise opportunities at
t_1, ..., t_N only; t_0 is not an exercise date in the algorithm. What it
computes is therefore the continuation value from t_0, not the option
price. Measured at S = 30, K = 40, sigma = 0.20, T = 1: the induction
returns 9.952 against an intrinsic value of 10.000. This does not affect
LS Table 1, where S is between 36 and 44 and continuation always dominates,
but it matters wherever immediate exercise is optimal.

---

## DD16 — Default basis degree is 3, chosen by measured convergence

**Decision.** `degree = 3`, i.e. four basis functions.

**Alternative rejected.** Three basis functions, matching the paper's
description of using the first three Laguerre polynomials.

**Reason.** The LSM estimate is a lower bound in expectation, because the
exercise policy is estimated from a finite basis and is therefore
suboptimal. Adding basis functions improves the policy and raises the
estimate monotonically. Measured at S = 36, sigma = 0.40, T = 2:

    degree 2 -> 8.472    degree 4 -> 8.496
    degree 3 -> 8.496    degree 5 -> 8.504

with a standard error of 0.023. Degree 2 sits about one standard error
below the converged value; degree 3 is within noise of degrees 4 and 5.
The paper's own basis is ambiguous — "the first three Laguerre
polynomials" may or may not include the constant term — so the degree is
selected by observed convergence rather than by interpretation.

---

## DD17 — Spot normalised by strike before the regression

**Decision.** The design matrix is built from x = S/K, not from S.

**Alternative rejected.** Passing raw spot to the basis functions.

**Reason.** The weighted Laguerre polynomials carry a factor exp(-x/2). At
LS Table 1 scale, S ~ 40 gives exp(-20) ~ 2e-9, so every column of the
design matrix underflows toward zero and the least-squares problem becomes
numerically meaningless. Normalising by the strike puts x near 1 and keeps
the condition number below 1e3 for degrees up to 4. The regression is
invariant to this rescaling in exact arithmetic; it is not in floating
point.

## DD18 — Forward extracted by two-pass weighted regression across strikes

**Decision.** `implied_forward` regresses the parity difference C - P on
strike, weighted by 1/(call spread + put spread)^2. A first pass over all
paired strikes gives a rough forward; the fit is then repeated on strikes
within 10% of it.

**Alternative rejected.** CBOE's single-strike method, F = K + exp(rT)(C-P)
at the strike minimising |C - P|.

**Reason.** Put-call parity holds exactly at every strike, so a regression
uses all of them and averages away quote noise, while the single-strike
method inherits the noise of one pair. The second pass exists because deep
wing strikes have one nearly worthless leg whose parity difference is
dominated by tick rounding; it is applied after a first estimate because
"near the money" cannot be defined before the forward is known. Measured
sensitivity to the band is below 1e-4 relative across 5%, 10% and 20%.

Note that `vix.py` deliberately uses the single-strike method instead, since
the VIX benchmark requires following CBOE's published specification exactly
(DD2). The two estimators are expected to differ slightly and the
difference is quantified in Phase 6.

---

## DD19 — Discount factor is materially less precise than the forward

**Observation, not a choice.** On a synthetic chain with quotes rounded to
a 0.05 tick and perturbed within the spread, the forward is recovered to
8.5e-7 relative while the discount factor is recovered only to 1.9e-4 —
more than two orders of magnitude worse.

**Reason.** The forward enters through the regression intercept, which is
pinned by the level of the parity difference. The discount factor is the
slope, estimated from how that difference varies across strikes, and a
slope estimated over a strike range narrow relative to its absolute level
is intrinsically noisier. This is a property of the estimator, not a defect
in the fit; R-squared exceeds 0.9999 in both cases.

It matters for Phase 6, where the discount factor multiplies every option
price in the VIX strip. The resulting sensitivity is quantified there
rather than assumed negligible.

## DD20 — SVI fitted by quasi-explicit reduction, not five-parameter search

**Decision.** Substituting y = (k - m)/s makes the model linear in
(a, b*s*rho, b*s), so the inner problem is ordinary least squares and only
the two-dimensional shape parameters (m, s) require an optimiser. Six
starts, Nelder-Mead on (m, log s).

**Alternative rejected.** Direct minimisation over all five parameters.

**Reason.** The raw SVI objective is non-convex in (m, s) and a
five-parameter search lands in local minima on ordinary data. The
reduction removes three parameters from the search entirely and makes the
remaining problem two-dimensional and cheap enough to restart. Verified by
exact recovery of known parameters from noiseless SVI data to 6.7e-14; a
fit that cannot reproduce an exact SVI curve is finding a local minimum
rather than the solution.

The positivity constraint a + b*s*sqrt(1 - rho^2) >= 0 is applied inside
the linear subproblem by shifting the level, rather than checked after the
fit. Raw SVI permits a negative level parameter, and the unconstrained
solution can fit the quoted strikes well while implying negative total
variance outside them, which has no implied volatility.

---

## DD21 — Fit weighted by vega relative to bid-ask spread

**Decision.** Weights are (vega / (2 * sigma * T * spread))^2 in total
variance space.

**Alternative rejected.** Unweighted least squares on total variance.

**Reason.** A residual in total variance maps to a price residual through
dV = vega * dw / (2 * sigma * T). Dividing by the spread expresses that
price error in units of quote uncertainty, and squaring gives inverse
variance weighting. Unweighted fitting lets illiquid wing quotes, whose
spreads can exceed their mids, pull the surface as hard as tight
near-money quotes.

RMSE is reported in volatility points rather than variance, because a
variance residual cannot be compared to a bid-ask spread and a market
maker quotes in vol points.

---

## DD22 — Butterfly scan uses executable prices and gap-weighted strikes

**Decision.** `butterfly_violations_from_quotes` weights the wings by the
opposite strike gaps and, in executable mode, prices the legs bought at the
ask and the leg sold at the bid.

**Alternative rejected.** An unweighted 1-2-1 butterfly evaluated at mid
quotes.

**Reason.** The 1-2-1 butterfly has a non-negative payoff only when the
strikes are equally spaced. SPX strike grids are not — 25 points near the
money, 50 or 100 in the wings — so an unweighted scan reports violations
that are artefacts of the grid rather than of the prices.

Evaluating at mids is the more consequential error. On a synthetic chain
with realistic tick rounding, 44 butterfly violations appear at mid and
zero survive at executable prices. All 44 lie deep in the money, where the
option is almost entirely intrinsic value and the true convexity across a
25-point gap is smaller than a tick of quote noise. Reporting mid-quote
violations as mispricings would therefore be reporting quote noise. This
is recorded because the executable check is the substantive finding, not a
refinement of one.

## DD27 — Hedging uses spot delta, not the forward delta returned by bs_greeks

**Decision.** `hedge.spot_delta` computes dV/dS = N(d1) directly rather
than calling `bs_greeks` and using its delta.

**Alternative rejected.** Using `bs_greeks(...)["delta"]` as the hedge ratio.

**Reason.** Under the forward parameterisation (DD1, DD6) `bs_greeks`
returns dV/dF = DF*N(d1). The hedge trades the underlying, so the required
ratio is

    dV/dS = dV/dF * dF/dS = DF*N(d1) * exp(r*tau) = N(d1)

and the discount factor cancels. Using the forward delta would understate
every position by DF — about 2% at r = 2% with one year to expiry. The
error produces no exception and no obviously wrong number; it would appear
only as a small systematic P&L bias that is hard to attribute after the
fact. Verified against a central finite difference of the price with
respect to spot.

---

## DD28 — Measured hedging exponent is -0.487, not -0.500

**Observation, not a choice.** The fitted exponent over n in [10, 640] is
-0.4869 with a standard error of 0.0010, i.e. thirteen standard errors from
the theoretical -0.5.

**Reason.** Boyle & Emanuel's result is asymptotic in the rebalance count.
Restricting the fit to larger n moves the estimate monotonically toward the
limit:

    fit from n =  10 -> -0.4879
    fit from n =  40 -> -0.4895
    fit from n = 160 -> -0.4946

The discrepancy is a finite-n correction, not a defect in the simulation.
It is recorded because reporting -0.49 as though it were -0.50 would
misstate the precision, and because the direction and magnitude of the
correction are themselves evidence that the experiment behaves correctly.

---

## DD29 — Volatility mismatch reported under two hedging conventions

**Decision.** `vol_mismatch_study` computes the hedge delta either at the
volatility that realises or at the volatility the option was sold at, and
both are reported.

**Alternative rejected.** Reporting mean P&L under a single convention.

**Reason.** The two have the same expectation and very different risk, so
one number would obscure the result. Selling a one-year ATM call at 20%,
50,000 paths, 500 rebalances:

    realised   hedged at realised        hedged at priced
    vol        mean      std             mean      std
    0.10       +3.979    0.155           +3.981    1.005
    0.15       +1.995    0.234           +1.995    0.681
    0.20       +0.001    0.312           +0.001    0.312
    0.25       -1.992    0.390           -1.992    0.921
    0.30       -3.983    0.467           -3.980    1.869

Hedging at the realised volatility makes the P&L deterministic up to
discretisation, equal to [V(sigma_priced) - V(sigma_realised)] * exp(rT),
and the simulated means agree with that closed form to within a standard
error at every point. Hedging at the priced volatility delivers the same
expectation as accumulated gamma P&L against realised moves, with a
standard deviation up to 6.5 times larger. Since the realised volatility is
not known in advance, the second convention is the one that describes an
actual book.

