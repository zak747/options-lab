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

## DD4 — (next entry)
