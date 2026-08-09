"""CBOE VIX reconstruction from a raw SPX chain.

The headline benchmark. VIX is not a model: it is the fair strike of a
30-day variance swap, replicated by a weighted strip of OTM option prices,

    sigma^2 = (2/T) * sum_i [ (dK_i / K_i^2) * exp(r*T) * Q(K_i) ]
              - (1/T) * (F/K_0 - 1)^2

Procedure, per the CBOE white paper:

  1. Select two expiries bracketing 30 days, inside the 23-37 day window,
     using both standard and weekly SPX options.
  2. Forward from the strike with the smallest |C - P|:
     F = K + exp(r*T) * (C - P).   NOTE: this is deliberately a different
     forward estimator from chain.implied_forward. See DEVIATIONS.md, DD2.
  3. K_0 = highest strike at or below F.
  4. Build the strip: puts below K_0, calls above, average of both at K_0.
     Then the truncation rule — walk outward from K_0 and stop at the
     SECOND consecutive zero bid, discarding everything beyond.
  5. dK_i = half the distance between neighbouring strikes; one-sided at
     the endpoints of the strip.
  6. Compute sigma^2 for each expiry, interpolate to exactly 30 days,
     take the square root, multiply by 100.

Expected error sources, roughly in order of magnitude:
  - timing gap between the snapshot and CBOE's calculation window
  - stale quotes in the wings
  - sensitivity of the truncation rule to exactly which bids are zero
  - the interest rate input
Each is to be quantified in DEVIATIONS.md, not just listed.
"""

import numpy as np

__all__ = [
    "select_expiries",
    "forward_from_parity",
    "strike_strip",
    "expiry_variance",
    "compute_vix",
    "benchmark_vs_published",
]


def select_expiries(chain, target_days=30, min_days=23, max_days=37):
    """Return the near-term and next-term expiries."""
    raise NotImplementedError


def forward_from_parity(df_expiry, r, T):
    """Single-strike forward, per the CBOE specification.

    Returns (F, K_0, K_star) where K_star is the strike used.
    """
    raise NotImplementedError


def strike_strip(df_expiry, K_0):
    """Build the OTM strip and apply the two-consecutive-zero-bid rule.

    Returns a frame with columns: strike, price, dK. Also return the
    number of strikes dropped at each wing — that count is the main
    lever on the final number and belongs in the error decomposition.
    """
    raise NotImplementedError


def expiry_variance(strip, F, K_0, r, T):
    """Sigma^2 for a single expiry, from the formula above."""
    raise NotImplementedError


def compute_vix(chain, r_curve, snapshot_ts):
    """Full pipeline for one snapshot. Returns the VIX level.

    Use minutes to expiry, not days: CBOE's specification is
    minute-precise and the difference is not negligible at 23 days.
    """
    raise NotImplementedError


def benchmark_vs_published(computed, published):
    """Benchmark 8.

    Returns a DataFrame: date, computed, published, abs_error, plus
    summary statistics (mean absolute error, max absolute error,
    correlation). Target is mean absolute error below 0.20 vol points.
    """
    raise NotImplementedError
