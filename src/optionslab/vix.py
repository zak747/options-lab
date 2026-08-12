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
import pandas as pd

__all__ = [
    "select_expiries",
    "forward_from_parity",
    "strike_strip",
    "expiry_variance",
    "reconstruct",
    "compute_vix",
    "benchmark_vs_published",
]

# CBOE works in minutes, not days: the specification is minute-precise and
# the difference is not negligible at 23 days.
MINUTES_PER_YEAR = 365.0 * 24.0 * 60.0   # N_365 = 525600
MINUTES_PER_30D = 30.0 * 24.0 * 60.0     # N_30  = 43200
MINUTES_PER_DAY = 24.0 * 60.0

# US-Eastern settlement times. SPX standard monthlies are AM-settled at the
# 09:30 ET open; SPXW weeklys are PM-settled at the 16:00 ET close. That
# 6.5-hour gap is exactly why snapshot.py now retains the option root.
_AM_SETTLE = (9, 30)
_PM_SETTLE = (16, 0)


# --------------------------------------------------------------------------
# time to expiry, minute-precise and root-aware
# --------------------------------------------------------------------------

def _eastern_offset(ts):
    """US-Eastern UTC offset in hours (-4 EDT / -5 EST) for a naive date.

    The US rule since 2007: DST runs from the second Sunday of March to the
    first Sunday of November. Settlement is at 09:30 or 16:00 local, both
    well clear of the 02:00 transition, so a day-level comparison is exact.
    """
    ts = pd.Timestamp(ts)
    year = ts.year
    march1 = pd.Timestamp(year, 3, 1)
    first_sun_mar = march1 + pd.Timedelta(days=(6 - march1.weekday()) % 7)
    dst_start = first_sun_mar + pd.Timedelta(days=7)      # second Sunday
    nov1 = pd.Timestamp(year, 11, 1)
    dst_end = nov1 + pd.Timedelta(days=(6 - nov1.weekday()) % 7)  # first Sunday
    day = pd.Timestamp(ts.year, ts.month, ts.day)
    return -4 if (dst_start <= day < dst_end) else -5


def _settlement_utc(expiry, root):
    """UTC timestamp at which an expiry settles, given its root."""
    ts = pd.Timestamp(expiry)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    is_am = isinstance(root, str) and root.upper() == "SPX"
    hh, mm = _AM_SETTLE if is_am else _PM_SETTLE
    local = pd.Timestamp(ts.year, ts.month, ts.day, hh, mm)
    offset = _eastern_offset(local)
    return (local - pd.Timedelta(hours=offset)).tz_localize("UTC")


def _term_table(chain, snapshot_utc):
    """One record per (expiry, root) group with its minute-precise horizon."""
    has_root = "root" in chain.columns and chain["root"].notna().any()
    terms = []
    if has_root:
        groups = chain.groupby(["expiry", "root"], dropna=False)
    else:
        groups = chain.groupby("expiry")
    for key, sub in groups:
        if has_root:
            expiry, root = key
        else:
            expiry, root = key, None
        settle = _settlement_utc(expiry, root)
        minutes = (settle - pd.Timestamp(snapshot_utc)).total_seconds() / 60.0
        terms.append({
            "expiry": expiry, "root": root, "settle_utc": settle,
            "minutes": minutes, "T": minutes / MINUTES_PER_YEAR,
            "days": minutes / MINUTES_PER_DAY, "sub": sub,
        })
    return terms


# --------------------------------------------------------------------------
# 1. expiry selection
# --------------------------------------------------------------------------

def select_expiries(chain, target_days=30, min_days=23, max_days=37):
    """Return the near-term and next-term expiries as (near, next) dicts.

    Each dict carries the expiry's minute-precise horizon (minutes, T, days),
    its root, and the sub-chain of option rows. The pair is the one that most
    tightly brackets `target_days` inside the [min_days, max_days] window.
    """
    snapshot_utc = pd.Timestamp(chain["snapshot_utc"].iloc[0])
    terms = _term_table(chain, snapshot_utc)
    window = [t for t in terms if min_days <= t["days"] <= max_days]
    if len(window) < 2:
        raise ValueError(
            f"need at least two expiries in [{min_days}, {max_days}] days; "
            f"found {len(window)} (of {len(terms)} total)"
        )
    window.sort(key=lambda t: t["minutes"])
    below = [t for t in window if t["days"] <= target_days]
    above = [t for t in window if t["days"] > target_days]
    if below and above:
        return below[-1], above[0]
    # target not bracketed — fall back to the two closest to it
    return window[0], window[1]


# --------------------------------------------------------------------------
# 2. forward and K_0
# --------------------------------------------------------------------------

def _by_strike(df_expiry):
    """Call/put mid and bid per strike, outer-joined and strike-sorted."""
    calls = df_expiry[df_expiry["is_call"]]
    puts = df_expiry[~df_expiry["is_call"]]
    c = calls.groupby("K").agg(call_mid=("mid", "mean"), call_bid=("bid", "mean"))
    p = puts.groupby("K").agg(put_mid=("mid", "mean"), put_bid=("bid", "mean"))
    return c.join(p, how="outer").sort_index()


def forward_from_parity(df_expiry, r, T):
    """Single-strike forward, per the CBOE specification.

    Uses the strike with the smallest |C - P|:  F = K + exp(r*T) * (C - P).
    Returns (F, K_0, K_star) where K_star is that strike and K_0 is the
    highest strike at or below F.
    """
    grid = _by_strike(df_expiry)
    both = grid.dropna(subset=["call_mid", "put_mid"])
    if both.empty:
        raise ValueError("no strike has both a call and a put quote")
    diff = (both["call_mid"] - both["put_mid"]).abs()
    k_star = float(diff.idxmin())
    call = float(both.loc[k_star, "call_mid"])
    put = float(both.loc[k_star, "put_mid"])
    forward = k_star + np.exp(r * T) * (call - put)

    strikes = grid.index.to_numpy(dtype=float)
    at_or_below = strikes[strikes <= forward]
    K_0 = float(at_or_below.max()) if at_or_below.size else float(strikes.min())
    return float(forward), K_0, k_star


# --------------------------------------------------------------------------
# 3. the OTM strip and the two-consecutive-zero-bid truncation
# --------------------------------------------------------------------------

def _walk_wing(grid, strikes_in_order, bid_col, mid_col):
    """Walk one wing outward, applying the two-zero-bid stopping rule.

    A strike with a non-positive or missing bid is excluded. The moment two
    such strikes occur back to back, stop and discard everything beyond.
    """
    kept = []
    consecutive_zero = 0
    for k in strikes_in_order:
        bid = grid.loc[k, bid_col]
        if not np.isfinite(bid) or bid <= 0.0:
            consecutive_zero += 1
            if consecutive_zero >= 2:
                break
            continue
        consecutive_zero = 0
        kept.append((k, float(grid.loc[k, mid_col])))
    return kept


def _delta_k(strikes):
    """dK_i = half the gap between neighbours; one-sided at the endpoints."""
    strikes = np.asarray(strikes, dtype=float)
    n = strikes.size
    dK = np.empty(n)
    if n == 1:
        dK[0] = np.nan
        return dK
    dK[1:-1] = (strikes[2:] - strikes[:-2]) / 2.0
    dK[0] = strikes[1] - strikes[0]
    dK[-1] = strikes[-1] - strikes[-2]
    return dK


def strike_strip(df_expiry, K_0):
    """Build the OTM strip and apply the two-consecutive-zero-bid rule.

    Puts are used below K_0, calls above, and the average of the two at K_0.
    Returns (strip, dropped) where strip has columns strike, price, dK and
    dropped counts how many strikes were discarded on each wing — that count
    is the main lever on the final number and belongs in the error budget.
    """
    grid = _by_strike(df_expiry)
    strikes = grid.index.to_numpy(dtype=float)

    below = sorted((k for k in strikes if k < K_0), reverse=True)
    above = sorted(k for k in strikes if k > K_0)
    put_wing = _walk_wing(grid, below, "put_bid", "put_mid")
    call_wing = _walk_wing(grid, above, "call_bid", "call_mid")

    row = grid.loc[K_0]
    k0_price = np.nanmean([row.get("call_mid", np.nan), row.get("put_mid", np.nan)])

    rows = sorted(put_wing + [(K_0, float(k0_price))] + call_wing, key=lambda kv: kv[0])
    strip = pd.DataFrame(rows, columns=["strike", "price"])
    strip["dK"] = _delta_k(strip["strike"].to_numpy(dtype=float))

    dropped = {"put_wing": len(below) - len(put_wing),
               "call_wing": len(above) - len(call_wing)}
    return strip, dropped


# --------------------------------------------------------------------------
# 4. per-expiry variance
# --------------------------------------------------------------------------

def expiry_variance(strip, F, K_0, r, T):
    """Sigma^2 for a single expiry, from the replication formula."""
    k = strip["strike"].to_numpy(dtype=float)
    q = strip["price"].to_numpy(dtype=float)
    dK = strip["dK"].to_numpy(dtype=float)
    contributions = (dK / k ** 2) * np.exp(r * T) * q
    return float((2.0 / T) * np.nansum(contributions) - (1.0 / T) * (F / K_0 - 1.0) ** 2)


# --------------------------------------------------------------------------
# 5. full pipeline for one snapshot
# --------------------------------------------------------------------------

def _rate(r_curve, T):
    """A flat rate or a callable r_curve(T) -> rate."""
    return float(r_curve(T)) if callable(r_curve) else float(r_curve)


def _prepare_chain(chain):
    """Add is_call / mid if missing and drop rows with no usable quote.

    Zero-bid options are kept: the truncation rule needs them. Only rows
    with no ask at all (a zero mid) are removed.
    """
    df = chain.copy()
    if "is_call" not in df.columns:
        df["is_call"] = df["right"].astype(str).str.upper().str[0] == "C"
    if "mid" not in df.columns:
        df["mid"] = 0.5 * (df["bid"] + df["ask"])
    return df[np.isfinite(df["mid"]) & (df["mid"] > 0.0)].copy()


def _interpolate_to_30d(near, next_term):
    """Blend two expiry variances to exactly 30 days and annualise."""
    N1, N2 = near["minutes"], next_term["minutes"]
    w_near = (N2 - MINUTES_PER_30D) / (N2 - N1)
    w_next = (MINUTES_PER_30D - N1) / (N2 - N1)
    blended = (near["T"] * near["sigma2"] * w_near
               + next_term["T"] * next_term["sigma2"] * w_next)
    return 100.0 * np.sqrt(blended * MINUTES_PER_YEAR / MINUTES_PER_30D)


def reconstruct(chain, r_curve, snapshot_ts=None):
    """Full reconstruction for one snapshot.

    Returns a dict with the VIX level and the full per-leg diagnostics
    (forward, K_0, variance, strike counts, dropped wings) so the benchmark
    script and the error decomposition can see inside the number.
    """
    chain = _prepare_chain(chain)
    near_sel, next_sel = select_expiries(chain)

    legs = {}
    for name, term in (("near", near_sel), ("next", next_sel)):
        T = term["T"]
        r = _rate(r_curve, T)
        F, K_0, K_star = forward_from_parity(term["sub"], r, T)
        strip, dropped = strike_strip(term["sub"], K_0)
        legs[name] = {
            "expiry": term["expiry"], "root": term["root"],
            "minutes": term["minutes"], "T": T, "days": term["days"],
            "r": r, "F": F, "K0": K_0, "K_star": K_star,
            "sigma2": expiry_variance(strip, F, K_0, r, T),
            "n_strikes": int(len(strip)), "dropped": dropped, "strip": strip,
        }

    vix = _interpolate_to_30d(legs["near"], legs["next"])
    if snapshot_ts is None:
        snapshot_ts = chain["snapshot_utc"].iloc[0]
    return {"vix": float(vix), "near": legs["near"], "next": legs["next"],
            "snapshot_utc": snapshot_ts}


def compute_vix(chain, r_curve, snapshot_ts=None):
    """Full pipeline for one snapshot. Returns the VIX level."""
    return reconstruct(chain, r_curve, snapshot_ts)["vix"]


# --------------------------------------------------------------------------
# 6. benchmark 8
# --------------------------------------------------------------------------

def _as_date_series(obj, name):
    """Coerce a DataFrame (date + value column) or Series to a date-indexed Series."""
    if isinstance(obj, pd.Series):
        s = obj.copy()
    else:
        df = obj.copy()
        date_col = next((c for c in df.columns
                         if c.lower() in ("date", "snapshot_utc", "snapshot", "day")), df.columns[0])
        value_candidates = [name, "vix", "computed", "published", "close", "value"]
        value_col = next((c for c in value_candidates if c in df.columns),
                         df.columns[-1])
        s = df.set_index(date_col)[value_col]
    s.index = pd.to_datetime(s.index).normalize()
    s.name = name
    return s


def benchmark_vs_published(computed, published):
    """Benchmark 8.

    Returns a DataFrame with columns date, computed, published, abs_error,
    and a summary dict (mean/max absolute error, correlation, n) on
    ``result.attrs["summary"]``. Target is mean absolute error below 0.20
    vol points.
    """
    comp = _as_date_series(computed, "computed")
    pub = _as_date_series(published, "published")
    joined = pd.concat([comp, pub], axis=1, join="inner")
    joined = joined.groupby(level=0).mean()  # collapse intraday snapshots to a daily value
    result = joined.reset_index()
    result.columns = ["date", "computed", "published"]
    result["abs_error"] = (result["computed"] - result["published"]).abs()

    result.attrs["summary"] = {
        "mean_abs_error": float(result["abs_error"].mean()),
        "max_abs_error": float(result["abs_error"].max()),
        "correlation": float(result["computed"].corr(result["published"]))
        if len(result) > 1 else float("nan"),
        "n": int(len(result)),
    }
    return result
