"""Tests for the CBOE VIX reconstruction.

No value is copied from the CBOE white paper here. Every expected number is
either a property that must hold (an expiry pair that brackets 30 days, a
forward that inverts put-call parity) or a known answer generated from the
project's own Black-Scholes pricer: a chain quoted at a single constant
volatility must reconstruct to a VIX of 100 * sigma, because the variance
strip is an exact replication of the log contract. That end-to-end test is
the one that would catch a VIX of 5 instead of 20, or a NaN.
"""

import numpy as np
import pandas as pd
import pytest

from optionslab.bs import bs_price
from optionslab.vix import (
    MINUTES_PER_YEAR,
    _settlement_utc,
    compute_vix,
    expiry_variance,
    forward_from_parity,
    select_expiries,
    strike_strip,
    benchmark_vs_published,
)

SNAPSHOT = pd.Timestamp("2026-08-12 15:30", tz="UTC")


def _horizon(expiry):
    """Minute-precise T the pipeline will assign to a (PM-settled) expiry."""
    settle = _settlement_utc(pd.Timestamp(expiry, tz="UTC"), None)
    minutes = (settle - SNAPSHOT).total_seconds() / 60.0
    return minutes / MINUTES_PER_YEAR


def _bs_expiry_rows(expiry, F, sigma, r, strikes):
    """OTM-and-through Black-Scholes quotes for one expiry, bid=ask=mid=price.

    Priced at the exact horizon the reconstruction will use, so the only
    residual between the recovered variance and sigma**2 is discretisation.
    """
    T = _horizon(expiry)
    df = np.exp(-r * T)
    rows = []
    for K in strikes:
        for is_call in (True, False):
            price = float(bs_price(F, K, T, sigma, df, is_call))
            rows.append({
                "snapshot_utc": SNAPSHOT,
                "expiry": pd.Timestamp(expiry, tz="UTC"),
                "K": float(K),
                "right": "C" if is_call else "P",
                "is_call": is_call,
                "bid": price,
                "ask": price,
                "mid": price,
            })
    return rows


def _constant_vol_chain(sigma=0.20, F=100.0, r=0.02,
                        near="2026-09-10", far="2026-09-11"):
    strikes = np.arange(40.0, 200.0, 1.0)
    rows = _bs_expiry_rows(near, F, sigma, r, strikes)
    rows += _bs_expiry_rows(far, F, sigma, r, strikes)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# expiry selection
# --------------------------------------------------------------------------

def test_select_expiries_brackets_thirty_days():
    chain = _constant_vol_chain()
    near, nxt = select_expiries(chain)
    assert near["days"] <= 30.0 < nxt["days"]
    assert 23.0 <= near["days"] and nxt["days"] <= 37.0
    assert near["minutes"] < nxt["minutes"]


def test_select_expiries_needs_two_expiries_in_window():
    # a lone expiry ~30 days out cannot support a 30-day interpolation
    chain = pd.DataFrame(_bs_expiry_rows("2026-09-11", 100.0, 0.2, 0.02,
                                         np.arange(80.0, 120.0, 5.0)))
    with pytest.raises(ValueError):
        select_expiries(chain)


# --------------------------------------------------------------------------
# forward and K_0
# --------------------------------------------------------------------------

def test_forward_inverts_put_call_parity():
    # By construction C - P = df * (F - K), so the parity forward must be F.
    F, sigma, r = 100.0, 0.2, 0.02
    rows = _bs_expiry_rows("2026-09-10", F, sigma, r, np.arange(80.0, 121.0, 1.0))
    df_expiry = pd.DataFrame(rows)
    T = _horizon("2026-09-10")
    recovered, K_0, K_star = forward_from_parity(df_expiry, r, T)
    assert recovered == pytest.approx(F, abs=1e-6)
    assert K_0 <= recovered
    assert K_0 == pytest.approx(100.0)          # highest strike at or below F
    assert abs(K_star - F) <= 1.0               # |C-P| smallest nearest the money


# --------------------------------------------------------------------------
# the strip and the two-consecutive-zero-bid rule
# --------------------------------------------------------------------------

def _wing_rows(specs, is_call):
    """specs: list of (strike, bid). mid is set to max(bid, 0.05)."""
    right = "C" if is_call else "P"
    return [{"K": float(k), "right": right, "is_call": is_call,
             "bid": float(b), "ask": float(b) + 0.1,
             "mid": max(float(b), 0.05)} for k, b in specs]


def test_two_consecutive_zero_bids_truncate_the_wing():
    K_0 = 100.0
    puts = _wing_rows([(95, 1.0), (90, 0.0), (85, 0.0), (80, 2.0)], is_call=False)
    calls = _wing_rows([(105, 1.0), (110, 1.0), (115, 0.0), (120, 0.0), (125, 3.0)],
                       is_call=True)
    at_money = (_wing_rows([(100, 1.0)], is_call=True)
                + _wing_rows([(100, 1.0)], is_call=False))
    df_expiry = pd.DataFrame(puts + calls + at_money)

    strip, dropped = strike_strip(df_expiry, K_0)
    kept = set(strip["strike"])

    # put wing stops at the second zero (85); 80 is discarded despite its bid
    assert 95.0 in kept and 80.0 not in kept
    # call wing stops at the second zero (120); 125 is discarded
    assert {105.0, 110.0}.issubset(kept) and 125.0 not in kept
    assert 100.0 in kept
    assert dropped["put_wing"] == 3 and dropped["call_wing"] == 3


def test_delta_k_is_central_inside_and_one_sided_at_the_ends():
    K_0 = 100.0
    puts = _wing_rows([(90, 1.0), (95, 1.0)], is_call=False)
    calls = _wing_rows([(110, 1.0), (115, 1.0)], is_call=True)
    at_money = (_wing_rows([(100, 1.0)], is_call=True)
                + _wing_rows([(100, 1.0)], is_call=False))
    strip, _ = strike_strip(pd.DataFrame(puts + calls + at_money), K_0)

    strikes = strip["strike"].to_numpy()
    dK = strip["dK"].to_numpy()
    assert list(strikes) == [90, 95, 100, 110, 115]
    np.testing.assert_allclose(dK, [5.0, 5.0, 7.5, 7.5, 5.0])


# --------------------------------------------------------------------------
# per-expiry variance
# --------------------------------------------------------------------------

def test_expiry_variance_matches_the_formula():
    strip = pd.DataFrame({
        "strike": [95.0, 100.0, 105.0],
        "price": [0.80, 1.20, 0.70],
        "dK": [5.0, 5.0, 5.0],
    })
    F, K_0, r, T = 100.0, 100.0, 0.03, 0.08
    expected = (2.0 / T) * np.sum(
        (strip["dK"] / strip["strike"] ** 2) * np.exp(r * T) * strip["price"]
    ) - (1.0 / T) * (F / K_0 - 1.0) ** 2
    assert expiry_variance(strip, F, K_0, r, T) == pytest.approx(float(expected))


# --------------------------------------------------------------------------
# end-to-end known answer
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", [0.12, 0.20, 0.35])
def test_constant_vol_chain_reconstructs_to_100_sigma(sigma):
    chain = _constant_vol_chain(sigma=sigma)
    vix = compute_vix(chain, r_curve=0.02)
    assert vix == pytest.approx(100.0 * sigma, abs=0.5)


def test_flat_rate_and_callable_rate_curve_agree():
    chain = _constant_vol_chain(sigma=0.2, r=0.03)
    flat = compute_vix(chain, r_curve=0.03)
    curve = compute_vix(chain, r_curve=lambda T: 0.03)
    assert flat == pytest.approx(curve, abs=1e-9)


# --------------------------------------------------------------------------
# benchmark 8 bookkeeping
# --------------------------------------------------------------------------

def test_benchmark_vs_published_reports_errors():
    computed = pd.DataFrame({
        "snapshot_utc": pd.to_datetime(["2026-08-10", "2026-08-11", "2026-08-12"]),
        "vix": [14.8, 15.2, 16.0],
    })
    published = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-10", "2026-08-11", "2026-08-12"]),
        "close": [15.0, 15.0, 16.0],
    })
    result = benchmark_vs_published(computed, published)
    assert list(result.columns) == ["date", "computed", "published", "abs_error"]
    np.testing.assert_allclose(sorted(result["abs_error"]), [0.0, 0.2, 0.2], atol=1e-9)
    summary = result.attrs["summary"]
    assert summary["n"] == 3
    assert summary["mean_abs_error"] == pytest.approx((0.2 + 0.2 + 0.0) / 3)
    assert summary["max_abs_error"] == pytest.approx(0.2)
