"""Tests for chain loading, cleaning and the parity-based forward.

The forward test is a known answer: quotes are built so that
call_mid - put_mid = df * (F - K) exactly, so the regression must return
the F and df that went in. The load test guards the snapshot_ts ->
snapshot_utc rename that every real file depends on.
"""

import numpy as np
import pandas as pd
import pytest

from optionslab.chain import (
    add_implied_vols,
    clean_chain,
    forwards_by_expiry,
    implied_forward,
    load_chain,
    otm_only,
    pair_by_strike,
    year_fraction,
)


def test_year_fraction_is_act_365():
    snap = pd.Series(pd.to_datetime(["2026-01-01"], utc=True))
    expiry = pd.Series(pd.to_datetime(["2027-01-01"], utc=True))
    assert float(year_fraction(snap, expiry).iloc[0]) == pytest.approx(1.0, abs=1e-9)


def _write_snapshot_parquet(path):
    """A minimal snapshot as snapshot.py writes it: snapshot_ts, not _utc."""
    snap = pd.Timestamp("2026-08-12 15:30", tz="UTC")
    rows = []
    for K in (95.0, 100.0, 105.0):
        for right in ("C", "P"):
            rows.append({"strike": K, "expiry": pd.Timestamp("2026-09-11"),
                         "right": right, "bid": 1.0, "ask": 1.2,
                         "snapshot_ts": snap})
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_load_chain_renames_snapshot_ts_and_derives_columns(tmp_path):
    path = tmp_path / "spx_test.parquet"
    _write_snapshot_parquet(path)
    chain = load_chain(path)

    # the rename every real file depends on
    assert "snapshot_utc" in chain.columns and "snapshot_ts" not in chain.columns
    # derived columns
    for col in ("K", "T", "is_call", "mid", "spread"):
        assert col in chain.columns
    assert chain["is_call"].sum() == 3                     # three calls
    np.testing.assert_allclose(chain["mid"].unique(), [1.1])
    np.testing.assert_allclose(chain["spread"].unique(), [0.2])
    assert (chain["T"] > 0).all()


def test_clean_chain_rejects_bad_quotes():
    snap = pd.Timestamp("2026-08-12", tz="UTC")
    base = {"snapshot_utc": snap, "is_call": True, "K": 100.0}
    chain = pd.DataFrame([
        {**base, "bid": 1.0, "ask": 1.1, "mid": 1.05, "spread": 0.1, "T": 0.1},   # good
        {**base, "bid": 0.0, "ask": 0.5, "mid": 0.25, "spread": 0.5, "T": 0.1},   # no market
        {**base, "bid": 0.01, "ask": 0.02, "mid": 0.015, "spread": 0.01, "T": 0.1},  # below min bid
        {**base, "bid": 1.0, "ask": 5.0, "mid": 3.0, "spread": 4.0, "T": 0.1},    # spread too wide
        {**base, "bid": 1.0, "ask": 1.1, "mid": 1.05, "spread": 0.1, "T": 0.0},   # expired
    ])
    kept = clean_chain(chain, min_bid=0.05, max_relative_spread=1.0)
    assert len(kept) == 1
    assert float(kept["bid"].iloc[0]) == 1.0


def _parity_expiry(F=100.0, discount=0.98, strikes=(80, 90, 95, 100, 105, 110, 120)):
    """One expiry whose quotes obey put-call parity exactly for (F, discount)."""
    rows = []
    for K in strikes:
        put_mid = 50.0
        call_mid = put_mid + discount * (F - K)
        rows.append({"K": float(K), "is_call": True, "mid": call_mid, "spread": 0.1,
                     "expiry": pd.Timestamp("2026-09-11"), "T": 0.08})
        rows.append({"K": float(K), "is_call": False, "mid": put_mid, "spread": 0.1,
                     "expiry": pd.Timestamp("2026-09-11"), "T": 0.08})
    return pd.DataFrame(rows)


def test_pair_by_strike_computes_parity_difference():
    expiry = _parity_expiry()
    paired = pair_by_strike(expiry)
    assert list(paired["K"]) == sorted(paired["K"])
    expected = paired["call_mid"] - paired["put_mid"]
    np.testing.assert_allclose(paired["parity_difference"], expected)


def test_implied_forward_recovers_known_forward_and_discount():
    expiry = _parity_expiry(F=100.0, discount=0.98)
    fit = implied_forward(expiry)
    assert fit["F"] == pytest.approx(100.0, abs=1e-6)
    assert fit["df"] == pytest.approx(0.98, abs=1e-6)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-9)


def test_implied_forward_reports_nan_when_too_few_pairs():
    expiry = _parity_expiry(strikes=(95, 100))     # only two pairs, min is four
    fit = implied_forward(expiry)
    assert np.isnan(fit["F"])
    assert fit["n_pairs"] == 2


def test_forwards_by_expiry_one_row_per_expiry():
    expiry = _parity_expiry()
    forwards = forwards_by_expiry(expiry)
    assert len(forwards) == 1
    assert forwards["F"].iloc[0] == pytest.approx(100.0, abs=1e-6)


def test_otm_only_keeps_out_of_the_money_side():
    chain = pd.DataFrame({
        "K": [90.0, 90.0, 110.0, 110.0],
        "is_call": [True, False, True, False],
        "F": [100.0, 100.0, 100.0, 100.0],
    })
    otm = otm_only(chain)
    # below the forward keep puts, above the forward keep calls
    assert set(zip(otm["K"], otm["is_call"])) == {(90.0, False), (110.0, True)}
