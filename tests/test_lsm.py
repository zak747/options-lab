"""Benchmark 6."""

import pytest

pytestmark = pytest.mark.skip(reason="lsm.py not yet implemented")


def test_american_call_no_dividends_equals_european():
    """The sharpest available correctness check: with no dividends an
    American call is never optimally exercised early, so LSM must
    reproduce the Black-Scholes call price within Monte Carlo error.
    If this fails, the backward induction is wrong."""
    raise NotImplementedError


def test_american_put_at_least_european():
    """Early exercise has non-negative value, so the American put price
    must not fall below the European price."""
    raise NotImplementedError


def test_longstaff_schwartz_table():
    """Benchmark 6. Reproduce the published American put table within
    the standard errors reported in the source.

    Longstaff & Schwartz (2001), RFS 14(1). K = 40, r = 0.06,
    spot in {36, 38, 40, 42, 44}, sigma in {0.20, 0.40}, T in {1, 2}.
    """
    raise NotImplementedError


def test_itm_restriction_matters():
    """Regressing on all paths rather than ITM paths only should change
    the price materially. Documenting the size of that difference is the
    justification for the choice recorded in DEVIATIONS.md."""
    raise NotImplementedError
