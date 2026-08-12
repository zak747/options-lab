"""Tests for the SVI surface fit and its arbitrage diagnostics.

The fit test is a known answer: total variance is generated from a chosen
SVI slice, so the fitted curve must reproduce it. The diagnostics are
tested by construction — benign parameters pass, and deliberately
arbitrageable ones are flagged.
"""

import numpy as np
import pandas as pd
import pytest

from optionslab.surface import (
    butterfly_function,
    butterfly_violations_from_quotes,
    check_butterfly,
    check_calendar,
    check_parameter_constraints,
    fit_summary,
    fit_svi,
    risk_neutral_density,
    svi_derivatives,
    svi_total_variance,
)

BENIGN = np.array([0.04, 0.10, -0.30, 0.00, 0.20])   # a, b, rho, m, s


def test_svi_total_variance_matches_the_closed_form():
    a, b, rho, m, s = BENIGN
    k = np.array([-0.2, 0.0, 0.3])
    shifted = k - m
    expected = a + b * (rho * shifted + np.sqrt(shifted ** 2 + s ** 2))
    np.testing.assert_allclose(svi_total_variance(k, BENIGN), expected)


def test_svi_derivatives_match_finite_difference():
    k = np.array([-0.4, -0.1, 0.25, 0.5])
    h = 1e-5
    first, second = svi_derivatives(k, BENIGN)
    w_plus = svi_total_variance(k + h, BENIGN)
    w_minus = svi_total_variance(k - h, BENIGN)
    w_mid = svi_total_variance(k, BENIGN)
    fd_first = (w_plus - w_minus) / (2 * h)
    fd_second = (w_plus - 2 * w_mid + w_minus) / h ** 2
    np.testing.assert_allclose(first, fd_first, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(second, fd_second, rtol=1e-3, atol=1e-6)


def test_fit_recovers_a_known_svi_slice():
    k = np.linspace(-0.6, 0.6, 41)
    w = svi_total_variance(k, BENIGN)
    params = fit_svi(k, w, n_starts=8, seed=1)
    fitted = svi_total_variance(k, params)
    # the total-variance curve must be reproduced, even if the raw
    # parameter vector is not identical
    assert np.max(np.abs(fitted - w)) < 1e-4


def test_benign_slice_is_free_of_butterfly_arbitrage():
    report = check_butterfly(BENIGN)
    assert report["n_violations"] == 0
    assert report["min_g"] > 0.0


def test_risk_neutral_density_is_positive_and_integrates_to_one():
    k = np.linspace(-4.0, 4.0, 20001)
    density = risk_neutral_density(k, BENIGN)
    assert (density >= 0.0).all()
    integral = getattr(np, "trapezoid", np.trapz)(density, k)
    assert integral == pytest.approx(1.0, abs=1e-2)


def test_parameter_constraints_pass_for_benign_and_flag_bad():
    good = check_parameter_constraints(BENIGN)
    assert good["b_non_negative"] and good["rho_in_range"] and good["s_positive"]
    assert good["variance_non_negative"] and good["wing_slope_below_four"]

    bad = check_parameter_constraints(np.array([-0.5, -0.1, 1.5, 0.0, -0.2]))
    assert not bad["b_non_negative"]
    assert not bad["rho_in_range"]
    assert not bad["s_positive"]


def test_calendar_flags_decreasing_total_variance():
    near = np.array([0.02, 0.10, -0.30, 0.00, 0.20])
    far_ok = np.array([0.05, 0.10, -0.30, 0.00, 0.20])     # uniformly higher variance
    far_bad = np.array([0.005, 0.10, -0.30, 0.00, 0.20])   # lower than near everywhere

    ok = check_calendar({0.08: near, 0.25: far_ok})
    assert ok["n_violations"] == 0 and ok["worst_gap"] >= 0.0

    bad = check_calendar({0.08: near, 0.25: far_bad})
    assert bad["n_violations"] > 0 and bad["worst_gap"] < 0.0


def test_fit_summary_reports_zero_error_on_exact_fit():
    k = np.linspace(-0.5, 0.5, 21)
    T = 0.1
    w = svi_total_variance(k, BENIGN)
    summary = fit_summary(k, w, BENIGN, T)
    assert summary["rmse_vol_points"] == pytest.approx(0.0, abs=1e-12)
    assert summary["max_abs_error"] == pytest.approx(0.0, abs=1e-12)
    assert summary["n_quotes"] == len(k)


def test_butterfly_violations_from_quotes_detects_non_convexity():
    strikes = [90.0, 100.0, 110.0]
    # convex, decreasing call prices -> no butterfly violation
    convex = butterfly_violations_from_quotes(strikes, [12.0, 6.0, 3.0])
    assert convex.empty
    # a dented middle price violates convexity (butterfly costs < 0)
    # a middle price above its neighbours' interpolant is the arbitrage
    dented = butterfly_violations_from_quotes(strikes, [12.0, 11.0, 3.0])
    assert len(dented) == 1
    assert dented["cost"].iloc[0] < 0.0
