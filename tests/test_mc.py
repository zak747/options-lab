"""Benchmarks 4 and 5."""

import pytest

pytestmark = pytest.mark.skip(reason="mc.py not yet implemented")


def test_mc_converges_to_analytic():
    """At a large path count, the MC price is within three standard
    errors of the Black-Scholes value. Uses a fixed seed."""
    raise NotImplementedError


def test_antithetic_preserves_mean():
    """Antithetic sampling changes the variance, not the expectation.
    The estimate must remain unbiased."""
    raise NotImplementedError


def test_control_variate_vrf_matches_theory():
    """Benchmark 5. The achieved variance reduction factor agrees with
    1/(1 - rho^2) to within 5%, using the empirically estimated rho."""
    raise NotImplementedError


def test_sobol_beats_pseudorandom():
    """Scrambled Sobol achieves lower RMSE than pseudorandom sampling at
    equal path count. State the expected direction and test it, rather
    than asserting a specific improvement factor."""
    raise NotImplementedError
