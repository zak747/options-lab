"""Longstaff-Schwartz least-squares Monte Carlo for American options.

Early exercise turns pricing into an optimal stopping problem. At each
exercise date the continuation value is needed: what is holding worth,
against exercising now? Longstaff and Schwartz estimate it by regression.

Working backwards from expiry:

  1. Set cashflow = terminal payoff on every path.
  2. Step back one date. Select the paths that are IN THE MONEY — only
     these enter the regression.
  3. Regress the discounted future cashflow on basis functions of the
     current underlying price.
  4. The fitted value is the estimated continuation value. Where
     immediate exercise exceeds it, exercise: record the payoff at this
     date and zero all later cashflows on that path.
  5. Repeat to time zero, discount, average.

Points to record in DEVIATIONS.md:
  - Restricting the regression to ITM paths is essential; including OTM
    paths wrecks the fit and biases the price.
  - The resulting price is a LOWER BOUND in expectation, because the
    estimated exercise policy is suboptimal.
  - The basis function choice materially affects the answer, so run a
    sensitivity check over degree and family.

Reference: Longstaff & Schwartz (2001), "Valuing American Options by
Simulation: A Simple Least-Squares Approach", Review of Financial
Studies 14(1), 113-147.
"""

import numpy as np

__all__ = ["laguerre_basis", "polynomial_basis", "lsm_price"]


def laguerre_basis(S, degree):
    """Weighted Laguerre polynomials, as used in the source paper.

    Return the design matrix of shape (len(S), degree + 1).
    """
    raise NotImplementedError


def polynomial_basis(S, degree):
    """Plain powers of S. Works about as well in practice; keep both so
    the sensitivity check has something to compare."""
    raise NotImplementedError


def lsm_price(paths, K, dt, r, is_call=False, basis=laguerre_basis, degree=3):
    """Price an American option on the given paths.

    Parameters
    ----------
    paths : ndarray, shape (n_paths, n_steps + 1)
    K : float
    dt : float
        Time between exercise dates in years.
    r : float
        Continuously compounded rate, for discounting between dates.

    Returns
    -------
    (price, standard_error)
    """
    raise NotImplementedError
