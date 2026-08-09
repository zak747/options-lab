"""Paths and shared constants.

No absolute path appears anywhere else in the project. Everything is
resolved relative to the repository root so that a clone runs unmodified
on another machine.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"

# Day count used to convert calendar dates to year fractions.
# ACT/365F is stated explicitly rather than assumed; see DEVIATIONS.md.
DAYS_PER_YEAR = 365.0

# Default RNG seed. Every stochastic result in the repo is reproducible.
SEED = 20260809
