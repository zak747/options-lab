"""Option chain ingestion, cleaning and forward extraction.

The unglamorous part, and the part that separates this from every other
options repository. A raw chain is full of stale quotes, zero bids,
crossed markets and strikes that have not traded in weeks.

The important function here is `implied_forward`. Put-call parity says

    C - P = DF*F - DF*K

so regressing (C - P) on K across all strikes of a single expiry gives
slope = -DF and intercept = DF*F. The forward and the discount factor
come out of the option prices themselves — no assumption about dividend
yield or funding rate is required. The R^2 of that regression is also a
clean data quality diagnostic.
"""

import numpy as np
import pandas as pd

__all__ = [
    "load_chain",
    "clean_chain",
    "implied_forward",
    "add_implied_vols",
]


def load_chain(path):
    """Read one snapshot into a tidy DataFrame.

    Expected columns after parsing: expiry, strike, right ('C'/'P'), bid,
    ask, mid, volume, open_interest, snapshot_ts (UTC).

    Compute time to expiry as an explicit year fraction using the day
    count in config.py. Record the convention; it moves short-dated
    implied vols measurably.
    """
    raise NotImplementedError


def clean_chain(df, max_rel_spread=None):
    """Apply quote filters.

    Drop: zero or missing bids, crossed markets (bid > ask), and strikes
    whose relative spread exceeds `max_rel_spread`.

    Keep the spread width as a column rather than discarding it — it is
    the natural weight for the surface fit, and throwing it away loses
    information about which quotes deserve to be trusted.

    Return the filtered frame plus a dict recording how many rows each
    filter removed. Those counts belong in MEMO.md.
    """
    raise NotImplementedError


def implied_forward(df_expiry):
    """Estimate (F, DF) for one expiry by regressing (C - P) on K.

    Requires strikes with both a call and a put quote. Returns
    (F, DF, r_squared, n_strikes). A low R^2 means the quotes for that
    expiry are not internally consistent and the expiry should probably
    be dropped.
    """
    raise NotImplementedError


def add_implied_vols(df, forwards):
    """Attach an implied vol to every OTM quote.

    OTM only: calls above the forward, puts below. See iv.py for why.
    """
    raise NotImplementedError
