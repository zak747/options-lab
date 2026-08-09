#!/usr/bin/env python3
"""Daily SPX option chain snapshot.

Option chain history cannot be back-filled. Every evening this is not run
is a day of data permanently lost, and benchmark 8 is only as strong as
the number of days it can be tested on.

Usage
-----
    python scripts/snapshot.py              # fetch and write
    python scripts/snapshot.py --dry-run    # fetch, report, write nothing

Scheduling (macOS / Linux). SPX options trade until 16:15 US Eastern.
Run a few minutes after that:

    crontab -e
    15 21 * * 1-5 cd /path/to/options-lab && .venv/bin/python scripts/snapshot.py >> data/raw/snapshot.log 2>&1

21:15 is British Summer Time. This shifts to 21:15 GMT-relative in
winter; the cron entry needs revisiting at the clock change, or set
CRON_TZ=America/New_York and use 16:20.
"""

import argparse
import json
import sys
from datetime import datetime, timezone

import pandas as pd
import requests

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from optionslab.config import DATA_RAW  # noqa: E402

URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"
HEADERS = {"User-Agent": "options-lab research snapshot"}
TIMEOUT = 30


def fetch(url=URL):
    """Fetch the raw JSON payload."""
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def parse(payload):
    """Flatten the payload into a tidy DataFrame.

    CBOE encodes expiry, right and strike in the option symbol, in the
    OCC format:  ROOT + YYMMDD + C/P + strike*1000 padded to 8 digits.

    VERIFY THIS AGAINST A REAL PAYLOAD BEFORE TRUSTING IT. The endpoint
    is undocumented and its schema is not guaranteed stable. Run with
    --dry-run first and inspect the columns.
    """
    options = payload["data"]["options"]
    df = pd.DataFrame(options)

    sym = df["option"].str.extract(
        r"^(?P<root>[A-Z]+)(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})"
        r"(?P<right>[CP])(?P<strike>\d{8})$"
    )
    df["expiry"] = pd.to_datetime(
        "20" + sym["yy"] + "-" + sym["mm"] + "-" + sym["dd"]
    )
    df["right"] = sym["right"]
    df["strike"] = sym["strike"].astype(float) / 1000.0

    df["underlying_last"] = payload["data"].get("close")
    df["snapshot_ts"] = datetime.now(timezone.utc)

    keep = [
        "expiry", "strike", "right", "bid", "ask", "last_trade_price",
        "volume", "open_interest", "iv", "delta", "gamma",
        "underlying_last", "snapshot_ts",
    ]
    return df[[c for c in keep if c in df.columns]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()

    payload = fetch(args.url)
    df = parse(payload)

    stamp = datetime.now(timezone.utc)
    print(f"snapshot {stamp:%Y-%m-%d %H:%M:%S} UTC")
    print(f"  rows            : {len(df):,}")
    print(f"  expiries        : {df['expiry'].nunique()}")
    print(f"  underlying      : {df['underlying_last'].iloc[0]}")
    print(f"  non-zero bids   : {(df['bid'] > 0).sum():,}")
    print(f"  columns         : {list(df.columns)}")

    if args.dry_run:
        print("dry run — nothing written")
        return

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out = DATA_RAW / f"spx_{stamp:%Y%m%d_%H%M}.parquet"
    df.to_parquet(out, index=False)
    print(f"  written         : {out}")


if __name__ == "__main__":
    main()
