#!/usr/bin/env python3
"""Benchmark 8 — VIX reconstruction against CBOE published closes.

The headline result. A reader should be able to open this file and see
the whole thing at a glance: load every snapshot, reconstruct VIX for
each, print the maturities / forward / K_0 / strike counts that went into
it, and write one tidy row per snapshot.

Usage
-----
    python scripts/run_vix_benchmark.py                 # rate 0.04
    python scripts/run_vix_benchmark.py --rate 0.045    # flat rate override
    python scripts/run_vix_benchmark.py --published data/raw/vix_closes.csv

Writes: data/processed/vix_reconstruction.csv (one row per snapshot). If a
published-closes CSV is supplied (columns: date, close), also reports the
mean and max absolute error against it — that is benchmark 8 proper.
"""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from optionslab.chain import load_chain            # noqa: E402
from optionslab.config import DATA_RAW, DATA_PROCESSED  # noqa: E402
from optionslab.vix import reconstruct, benchmark_vs_published  # noqa: E402


def reconstruct_all(files, rate):
    """One diagnostics row per snapshot file."""
    rows = []
    for path in files:
        name = Path(path).name
        chain = load_chain(path)
        try:
            result = reconstruct(chain, rate)
        except Exception as exc:  # a thin chain simply cannot make a 30-day VIX
            print(f"  {name}: skipped — {type(exc).__name__}: {exc}")
            continue
        near, nxt = result["near"], result["next"]
        rows.append({
            "snapshot_utc": pd.Timestamp(result["snapshot_utc"]),
            "file": name,
            "vix": round(result["vix"], 4),
            "near_days": round(near["days"], 3),
            "next_days": round(nxt["days"], 3),
            "near_F": round(near["F"], 2),
            "next_F": round(nxt["F"], 2),
            "near_K0": near["K0"],
            "next_K0": nxt["K0"],
            "near_strikes": near["n_strikes"],
            "next_strikes": nxt["n_strikes"],
        })
        print(
            f"  {name}: VIX = {result['vix']:6.2f}   "
            f"near {near['days']:5.2f}d (F={near['F']:.0f}, K0={near['K0']:.0f}, "
            f"{near['n_strikes']} strikes)   "
            f"next {nxt['days']:5.2f}d (F={nxt['F']:.0f}, K0={nxt['K0']:.0f}, "
            f"{nxt['n_strikes']} strikes)"
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rate", type=float, default=0.04,
                    help="flat continuously-compounded rate (default 0.04)")
    ap.add_argument("--raw", default=str(DATA_RAW),
                    help="directory of spx_*.parquet snapshots")
    ap.add_argument("--published", default=None,
                    help="optional CSV of published VIX closes (date, close)")
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.raw) / "spx_*.parquet")))
    if not files:
        print(f"no snapshots found in {args.raw} — run scripts/snapshot.py first")
        return

    print(f"reconstructing VIX from {len(files)} snapshot(s) at rate {args.rate:.3%}:")
    table = reconstruct_all(files, args.rate)
    if table.empty:
        print("no snapshot produced a VIX — nothing written")
        return

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = DATA_PROCESSED / "vix_reconstruction.csv"
    table.to_csv(out, index=False)

    print(f"\nVIX range: {table['vix'].min():.2f} .. {table['vix'].max():.2f} "
          f"(mean {table['vix'].mean():.2f}) across {len(table)} snapshot(s)")
    print(f"written: {out}")

    if args.published:
        published = pd.read_csv(args.published)
        bench = benchmark_vs_published(table[["snapshot_utc", "vix"]], published)
        summary = bench.attrs["summary"]
        print("\nbenchmark 8 — vs published closes:")
        print(bench.to_string(index=False))
        print(f"  mean abs error : {summary['mean_abs_error']:.3f} vol pts")
        print(f"  max  abs error : {summary['max_abs_error']:.3f} vol pts")
        print(f"  correlation    : {summary['correlation']:.4f}")


if __name__ == "__main__":
    main()
