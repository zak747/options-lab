import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optionslab.hedge import hedging_error_study, vol_mismatch_study


def main():
    grid = [10, 20, 40, 80, 160, 320, 640]
    table, slope, slope_error = hedging_error_study(grid, n_paths=100_000, seed=0)

    print("Benchmark 7 — discrete hedging error scaling")
    print(table.to_string(index=False))
    print(f"\nfitted exponent: {slope:.4f} +/- {slope_error:.4f}   (target -0.50 +/- 0.05)")

    print("\n\nVolatility mismatch — sold at 20%")
    for hedge_at in ("realised", "priced"):
        mismatch = vol_mismatch_study(0.20, [0.10, 0.15, 0.20, 0.25, 0.30],
                                      hedge_at=hedge_at, n_steps=500,
                                      n_paths=50_000, seed=5)
        print(f"\nhedged at {hedge_at} volatility:")
        print(mismatch.to_string(index=False))

    output = Path("data/processed")
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "hedging_error_scaling.csv", index=False)
    print(f"\nwritten: {output / 'hedging_error_scaling.csv'}")


if __name__ == "__main__":
    main()
