import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optionslab.bs import bs_price, forward_from_spot
from optionslab.lsm import price_american_put

# (S, sigma, T) -> (simulated price, standard error), transcribed from the
# "Simulated American (s.e.)" column of Table 1. Kept in step with the copy
# in tests/test_lsm.py, which is the authoritative one.
REFERENCE = {
    (36.0, 0.20, 1.0): (4.472, 0.010), (36.0, 0.20, 2.0): (4.821, 0.012),
    (36.0, 0.40, 1.0): (7.091, 0.020), (36.0, 0.40, 2.0): (8.488, 0.024),
    (38.0, 0.20, 1.0): (3.244, 0.009), (38.0, 0.20, 2.0): (3.735, 0.011),
    (38.0, 0.40, 1.0): (6.139, 0.019), (38.0, 0.40, 2.0): (7.669, 0.022),
    (40.0, 0.20, 1.0): (2.313, 0.009), (40.0, 0.20, 2.0): (2.879, 0.010),
    (40.0, 0.40, 1.0): (5.308, 0.018), (40.0, 0.40, 2.0): (6.921, 0.022),
    (42.0, 0.20, 1.0): (1.617, 0.007), (42.0, 0.20, 2.0): (2.206, 0.010),
    (42.0, 0.40, 1.0): (4.588, 0.017), (42.0, 0.40, 2.0): (6.243, 0.021),
    (44.0, 0.20, 1.0): (1.118, 0.007), (44.0, 0.20, 2.0): (1.675, 0.009),
    (44.0, 0.40, 1.0): (3.957, 0.017), (44.0, 0.40, 2.0): (5.622, 0.021),
}


def main():
    K, r = 40.0, 0.06
    n_paths, seed, degree = 200_000, 1, 3

    records = []
    for (S0, sigma, T), (reference, reference_error) in REFERENCE.items():
        ours, our_error = price_american_put(S0, K, r, sigma, T, n_paths=n_paths,
                                             seed=seed, degree=degree)
        forward = forward_from_spot(S0, r, 0.0, T)
        european = float(bs_price(forward, K, T, sigma, np.exp(-r * T), False))
        combined = np.sqrt(our_error ** 2 + reference_error ** 2)

        records.append({"S": S0, "sigma": sigma, "T": T, "ours": ours,
                        "our_se": our_error, "paper": reference,
                        "paper_se": reference_error, "difference": ours - reference,
                        "combined_se": combined, "z": (ours - reference) / combined,
                        "european": european,
                        "early_exercise_premium": ours - european})

    table = pd.DataFrame(records)
    z = table["z"].to_numpy()

    print("Benchmark 6 — Longstaff & Schwartz (2001) Table 1")
    print(f"K = {K}, r = {r}, {n_paths:,} paths, 50 exercise dates per year, "
          f"basis degree {degree}, seed {seed}\n")
    print(table.round(4).to_string(index=False))

    print(f"\ncells within 3 combined s.e. : {(np.abs(z) < 3.0).sum()}/{len(z)}")
    print(f"max |z|                      : {np.abs(z).max():.2f}")
    print(f"mean z                       : {z.mean():+.3f}")
    print("\nThe mean is the informative statistic. A near-zero mean indicates no")
    print("systematic bias against the published values; a large one would point")
    print("to a difference in the exercise policy rather than to sampling noise.")

    if (np.abs(z) < 3.0).all():
        print("\nBENCHMARKS.md row 6:")
        print(f"  20/20 cells pass; max {np.abs(z).max():.2f} combined s.e., "
              f"mean z = {z.mean():+.2f}")

    output = Path("data/processed")
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "lsm_replication.csv", index=False)
    print(f"\nwritten: {output / 'lsm_replication.csv'}")


if __name__ == "__main__":
    main()
