import numpy as np
import pandas as pd

from optionslab.iv import implied_vol_chain


def year_fraction(snapshot_time, expiry_time):
    seconds_per_year = 365.0 * 24.0 * 3600.0
    elapsed = (expiry_time - snapshot_time).dt.total_seconds()
    return elapsed / seconds_per_year


def load_chain(path):
    raw = pd.read_parquet(path)
    renames = {"strike_price": "K", "strike": "K", "expiration": "expiry",
               "option_type": "right", "bid_price": "bid", "ask_price": "ask"}
    chain = raw.rename(columns={k: v for k, v in renames.items() if k in raw.columns})

    chain["expiry"] = pd.to_datetime(chain["expiry"], utc=True)
    chain["snapshot_utc"] = pd.to_datetime(chain["snapshot_utc"], utc=True)
    chain["T"] = year_fraction(chain["snapshot_utc"], chain["expiry"])
    chain["is_call"] = chain["right"].astype(str).str.upper().str[0] == "C"
    chain["mid"] = 0.5 * (chain["bid"] + chain["ask"])
    chain["spread"] = chain["ask"] - chain["bid"]
    return chain


def clean_chain(chain, min_bid=0.05, max_relative_spread=1.0, min_maturity=1.0 / 365.0):
    cleaned = chain.copy()
    cleaned["relative_spread"] = cleaned["spread"] / cleaned["mid"].replace(0.0, np.nan)

    has_market = (cleaned["bid"] > 0.0) & (cleaned["ask"] > cleaned["bid"])
    quotable = cleaned["bid"] >= min_bid
    tight_enough = cleaned["relative_spread"] <= max_relative_spread
    alive = cleaned["T"] >= min_maturity

    cleaned["rejected_reason"] = np.select(
        [~has_market, ~quotable, ~tight_enough, ~alive],
        ["no_market", "below_min_bid", "spread_too_wide", "expired"],
        default="",
    )
    return cleaned[cleaned["rejected_reason"] == ""].copy()


def pair_by_strike(expiry_chain):
    calls = expiry_chain[expiry_chain["is_call"]].set_index("K")
    puts = expiry_chain[~expiry_chain["is_call"]].set_index("K")
    shared = calls.index.intersection(puts.index)

    paired = pd.DataFrame({
        "K": shared,
        "call_mid": calls.loc[shared, "mid"].to_numpy(),
        "put_mid": puts.loc[shared, "mid"].to_numpy(),
        "call_spread": calls.loc[shared, "spread"].to_numpy(),
        "put_spread": puts.loc[shared, "spread"].to_numpy(),
    }).sort_values("K").reset_index(drop=True)
    paired["parity_difference"] = paired["call_mid"] - paired["put_mid"]
    return paired


def _weighted_line_fit(x, y, weights):
    total_weight = weights.sum()
    mean_x = np.sum(weights * x) / total_weight
    mean_y = np.sum(weights * y) / total_weight
    centred_x = x - mean_x
    centred_y = y - mean_y

    slope = np.sum(weights * centred_x * centred_y) / np.sum(weights * centred_x ** 2)
    intercept = mean_y - slope * mean_x

    residuals = y - (intercept + slope * x)
    weighted_residual = np.sum(weights * residuals ** 2)
    weighted_total = np.sum(weights * centred_y ** 2)
    r_squared = 1.0 - weighted_residual / weighted_total
    return slope, intercept, r_squared


def implied_forward(expiry_chain, moneyness_band=0.10, min_pairs=4):
    paired = pair_by_strike(expiry_chain)
    if len(paired) < min_pairs:
        return {"F": np.nan, "df": np.nan, "r_squared": np.nan, "n_pairs": len(paired)}

    strikes = paired["K"].to_numpy(float)
    differences = paired["parity_difference"].to_numpy(float)
    weights = 1.0 / (paired["call_spread"] + paired["put_spread"]).to_numpy(float) ** 2

    slope, intercept, _ = _weighted_line_fit(strikes, differences, weights)
    rough_discount = -slope
    rough_forward = intercept / rough_discount

    near_money = np.abs(strikes / rough_forward - 1.0) <= moneyness_band
    if near_money.sum() < min_pairs:
        near_money = np.ones_like(strikes, dtype=bool)

    slope, intercept, r_squared = _weighted_line_fit(
        strikes[near_money], differences[near_money], weights[near_money])
    discount_factor = -slope
    forward = intercept / discount_factor

    return {"F": forward, "df": discount_factor, "r_squared": r_squared,
            "n_pairs": int(near_money.sum())}


def forwards_by_expiry(chain, moneyness_band=0.10, min_pairs=4):
    records = []
    for expiry, expiry_chain in chain.groupby("expiry"):
        fit = implied_forward(expiry_chain, moneyness_band, min_pairs)
        fit["expiry"] = expiry
        fit["T"] = float(expiry_chain["T"].iloc[0])
        records.append(fit)
    return pd.DataFrame(records).sort_values("T").reset_index(drop=True)


def add_implied_vols(chain, forwards):
    merged = chain.merge(forwards[["expiry", "F", "df"]], on="expiry", how="inner")
    merged = merged[np.isfinite(merged["F"]) & np.isfinite(merged["df"])].copy()
    return implied_vol_chain(merged, price_col="mid")


def otm_only(chain):
    w = np.where(chain["is_call"], 1.0, -1.0)
    is_otm = w * (chain["F"] - chain["K"]) <= 0.0
    return chain[is_otm].copy()