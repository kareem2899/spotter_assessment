"""Loading and cleaning.

Data-quality issues found during EDA and how each one is handled:

1. Negative weights (~0.6% of rows, e.g. -23,937 lb). Their rates behave exactly
   like loads with the positive weight, so they are sign errors -> abs().
2. Weights right-censored at 47,500 lb (~2.5%). Kept, plus a `weight_capped` flag.
3. Missing weight (~0.6%). Imputed with the per-equipment median of the training
   data, plus a `weight_missing` flag.
4. Missing market_index (~0.8%). market_index is a *daily* market level (within a
   day its std is 0.025 vs 0.17 overall), so it is imputed with the mean
   market_index of all loads on the same date.
5. Corrupted labels (~1.4% of training rows): rates that are 0.2-0.5x or 2-7x what
   the same distance/equipment normally pays. A robust per-equipment curve in
   log-space leaves a clean gap: nothing lies between |log residual| 0.25 and 0.6,
   so everything above 0.40 is flagged and removed from *training only*.
6. Clamped coordinates (lat 25.5 / lon -69.5) for a few cities. Coordinates are
   consistent per city, so they still work as a city identifier; distance (which is
   clean) carries the geometry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_csv(path) -> pd.DataFrame:
    """Read a raw CSV and parse the date column."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_train() -> pd.DataFrame:
    return load_csv(config.TRAIN_PATH)


def load_validation() -> pd.DataFrame:
    return load_csv(config.VALIDATION_PATH)


def daily_market_index(*frames: pd.DataFrame) -> pd.Series:
    """Mean market_index per date across every load we have features for.

    Uses only *features* (never labels), so including validation.csv rows is not
    leakage. It gives a market level for every date in Jan-Dec, which is also how
    the December chart rows (which have no market_index column) get a value.
    """
    stacked = pd.concat([f[["date", "market_index"]] for f in frames], ignore_index=True)
    return stacked.groupby("date")["market_index"].mean()


class Cleaner:
    """Fit-on-train / apply-anywhere cleaning so no statistic leaks from test data."""

    def fit(self, train: pd.DataFrame) -> "Cleaner":
        weight = train["weight"].abs()
        self.weight_median_ = weight.groupby(train["equipment"]).median().to_dict()
        self.weight_global_median_ = float(weight.median())
        return self

    def transform(self, df: pd.DataFrame, daily_mi: pd.Series) -> pd.DataFrame:
        out = df.copy()
        # 1-3. weight
        out["weight_missing"] = out["weight"].isna().astype(int)
        out["weight"] = out["weight"].abs()
        fill = out["equipment"].map(self.weight_median_).fillna(self.weight_global_median_)
        out["weight"] = out["weight"].fillna(fill)
        out["weight_capped"] = (out["weight"] >= config.WEIGHT_CAP).astype(int)
        # 4. market_index
        if "market_index" not in out:
            out["market_index"] = np.nan
        day_level = out["date"].map(daily_mi)
        out["market_index"] = out["market_index"].fillna(day_level)
        if out["market_index"].isna().any():
            # A date with no observed market data at all: use the nearest date.
            nearest = daily_mi.reindex(out["date"], method="nearest").to_numpy()
            out["market_index"] = out["market_index"].fillna(pd.Series(nearest, index=out.index))
        return out


def robust_log_residual(df: pd.DataFrame) -> pd.Series:
    """log(rate) minus a robust cubic-in-log(distance) curve, fitted per equipment.

    Iteratively re-weighted: points beyond 4 robust standard deviations are excluded
    from the next fit, so corrupted labels cannot drag the curve.
    """
    resid = pd.Series(np.nan, index=df.index)
    log_rate = np.log(df["posted_rate"])
    log_dist = np.log(df["distance"])
    for _, idx in df.groupby("equipment").groups.items():
        x = log_dist.loc[idx].to_numpy()
        y = log_rate.loc[idx].to_numpy()
        X = np.column_stack([np.ones_like(x), x, x**2, x**3])
        keep = np.ones(len(y), dtype=bool)
        for _ in range(5):
            beta = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
            r = y - X @ beta
            mad = np.median(np.abs(r[keep] - np.median(r[keep])))
            keep = np.abs(r) < 4 * 1.4826 * mad
        resid.loc[idx] = r
    return resid


def flag_label_outliers(train: pd.DataFrame) -> pd.Series:
    """Boolean mask of corrupted posted_rate values (training data only)."""
    return robust_log_residual(train).abs() > config.OUTLIER_LOG_RESIDUAL
