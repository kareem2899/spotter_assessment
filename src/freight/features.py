"""Feature engineering.

Two groups of features, matching the two parts of the model:

* Load-level features for the gradient-boosted trees: distance (log), equipment,
  weight + flags, city coordinates, market index, day of week.
* Market-level features for the linear component:
    - log_market_index : the daily market level
    - trend_years      : time since 2025-01-01, in years. At equal market_index,
                         rates drift up ~6-7% per year. Trees cannot extrapolate
                         this into Nov-Dec; a linear term can.
    - quarter_end_ramp : 0 outside Mar/Jun/Sep/Dec; inside those months it ramps
                         from ~0 to 1 with the day of month. Rates rise through
                         the last month of each quarter and reset after.

quote_signal is deliberately NOT used. In training it tracks distance, equipment
and the quarter-end cycle (signs of being derived from the price), but in
validation.csv it is flat across all three and has a narrower spread. Adding it made
2 of 3 time-based folds worse, so the model would be relying on a feature that
behaves differently at prediction time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def build_features(df: pd.DataFrame, passthrough=()) -> pd.DataFrame:
    """Expects a frame that has already gone through `Cleaner.transform`.

    `passthrough` lists raw columns to copy in unchanged (used for ablations,
    e.g. testing what quote_signal would add).
    """
    f = pd.DataFrame(index=df.index)
    f["log_distance"] = np.log(df["distance"].astype(float))
    f["equipment_code"] = df["equipment"].map(config.EQUIPMENT_CODES).astype("category")
    f["weight"] = df["weight"].astype(float)
    f["weight_capped"] = df["weight_capped"].astype(int)
    f["weight_missing"] = df["weight_missing"].astype(int)
    for col in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        f[col] = df[col].astype(float)
    f["log_market_index"] = np.log(df["market_index"].astype(float))
    f["day_of_week"] = df["date"].dt.dayofweek

    date = df["date"]
    f["trend_years"] = (date - pd.Timestamp(config.TREND_ORIGIN)).dt.days / 365.0
    in_quarter_end = date.dt.month.isin(config.QUARTER_END_MONTHS)
    f["quarter_end_ramp"] = in_quarter_end * (date.dt.day / date.dt.days_in_month)
    f["day_of_year"] = date.dt.dayofyear
    for col in passthrough:
        f[col] = df[col].astype(float)
    return f


def lane_key(df: pd.DataFrame) -> pd.Series:
    return df["pickup"].astype(str) + " > " + df["delivery"].astype(str)


def attach_city_coordinates(df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Add pickup/delivery coordinates for frames that only have city names
    (the December chart inputs). Coordinates are looked up from the data."""
    coords = pd.concat(
        [
            reference[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
            reference[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1),
        ]
    ).drop_duplicates("city").set_index("city")
    out = df.copy()
    out["pickup_lat"] = out["pickup"].map(coords["lat"])
    out["pickup_lon"] = out["pickup"].map(coords["lon"])
    out["delivery_lat"] = out["delivery"].map(coords["lat"])
    out["delivery_lon"] = out["delivery"].map(coords["lon"])
    if out[["pickup_lat", "delivery_lat"]].isna().any().any():
        raise ValueError("Unknown city in input; cannot look up coordinates")
    return out
