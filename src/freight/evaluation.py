"""Validation: how the development data is split and how models are compared.

Split design (the most important decision in this project)
----------------------------------------------------------
validation.csv covers Nov-Dec 2025; the labelled data covers Jan-Oct 2025. The task
is therefore *forecasting the next two months*, so every evaluation here trains
on the past and tests on a later period:

1. Main hold-out: train Jan-Aug, test Sep-Oct (two unseen months, like Nov-Dec).
2. Rolling-origin folds (expanding window): test Jul-Aug, Aug-Sep, Sep-Oct, each
   trained only on earlier data. Shows the result is stable, not one lucky window.
3. Unseen-city stress test: 8 random cities are removed from training and the
   model is scored on Sep-Oct loads touching them. This mirrors the 8 cities that
   appear in validation.csv but never in training (~12% of rows).
4. For contrast, a random 80/20 split is also reported. It mixes future loads
   into training, so it looks better than it should and is NOT used for model
   choice.

Corrupted labels are removed from training folds. Test metrics are reported on
clean rows (primary) and on all rows (secondary), so the cleaning cannot flatter
the numbers.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config
from .data import Cleaner, daily_market_index
from .model import HybridRateModel, RatePipeline


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(actual, predicted) -> dict:
    a = np.asarray(actual, float)
    p = np.asarray(predicted, float)
    err = p - a
    return {
        "MAE": float(np.mean(np.abs(err))),
        "MAPE_%": float(100 * np.mean(np.abs(err) / a)),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "Bias_%": float(100 * np.mean(err / a)),
        "n": int(len(a)),
    }


# ---------------------------------------------------------------------------
# Comparison models (same fit/predict interface as RatePipeline)
# ---------------------------------------------------------------------------
class RatePerMileBaseline:
    """Median $/mile per equipment x distance band, times distance."""

    bands = [0, 150, 300, 500, 800, 1200, 1800, 10_000]

    def fit(self, train, daily_mi):
        band = pd.cut(train["distance"], self.bands)
        rpm = train["posted_rate"] / train["distance"]
        self.table_ = rpm.groupby([train["equipment"], band], observed=True).median()
        return self

    def predict(self, df):
        band = pd.cut(df["distance"], self.bands)
        keys = pd.MultiIndex.from_arrays([df["equipment"], band])
        return self.table_.reindex(keys).to_numpy() * df["distance"].to_numpy()


class PlainLightGBM(RatePipeline):
    """A single LightGBM on all raw features, including quote_signal and day of year.
    This is the 'obvious' model; it's kept here to show what the hybrid design adds."""

    def __init__(self):
        super().__init__(passthrough=("quote_signal",))

    def fit(self, train_clean, daily_mi):
        self.daily_mi_ = daily_mi
        self.cleaner_ = Cleaner().fit(train_clean)
        prepared, X = self._features(train_clean)
        self.cols_ = config.TREE_FEATURES + ["quote_signal", "day_of_year"]
        self.m_ = lgb.LGBMRegressor(**config.LGBM_PARAMS).fit(X[self.cols_], np.log(prepared["posted_rate"]))
        return self

    def predict(self, df):
        _, X = self._features(df)
        return np.exp(self.m_.predict(X[self.cols_]))


def hybrid(passthrough=(), **kw):
    return RatePipeline(HybridRateModel(**kw), passthrough=passthrough)


def candidate_models() -> dict:
    """Name -> factory. The first entry is the final model; the rest are ablations
    (each removes or adds one ingredient) and simpler reference models."""
    return {
        "Hybrid (final)": lambda: hybrid(),
        "Hybrid + quote_signal": lambda: hybrid(
            passthrough=("quote_signal",), tree_features=config.TREE_FEATURES + ["quote_signal"]),
        "Hybrid, no lane effect": lambda: hybrid(lane_shrinkage=None),
        "Hybrid, no quarter-end ramp": lambda: hybrid(linear_features=["log_market_index", "trend_years"]),
        "Plain LightGBM (all raw features)": lambda: PlainLightGBM(),
        "Baseline: $/mile by equipment x distance": lambda: RatePerMileBaseline(),
    }


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------
def _score(model_factory, train, test, daily_mi, test_outlier_mask):
    model = model_factory().fit(train, daily_mi)
    pred = model.predict(test)
    clean = ~test_outlier_mask.to_numpy()
    return (
        metrics(test["posted_rate"].to_numpy()[clean], pred[clean]),
        metrics(test["posted_rate"].to_numpy(), pred),
        pred,
    )


def time_holdout(df, outliers, daily_mi, factories, start=config.HOLDOUT_START):
    train_mask = (df["date"] < start) & ~outliers
    test_mask = df["date"] >= start
    test = df[test_mask]
    rows, predictions = [], {}
    for name, factory in factories.items():
        clean_m, all_m, pred = _score(factory, df[train_mask], test, daily_mi, outliers[test_mask])
        rows.append({"model": name, **clean_m, "MAPE_%_incl_outliers": all_m["MAPE_%"]})
        predictions[name] = pred
    return pd.DataFrame(rows), test.assign(is_outlier=outliers[test_mask].to_numpy()), predictions


def rolling_folds(df, outliers, daily_mi, factories, folds=config.ROLLING_FOLDS):
    rows = []
    for start, end in folds:
        train_mask = (df["date"] < start) & ~outliers
        test_mask = (df["date"] >= start) & (df["date"] < end)
        for name, factory in factories.items():
            clean_m, _, _ = _score(factory, df[train_mask], df[test_mask], daily_mi, outliers[test_mask])
            rows.append({"fold": f"{start[:7]} to {pd.Timestamp(end) - pd.Timedelta(days=1):%Y-%m}",
                         "model": name, **clean_m})
    return pd.DataFrame(rows)


def unseen_city_test(df, outliers, daily_mi, factory, n_cities=config.N_UNSEEN_CITIES,
                     seed=config.RANDOM_STATE, start=config.HOLDOUT_START):
    rng = np.random.default_rng(seed)
    cities = sorted(set(df["pickup"]) | set(df["delivery"]))
    held = set(rng.choice(cities, size=n_cities, replace=False))
    touches = df["pickup"].isin(held) | df["delivery"].isin(held)
    train_mask = (df["date"] < start) & ~outliers & ~touches
    test_new = (df["date"] >= start) & touches & ~outliers
    test_seen = (df["date"] >= start) & ~touches & ~outliers
    model = factory().fit(df[train_mask], daily_mi)
    return {
        "held_out_cities": sorted(held),
        "loads_touching_unseen_cities": metrics(df.loc[test_new, "posted_rate"], model.predict(df[test_new])),
        "loads_between_seen_cities": metrics(df.loc[test_seen, "posted_rate"], model.predict(df[test_seen])),
    }


def random_split_contrast(df, outliers, daily_mi, factory, seed=config.RANDOM_STATE):
    """Same model, random 80/20 split instead of time split (for the report only)."""
    rng = np.random.default_rng(seed)
    is_test = rng.random(len(df)) < 0.2
    train = df[~is_test & ~outliers.to_numpy()]
    test_mask = is_test & ~outliers.to_numpy()
    model = factory().fit(train, daily_mi)
    return metrics(df.loc[test_mask, "posted_rate"], model.predict(df[test_mask]))
