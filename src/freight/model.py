"""The hybrid rate model.

    log(rate) = tree(load features)                              # LightGBM
              + b0 + b1*log(market_index) + b2*trend + b3*ramp    # linear market level
              + lane_effect[pickup > delivery]                    # shrunken lane premium

Why this shape:

* log target: freight errors scale with load size, so predicting log(rate) treats
  a 3% miss the same on a 100-mile and a 3,000-mile load, and keeps
  predictions strictly positive.
* Trees handle the non-linear structure (short hauls cost much more per mile,
  equipment x distance interactions, weight, city effects).
* The linear part carries the time trend. Trees predict a constant beyond the last
  date they saw, so a pure tree model cannot follow a trend into Nov-Dec.
* Lane effect: after the other terms, some lanes are consistently ~1% above or
  below. Each lane's mean residual is shrunk towards 0 with a pseudo-count, so rare
  lanes barely move, and unseen lanes (incl. the 8 new cities) get 0.

The three parts are fitted by backfitting: fit trees on y minus the other parts,
then the linear part on what remains, then the lane effects, and repeat.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config
from .data import Cleaner, daily_market_index
from .features import build_features, lane_key


class HybridRateModel:
    def __init__(
        self,
        tree_features=None,
        linear_features=None,
        lgbm_params=None,
        iterations: int = config.BACKFIT_ITERATIONS,
        lane_shrinkage: float | None = config.LANE_SHRINKAGE,
    ):
        self.tree_features = list(tree_features or config.TREE_FEATURES)
        self.linear_features = list(linear_features or config.LINEAR_FEATURES)
        self.lgbm_params = dict(lgbm_params or config.LGBM_PARAMS)
        self.iterations = iterations
        self.lane_shrinkage = lane_shrinkage

    # -- helpers ---------------------------------------------------------
    def _linear_design(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.ones(len(X)), X[self.linear_features].to_numpy(float)])

    def _lane_part(self, lanes: pd.Series | None, n: int) -> np.ndarray:
        if self.lane_shrinkage is None or lanes is None:
            return np.zeros(n)
        return lanes.map(self.lane_effects_).fillna(0.0).to_numpy()

    # -- API -------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y_log: np.ndarray, lanes: pd.Series | None = None):
        y = np.asarray(y_log, dtype=float)
        linear = np.zeros(len(y))
        lane = np.zeros(len(y))
        self.lane_effects_ = pd.Series(dtype=float)
        for _ in range(self.iterations):
            self.trees_ = lgb.LGBMRegressor(**self.lgbm_params)
            self.trees_.fit(X[self.tree_features], y - linear - lane)
            tree_pred = self.trees_.predict(X[self.tree_features])

            D = self._linear_design(X)
            self.coef_ = np.linalg.lstsq(D, y - tree_pred - lane, rcond=None)[0]
            linear = D @ self.coef_

            if self.lane_shrinkage is not None and lanes is not None:
                resid = pd.Series(y - tree_pred - linear, index=lanes.index)
                stats = resid.groupby(lanes.to_numpy()).agg(["sum", "count"])
                self.lane_effects_ = stats["sum"] / (stats["count"] + self.lane_shrinkage)
                lane = self._lane_part(lanes, len(y))
        return self

    def predict_log(self, X: pd.DataFrame, lanes: pd.Series | None = None) -> np.ndarray:
        return (
            self.trees_.predict(X[self.tree_features])
            + self._linear_design(X) @ self.coef_
            + self._lane_part(lanes, len(X))
        )

    def linear_coefficients(self) -> dict:
        return dict(zip(["intercept"] + self.linear_features, map(float, self.coef_)))


@dataclass
class RatePipeline:
    """Cleaning + features + model in one object, so training and inference
    go through exactly the same code path."""

    model: HybridRateModel = field(default_factory=HybridRateModel)
    passthrough: tuple = ()  # extra raw columns for ablations (empty in the final model)

    def _features(self, df: pd.DataFrame):
        prepared = self.cleaner_.transform(df, self.daily_mi_)
        return prepared, build_features(prepared, self.passthrough)

    def fit(self, train_clean: pd.DataFrame, daily_mi: pd.Series) -> "RatePipeline":
        """`train_clean` = labelled rows with outliers already removed."""
        self.daily_mi_ = daily_mi
        self.cleaner_ = Cleaner().fit(train_clean)
        prepared, X = self._features(train_clean)
        self.model.fit(X, np.log(prepared["posted_rate"].to_numpy()), lane_key(prepared))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        prepared, X = self._features(df)
        return np.exp(self.model.predict_log(X, lane_key(prepared)))

    def save(self, path=config.MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path=config.MODEL_PATH) -> "RatePipeline":
        return joblib.load(path)


def make_pipeline(train: pd.DataFrame, other_feature_frames=(), **model_kwargs) -> RatePipeline:
    """Convenience: build the daily market index from all available feature rows
    and fit a pipeline on `train`."""
    daily_mi = daily_market_index(train, *other_feature_frames)
    return RatePipeline(HybridRateModel(**model_kwargs)).fit(train, daily_mi)
