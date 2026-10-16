"""Fast sanity tests: `python -m pytest -q`."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from freight import config  # noqa: E402
from freight.data import Cleaner, daily_market_index, flag_label_outliers  # noqa: E402
from freight.features import build_features  # noqa: E402


def _frame():
    return pd.DataFrame(
        {
            "equipment": ["Dry Van", "Reefer", "Dry Van", "Flatbed"],
            "weight": [-20000.0, np.nan, 50000.0, 30000.0],
            "market_index": [1.0, np.nan, 0.9, 1.1],
            "date": pd.to_datetime(["2025-03-31", "2025-03-31", "2025-04-15", "2025-12-16"]),
            "distance": [100.0, 500.0, 1000.0, 2000.0],
            "pickup_lat": 35.0, "pickup_lon": -90.0, "delivery_lat": 40.0, "delivery_lon": -80.0,
        }
    )


def test_cleaner_fixes_weight_and_market_index():
    df = _frame()
    daily = daily_market_index(df)
    out = Cleaner().fit(df).transform(df, daily)
    assert (out["weight"] > 0).all()
    assert out.loc[1, "weight_missing"] == 1
    assert out.loc[1, "market_index"] == pytest.approx(1.0)  # same-day mean
    assert out.loc[2, "weight_capped"] == 1


def test_quarter_end_ramp():
    df = Cleaner().fit(_frame()).transform(_frame(), daily_market_index(_frame()))
    f = build_features(df)
    assert f.loc[0, "quarter_end_ramp"] == pytest.approx(1.0)      # Mar 31
    assert f.loc[2, "quarter_end_ramp"] == 0                       # April
    assert f.loc[3, "quarter_end_ramp"] == pytest.approx(16 / 31)  # Dec 16


@pytest.mark.skipif(not config.TRAIN_PATH.exists(), reason="data not present")
def test_outlier_share_is_small():
    train = pd.read_csv(config.TRAIN_PATH)
    share = flag_label_outliers(train).mean()
    assert 0.005 < share < 0.03


@pytest.mark.skipif(not config.PREDICTIONS_PATH.exists(), reason="run the pipeline first")
def test_submission_format():
    sub = pd.read_csv(config.PREDICTIONS_PATH)
    template = pd.read_csv(config.TEMPLATE_PATH)
    assert list(sub.columns) == ["load_id", "predicted_rate"]
    assert sub["load_id"].tolist() == template["load_id"].tolist()
    assert (sub["predicted_rate"] > 0).all()
