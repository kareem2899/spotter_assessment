"""Central configuration: paths, constants and model hyper-parameters.

Everything that a reviewer might want to tweak lives here so the rest of the
code base contains no magic numbers.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"

PREDICTIONS_PATH = ROOT / "validation_predictions.csv"
MODEL_PATH = ROOT / "models" / "hybrid_rate_model.joblib"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"
EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.json"

# ---------------------------------------------------------------------------
# Data-cleaning constants (all derived from the EDA, see reports/eda_summary.json)
# ---------------------------------------------------------------------------
WEIGHT_CAP = 47_500.0            # weights are right-censored at this value
OUTLIER_LOG_RESIDUAL = 0.40      # |log(rate / robust expected rate)| above this = corrupted label
TREND_ORIGIN = "2025-01-01"      # t = 0 for the linear time trend
QUARTER_END_MONTHS = (3, 6, 9, 12)
EQUIPMENT_CODES = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}

# ---------------------------------------------------------------------------
# Validation design
# ---------------------------------------------------------------------------
# Main hold-out: train on Jan-Aug, test on Sep-Oct. This mirrors the real task,
# which is to predict the two months (Nov-Dec) that follow the labelled data.
HOLDOUT_START = "2025-09-01"
# Rolling-origin folds: (test_start, test_end_exclusive). Each fold trains on
# everything before test_start.
ROLLING_FOLDS = [
    ("2025-07-01", "2025-09-01"),
    ("2025-08-01", "2025-10-01"),
    ("2025-09-01", "2025-11-01"),
]
N_UNSEEN_CITIES = 8              # validation.csv contains 8 cities never seen in training
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Features used by the gradient-boosted trees (load-level / static structure).
TREE_FEATURES = [
    "log_distance",
    "equipment_code",
    "weight",
    "weight_capped",
    "weight_missing",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "log_market_index",
    "day_of_week",
]
# Features of the linear "market level" component. Linear so it can extrapolate
# the time trend into Nov-Dec, something trees cannot do.
LINEAR_FEATURES = ["log_market_index", "trend_years", "quarter_end_ramp"]

LGBM_PARAMS = dict(
    objective="regression",
    learning_rate=0.05,
    n_estimators=800,
    num_leaves=31,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    verbose=-1,
)
BACKFIT_ITERATIONS = 3           # alternate tree / linear / lane fits this many times
LANE_SHRINKAGE = 10              # pseudo-count for the lane random effect
