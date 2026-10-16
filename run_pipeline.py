"""End-to-end pipeline.

    python run_pipeline.py all        # eda -> evaluate -> train -> predict
    python run_pipeline.py eda        # summary stats + EDA figures
    python run_pipeline.py evaluate   # time-based validation of all candidate models
    python run_pipeline.py train      # fit the final model on all clean labelled data
    python run_pipeline.py predict    # write validation_predictions.csv + December rates
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from freight import config, eda  # noqa: E402
from freight.data import daily_market_index, flag_label_outliers, load_csv, load_train, load_validation  # noqa: E402
from freight.evaluation import (candidate_models, metrics, random_split_contrast, rolling_folds,  # noqa: E402
                                time_holdout, unseen_city_test)
from freight.features import attach_city_coordinates  # noqa: E402
from freight.model import RatePipeline, make_pipeline  # noqa: E402
from freight.plotting import MUTED, NEUTRAL, PRIMARY, SECONDARY, apply_style, save  # noqa: E402

FINAL = "Hybrid (final)"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
def step_eda() -> None:
    log("EDA ...")
    s = eda.run()
    log(f"  label outliers: {s['label_outliers']['count']} ({s['label_outliers']['share_%']}%), "
        f"new cities in validation: {len(s['cities']['new_in_validation'])}")


# ---------------------------------------------------------------------------
def step_evaluate() -> None:
    apply_style()
    train, val = load_train(), load_validation()
    outliers = flag_label_outliers(train)
    daily_mi = daily_market_index(train, val)
    factories = candidate_models()
    results: dict = {"outliers_removed_from_training": int(outliers.sum())}

    log("Main hold-out: train Jan-Aug, test Sep-Oct ...")
    holdout, test, preds = time_holdout(train, outliers, daily_mi, factories)
    results["holdout_sep_oct"] = holdout.round(4).to_dict(orient="records")
    print(holdout[["model", "MAE", "MAPE_%", "RMSE", "Bias_%", "MAPE_%_incl_outliers"]].round(3).to_string(index=False))

    log("Rolling-origin folds ...")
    rolling = rolling_folds(train, outliers, daily_mi, factories)
    results["rolling_folds"] = rolling.round(4).to_dict(orient="records")
    pivot = rolling.pivot_table(index="model", columns="fold", values="MAPE_%", sort=False)
    pivot["mean"] = pivot.mean(axis=1)
    results["rolling_mape_table"] = pivot.round(3).reset_index().to_dict(orient="records")
    print(pivot.round(3).to_string())

    log("Unseen-city stress test ...")
    results["unseen_cities"] = unseen_city_test(train, outliers, daily_mi, factories[FINAL])
    print(json.dumps(results["unseen_cities"], indent=1))

    log("Random-split contrast (for the report only) ...")
    results["random_split_80_20"] = {
        FINAL: random_split_contrast(train, outliers, daily_mi, factories[FINAL]),
        "Plain LightGBM (all raw features)": random_split_contrast(
            train, outliers, daily_mi, factories["Plain LightGBM (all raw features)"]),
    }
    print(json.dumps(results["random_split_80_20"], indent=1))

    # coefficients of the final model trained on Jan-Aug, for interpretation
    m = factories[FINAL]().fit(train[(train["date"] < config.HOLDOUT_START) & ~outliers], daily_mi)
    results["linear_coefficients_jan_aug"] = m.model.linear_coefficients()

    config.METRICS_PATH.write_text(json.dumps(results, indent=2))
    _evaluation_figures(holdout, pivot, test, preds)
    log(f"Wrote {config.METRICS_PATH.relative_to(config.ROOT)}")


def _evaluation_figures(holdout, pivot, test, preds) -> None:
    # 1. model comparison
    fig, ax = plt.subplots(figsize=(8, 3.4))
    h = holdout.iloc[::-1]
    colors = [PRIMARY if m == FINAL else NEUTRAL for m in h["model"]]
    ax.barh(h["model"], h["MAPE_%"], color=colors, height=0.6)
    for y, v in enumerate(h["MAPE_%"]):
        ax.text(v + 0.04, y, f"{v:.2f}%", va="center", fontsize=9, color=MUTED)
    ax.set_xlabel("MAPE on Sep-Oct hold-out (%), trained on Jan-Aug")
    ax.set_title("Model comparison on the time-based hold-out")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    save(fig, config.FIGURES_DIR / "eval_model_comparison.png")

    # 2. predicted vs actual
    clean = ~test["is_outlier"].to_numpy()
    a, p = test["posted_rate"].to_numpy()[clean], preds[FINAL][clean]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(a, p, s=4, alpha=0.25, color=PRIMARY, linewidths=0)
    lim = [a.min() * 0.9, a.max() * 1.1]
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("Actual rate ($)"), ax.set_ylabel("Predicted rate ($)")
    ax.set_title("Final model, Sep-Oct hold-out")
    ax.grid(axis="x")
    save(fig, config.FIGURES_DIR / "eval_pred_vs_actual.png")

    # 3. daily error over time: final vs plain LightGBM
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for name, color in ((FINAL, PRIMARY), ("Plain LightGBM (all raw features)", SECONDARY)):
        pct = 100 * (preds[name][clean] / a - 1)
        daily = pd.Series(pct, index=test["date"].to_numpy()[clean]).groupby(level=0).mean()
        ax.plot(daily.index, daily.values, color=color, lw=1.6, label=name)
    ax.axhline(0, color=NEUTRAL, lw=0.8)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_ylabel("Mean daily error (%)")
    ax.set_title("Hold-out bias by day (trained on Jan-Aug): hybrid stays closer to zero")
    ax.legend(loc="lower left")
    save(fig, config.FIGURES_DIR / "eval_daily_bias.png")


# ---------------------------------------------------------------------------
def step_train() -> None:
    train, val = load_train(), load_validation()
    outliers = flag_label_outliers(train)
    log(f"Training final model on {int((~outliers).sum()):,} clean rows (Jan-Oct) ...")
    pipe = make_pipeline(train[~outliers], other_feature_frames=[val])
    pipe.save()
    log(f"  linear coefficients: {json.dumps({k: round(v, 4) for k, v in pipe.model.linear_coefficients().items()})}")
    log(f"Saved {config.MODEL_PATH.relative_to(config.ROOT)}")


# ---------------------------------------------------------------------------
def step_predict() -> None:
    pipe = RatePipeline.load()
    train, val = load_train(), load_validation()

    # 1) validation predictions in template order
    template = pd.read_csv(config.TEMPLATE_PATH)
    pred = pd.Series(pipe.predict(val), index=val["load_id"])
    out = template[["load_id"]].copy()
    out["predicted_rate"] = out["load_id"].map(pred).round(2)
    assert out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all()
    assert len(out) == 12_000 and out["load_id"].is_unique
    out.to_csv(config.PREDICTIONS_PATH, index=False)
    log(f"Wrote {config.PREDICTIONS_PATH.name}: {len(out):,} rows, "
        f"median ${out['predicted_rate'].median():,.0f}")

    # 2) December chart inputs: only the date changes. The file has no coordinates,
    # market_index or quote_signal; coordinates are looked up by city and
    # market_index comes from the daily mean of validation.csv for that date.
    dec = pd.read_csv(config.DECEMBER_PATH)
    columns = list(dec.columns)
    rows = attach_city_coordinates(dec.assign(date=pd.to_datetime(dec["date"])), pd.concat([train, val]))
    rows["market_index"] = np.nan
    rows["weight"] = rows["weight"].astype(float)
    dec["predicted_rate"] = np.round(pipe.predict(rows), 2)
    dec[columns].to_csv(config.DECEMBER_PATH, index=False)
    detail = dec.assign(market_index_used=rows["date"].map(pipe.daily_mi_).round(4).to_numpy())
    detail.to_csv(config.REPORTS_DIR / "december_predictions_detail.csv", index=False)
    log(f"Filled {config.DECEMBER_PATH.relative_to(config.ROOT)}: "
        f"${dec['predicted_rate'].min():,.2f} - ${dec['predicted_rate'].max():,.2f}")


STEPS = {"eda": step_eda, "evaluate": step_evaluate, "train": step_train, "predict": step_predict}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=[*STEPS, "all"], nargs="?", default="all")
    args = parser.parse_args()
    for name in (STEPS if args.step == "all" else [args.step]):
        STEPS[name]()


if __name__ == "__main__":
    main()
