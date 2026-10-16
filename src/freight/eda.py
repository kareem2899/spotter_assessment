"""Exploratory data analysis: summary statistics + figures for the report.

Run via `python run_pipeline.py eda`. Writes reports/eda_summary.json and
reports/figures/eda_*.png.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .data import load_train, load_validation, robust_log_residual
from .plotting import (ALERT, EQUIPMENT_COLORS, MUTED, NEUTRAL, PRIMARY, SECONDARY,
                       apply_style, save)


def summarize(train: pd.DataFrame, val: pd.DataFrame, resid: pd.Series) -> dict:
    outlier = resid.abs() > config.OUTLIER_LOG_RESIDUAL
    rpm = train["posted_rate"] / train["distance"]
    train_cities = set(train["pickup"]) | set(train["delivery"])
    val_cities = set(val["pickup"]) | set(val["delivery"])
    new_cities = sorted(val_cities - train_cities)
    clamped = lambda d: int(((d[["pickup_lat", "delivery_lat"]] == 25.5).any(axis=1)
                             | (d[["pickup_lon", "delivery_lon"]] == -69.5).any(axis=1)).sum())
    within_day_std = train.groupby("date")["market_index"].std().mean()
    return {
        "rows": {"train": len(train), "validation": len(val)},
        "date_range": {
            "train": [str(train["date"].min().date()), str(train["date"].max().date())],
            "validation": [str(val["date"].min().date()), str(val["date"].max().date())],
        },
        "missing": {
            "train": train.isna().sum()[lambda s: s > 0].to_dict(),
            "validation": val.isna().sum()[lambda s: s > 0].to_dict(),
        },
        "duplicates": {"load_id": int(train["load_id"].duplicated().sum()),
                       "full_rows": int(train.drop(columns="load_id").duplicated().sum())},
        "negative_weight": {"train": int((train["weight"] < 0).sum()), "validation": int((val["weight"] < 0).sum())},
        "weight_at_cap_47500": {"train": int((train["weight"].abs() >= config.WEIGHT_CAP).sum()),
                                "validation": int((val["weight"].abs() >= config.WEIGHT_CAP).sum())},
        "label_outliers": {
            "count": int(outlier.sum()),
            "share_%": round(100 * outlier.mean(), 2),
            "rows_with_abs_log_residual_between_0.25_and_0.6": int(resid.abs().between(0.25, 0.6).sum()),
        },
        "clamped_coordinate_rows": {"train": clamped(train), "validation": clamped(val)},
        "cities": {"train": len(train_cities), "validation": len(val_cities),
                   "new_in_validation": new_cities,
                   "validation_rows_touching_new_cities": int(
                       (val["pickup"].isin(new_cities) | val["delivery"].isin(new_cities)).sum())},
        "lanes": {"train": int((train["pickup"] + ">" + train["delivery"]).nunique())},
        "rate_per_mile_median_by_equipment": rpm[~outlier].groupby(train["equipment"]).median().round(3).to_dict(),
        "corr_rate_distance": round(float(train["posted_rate"].corr(train["distance"])), 3),
        "market_index": {
            "within_day_std": round(float(within_day_std), 4),
            "overall_std": round(float(train["market_index"].std()), 4),
            "train_range": [round(float(train["market_index"].min()), 3), round(float(train["market_index"].max()), 3)],
            "validation_range": [round(float(val["market_index"].min()), 3), round(float(val["market_index"].max()), 3)],
        },
        "quote_signal": {
            "train_std": round(float(train["quote_signal"].std()), 3),
            "validation_std": round(float(val["quote_signal"].std()), 3),
            "train_mean_by_equipment": train.groupby("equipment")["quote_signal"].mean().round(3).to_dict(),
            "validation_mean_by_equipment": val.groupby("equipment")["quote_signal"].mean().round(3).to_dict(),
        },
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_rate_per_mile(train, outlier):
    clean = train[~outlier]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sample = clean.sample(6000, random_state=0)
    for eq, color in EQUIPMENT_COLORS.items():
        s = sample[sample["equipment"] == eq]
        ax.scatter(s["distance"], s["posted_rate"] / s["distance"], s=6, alpha=0.35, color=color,
                   linewidths=0, label=eq)
    for eq, color in EQUIPMENT_COLORS.items():
        s = clean[clean["equipment"] == eq]
        bins = pd.qcut(s["distance"], 25)
        med = (s["posted_rate"] / s["distance"]).groupby(bins, observed=True).median()
        ax.plot(s["distance"].groupby(bins, observed=True).median(), med, color=color, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("Distance (miles, log scale)")
    ax.set_ylabel("Rate per mile ($)")
    ax.set_title("Short hauls cost far more per mile; Reefer > Flatbed > Dry Van")
    ax.legend(markerscale=3, loc="upper right")
    ax.set_ylim(1.4, 4.2)
    return fig


def fig_outlier_gap(resid):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.linspace(-2.2, 2.2, 177)
    ax.hist(resid.clip(-2.2, 2.2), bins=bins, color=PRIMARY, log=True)
    for x in (-config.OUTLIER_LOG_RESIDUAL, config.OUTLIER_LOG_RESIDUAL):
        ax.axvline(x, color=ALERT, lw=1.2, ls="--")
    ax.set_xlabel("log(posted rate / robust expected rate for that distance & equipment)")
    ax.set_ylabel("Loads (log scale)")
    ax.set_title("Corrupted labels are separated from normal loads by an empty gap")
    ax.text(config.OUTLIER_LOG_RESIDUAL + 0.05, ax.get_ylim()[1] * 0.3, "threshold ±0.40", color=ALERT, fontsize=9)
    return fig


def fig_market_over_time(train, val, resid, outlier):
    clean = train[~outlier].assign(resid=resid[~outlier])
    daily = clean.groupby("date").agg(level=("resid", "mean"), mi=("market_index", "mean"),
                                      qs=("quote_signal", "mean"))
    vdaily = val.groupby("date").agg(mi=("market_index", "mean"), qs=("quote_signal", "mean"))
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(daily.index, 100 * (np.exp(daily["level"]) - 1), color=PRIMARY, lw=1.6)
    axes[0].axhline(0, color=NEUTRAL, lw=0.8)
    axes[0].set_ylabel("% vs typical rate")
    axes[0].set_title("Daily price level (after removing distance & equipment)")
    axes[1].plot(daily.index, daily["mi"], color=PRIMARY, lw=1.6, label="train")
    axes[1].plot(vdaily.index, vdaily["mi"], color=SECONDARY, lw=1.6, label="validation (Nov-Dec)")
    axes[1].set_ylabel("market_index")
    axes[1].set_title("Daily mean market_index", fontsize=10.5)
    axes[1].legend(loc="upper right")
    axes[2].plot(daily.index, daily["qs"], color=PRIMARY, lw=1.6)
    axes[2].plot(vdaily.index, vdaily["qs"], color=SECONDARY, lw=1.6)
    axes[2].set_ylabel("quote_signal")
    axes[2].set_title("Daily mean quote_signal: quarter-end spikes in train, flat in validation", fontsize=10.5)
    for ax in axes:
        for m in ("2025-03-01", "2025-06-01", "2025-09-01", "2025-12-01"):
            ax.axvspan(pd.Timestamp(m), pd.Timestamp(m) + pd.offsets.MonthEnd(0), color="#f1f4f5", zorder=0)
    fig.tight_layout()
    return fig


def fig_quote_signal_shift(train, val):
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    edges = np.quantile(train["distance"], np.linspace(0, 1, 9))
    edges[0], edges[-1] = 0, 1e6
    for frame, name, color in ((train, "train (Jan-Oct)", PRIMARY), (val, "validation (Nov-Dec)", SECONDARY)):
        g = frame.groupby(pd.cut(frame["distance"], edges), observed=True)["quote_signal"].mean()
        mids = frame.groupby(pd.cut(frame["distance"], edges), observed=True)["distance"].median()
        ax.plot(mids, g, marker="o", ms=5, color=color, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Distance (miles, log scale)")
    ax.set_ylabel("Mean quote_signal")
    ax.set_title("quote_signal tracks distance in train but not in validation")
    ax.legend()
    return fig


def fig_weight(train):
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.hist(train["weight"].dropna(), bins=120, color=PRIMARY)
    ax.set_xlabel("weight (lb) as given")
    ax.set_ylabel("Loads")
    ax.set_title("Weight: negative values (sign errors) and a spike at the 47,500 lb cap")
    return fig


def run() -> dict:
    apply_style()
    train, val = load_train(), load_validation()
    resid = robust_log_residual(train)
    outlier = resid.abs() > config.OUTLIER_LOG_RESIDUAL
    summary = summarize(train, val, resid)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.EDA_SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))

    figs = {
        "eda_rate_per_mile.png": fig_rate_per_mile(train, outlier),
        "eda_outlier_gap.png": fig_outlier_gap(resid),
        "eda_market_over_time.png": fig_market_over_time(train, val, resid, outlier),
        "eda_quote_signal_shift.png": fig_quote_signal_shift(train, val),
        "eda_weight.png": fig_weight(train),
    }
    for name, fig in figs.items():
        save(fig, config.FIGURES_DIR / name)
    return summary
