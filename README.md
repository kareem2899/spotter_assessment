# Freight Rate Prediction

Predicts the posted rate of truck loads for Nov–Dec 2025 from labelled loads dated Jan–Oct 2025.

**Result (time-based hold-out: train Jan–Aug, test Sep–Oct):** MAPE **1.74%**, MAE **$39.5**.
For comparison, a plain LightGBM scores 2.43% and a $/mile baseline 3.74%. Mean MAPE over 3 rolling time folds: **1.71%**.

The full write-up is in [`reports/report.pdf`](reports/report.pdf), also available as `.docx`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

python run_pipeline.py all        # EDA -> validation -> final training -> predictions (~3 min)
python score.py --predictions validation_predictions.csv \
                --december-predictions data/december_chart_inputs.csv
python -m pytest -q               # sanity tests
```

Individual steps: `python run_pipeline.py eda | evaluate | train | predict`.

## Outputs

| File | What it is |
|---|---|
| `validation_predictions.csv` | Submission: `load_id,predicted_rate` for all 12,000 validation loads |
| `data/december_chart_inputs.csv` | December inputs with `predicted_rate` filled (original 7 columns) |
| `scorer_results/candidate_december.png` | Chart produced by `score.py` |
| `reports/report.pdf` / `.docx` | Report: data findings, data quality, split & validation, model, results |
| `reports/metrics.json` | All validation numbers (hold-out, rolling folds, unseen cities, random-split contrast) |
| `reports/eda_summary.json` | Data-quality and EDA statistics |
| `reports/figures/` | All report figures |
| `models/hybrid_rate_model.joblib` | Trained final pipeline (created by `run_pipeline.py train`; not committed) |

## Project layout

```
run_pipeline.py            CLI entry point (eda / evaluate / train / predict / all)
score.py                   provided scorer (unchanged)
src/freight/
  config.py                paths, constants, hyper-parameters, validation folds
  data.py                  loading, cleaning (Cleaner), corrupted-label detection
  features.py              feature engineering
  model.py                 HybridRateModel + RatePipeline (cleaning + features + model)
  evaluation.py            metrics, time-based splits, baselines & ablations
  eda.py, plotting.py      EDA summary and figures
tests/test_pipeline.py     unit + output-format tests
data/                      input CSVs (as provided)
```

## Approach in brief

1. **Split by time, not at random.** The validation set covers the two months after the
   labelled data, so the main hold-out trains on Jan–Aug and tests on Sep–Oct. Three
   rolling-origin folds and an unseen-city test confirm stability. A random split is
   reported only to show how optimistic it is.
2. **Cleaning:**
   - 677 corrupted rates (1.4%), detected with a robust distance/equipment curve and removed from training only.
   - Negative weights → `abs()`.
   - Weights capped at 47,500 lb → kept, with a flag.
   - Missing weight → per-equipment median.
   - Missing market_index → same-day mean.
3. **Model.** `log(rate)` = LightGBM(load features) + linear(log market_index, time trend,
   quarter-end ramp) + shrunken lane effect. The linear part lets the model extrapolate
   the market trend into Nov–Dec, which trees alone cannot.
4. **`quote_signal` is excluded.** In training it tracks distance, equipment and the
   quarter-end cycle; in validation it doesn't. Adding it made 2 of 3 time folds worse.
5. **December chart.** The December inputs have no market_index, so the pipeline uses the
   mean market_index of the validation loads on each December date.

## Requirements

Python 3.10+. Package versions are in `requirements.txt` (pandas, numpy, matplotlib,
lightgbm, joblib, pytest).
