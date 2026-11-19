# Prediction of Solar Energy Potential Based on Weather and Location Data

![Project Header](assets/Project%20Header.jpg)

Predicting the power output of horizontal photovoltaic panels from weather, location, and
time — then serving those models through an interactive Streamlit application.

![Python](https://img.shields.io/badge/python-3.14-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4-189FDD)
![Streamlit](https://img.shields.io/badge/Streamlit-1.67-FF4B4B?logo=streamlit&logoColor=white)

---

## Overview

US electricity demand is projected to grow by roughly **30% by 2035**, while solar's share of
supply is projected to rise from about **3% to as much as 40%**. Meeting that target is less a
panel-technology problem than a siting and yield-forecasting problem: output at a given moment
depends on cloud cover, temperature, humidity, air mass, season, and hour of day.

This project models that relationship directly. Given weather and time-of-observation features,
it predicts panel output in Watts, and it ranks sites and conditions by expected yield so that
new capacity can be directed where the resource is strongest.

**Target:** `PolyPwr` — panel power output in Watts.

---

## What's in this repository

| Path | Purpose |
|---|---|
| `notebook.ipynb` | Full analysis: EDA → preprocessing → feature selection → modelling → tuning → persistence |
| `solar_potential/` | Application package (schema, data access, analytics, charts, views) |
| `main.py` | Streamlit entry point |
| `models/` | Persisted model bundles consumed by the app |
| `utils/` | Vendored data-science toolkit (preprocessing, evaluation, tuning, plotting) |
| `scripts/smoke_check.py` | Verification gate for the data, analytics, and scoring paths |
| `data/` | Source dataset and a five-row batch-scoring example |
| `docs/` | Design spec and implementation plan |

---

## Dataset

The data accompanies Pasion et al., *Machine Learning Modeling of Horizontal Photovoltaics Using
Weather and Location Data* (Energies, 2020).

- **21,045 observations**, 17 raw columns, no missing values or duplicate rows
- **12 northern-hemisphere sites** across the United States
- **December 2017 – February 2019**, sampled during daylight hours **10:00–15:00**
- Features: location, date, time, latitude, longitude, altitude, month, hour, season, humidity,
  ambient temperature, wind speed, visibility, pressure, and cloud ceiling

Source: [Kaggle — Northern Hemisphere Horizontal Photovoltaic](https://www.kaggle.com/datasets/saurabhshahane/northern-hemisphere-horizontal-photovoltaic)

---

## Approach

The notebook is organised into six stages:

1. **Exploratory analysis** — distribution, correlation, and association analysis, plus a
   location × hour and season × hour view of output.
2. **Preprocessing** — timestamp reconstruction, train/test split, encoding, outlier filtering,
   and scaling, with every fitted object learned from the training split only.
3. **Feature selection** — Spearman correlation, variance inflation factor, and SelectKBest
   (mutual information and F-regression).
4. **Modelling** — eleven regressors compared under one evaluation harness.
5. **Tuning** — randomized search with 5-fold cross-validation for the two strongest models.
6. **Persistence** — self-describing model bundles written to `models/`.

### Modelling decisions worth knowing

- **Six features were enough.** `AmbientTemp`, `Pressure`, `Cloud.Ceiling`, `Humidity`, `Month`,
  and `Hour` reproduce the accuracy of the original ten-feature set.
- **`Season` was dropped, `Month` kept.** Season is a deterministic recode of Month, so the
  one-hot indicators added redundancy, not signal. VIF is structurally blind to this
  (0/1 flags cannot linearly reconstruct a continuous month index), which is why the EDA
  crosstab is the evidence.
- **`Altitude` was dropped.** It correlates with station pressure at ρ = −0.97.
- **Site dummies were dropped.** They encode real fixed effects and rank mid-table, but a model
  built on them cannot score a site outside the training set — which is the actual use case.
  The accuracy cost of that choice is disclosed rather than hidden.
- **Trees are trained unscaled.** Min-max scaling is applied only to the scale-sensitive
  models (linear, KNN, MLP) so the comparison is fair; tree ensembles split on thresholds.
- **Artifacts are bundles, not bare estimators.** Each `.joblib` carries the model, the exact
  feature order, its metrics, a label, and the target name, so a consumer can rebuild a valid
  input row without reading the notebook.

---

## Results

Finalists evaluated with 5-fold, 3-repeat cross-validation on the training split, plus a
held-out 25% test split:

| Model | CV R² | CV MAE (W) | CV RMSE (W) | Test R² |
|---|---|---|---|---|
| Random Forest (tuned) | **0.6418 ± 0.0124** | **2.991** | 4.245 | **0.6542** |
| XGBoost (tuned) | 0.6359 ± 0.0119 | 3.017 | **4.280** | 0.6466 |

For context, from the 11-model screening pass (3-fold CV R²):

| Model | CV R² |
|---|---|
| Random Forest | 0.629 |
| XGBoost | 0.613 |
| Gradient Boosting | 0.606 |
| K-Nearest Neighbours | 0.577 |
| MLP | 0.546 |
| Linear / Ridge Regression | 0.427 |
| AdaBoost | 0.403 |
| Decision Tree | 0.310 |

Two things stand out: the tree ensembles win because the relationships are monotone but
non-linear, and tuning buys about **+0.01 R²** — the default ensembles were already near their
ceiling on six features.

---

## Quickstart

Requires **Python 3.14** and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies into .venv
uv sync
```

**Run the analysis notebook**

Open `notebook.ipynb` in VS Code or Jupyter and select the `.venv` kernel. Running it end to
end regenerates both files in `models/` (allow roughly 10 minutes; tuning dominates).

**Run the application**

```bash
uv run streamlit run main.py
```

The app opens with four tabs:

- **Predict** — pick a site, month, and hour to prefill that site's median weather, adjust the
  conditions, and read the expected output with a local sensitivity breakdown.
- **Batch** — upload a CSV of weather observations and score every row, with a downloadable result.
- **Insights** — model comparison, output by site / hour / season, feature importance, and
  correlation with the target.
- **About** — model card, data dictionary, and limitations.

**Verify the scoring paths**

```bash
uv run python scripts/smoke_check.py
```

---

## Project structure

```
.
├── main.py                     # Streamlit entry point
├── notebook.ipynb              # Analysis, modelling, and model persistence
├── solar_potential/
│   ├── schema.py               # Feature contract and data dictionary
│   ├── config.py               # Paths
│   ├── data.py                 # Loading, site presets, batch preparation
│   ├── analytics.py            # Artifact loading, scoring, sensitivity
│   ├── ui.py                   # Altair chart builders
│   └── views/                  # predict · batch · insights · about
├── models/                     # tuned_rf_model.joblib · tuned_xgb_model.joblib
├── data/                       # Source dataset and batch example
├── utils/                      # Vendored data-science toolkit
├── scripts/smoke_check.py
├── docs/superpowers/           # Design spec and implementation plan
└── .streamlit/config.toml      # Theme
```

---

## Limitations

- **Random split, not temporal.** The test split is random over rows, so the same site and
  timestamp can appear in both train and test. Reported R² is therefore optimistic for a
  genuinely new day. A grouped or time-based split is the correct next experiment.
- **Daylight only.** Observations cover 10:00–15:00; the model says nothing about night or
  early-morning output.
- **Twelve US sites, fourteen months.** Nothing here transfers to the southern hemisphere
  without re-validation.
- **Location is not a model input.** Two sites with identical weather receive the same
  prediction, even if their panels or wiring differ.
- **Cloud ceiling is capped at 722**, so at the top of its range the feature is effectively
  binary (clear or not).
- **MAPE is unstable near zero.** With output as low as 0.26 W, percentage error is dominated
  by low-output rows. MAE and RMSE are the metrics to quote.

---

## References

- Pasion C, Wagner T, Koschnick C, Schuldt S, Williams J, Hallinan K.
  *Machine Learning Modeling of Horizontal Photovoltaics Using Weather and Location Data.*
  Energies. 2020; 13(10):2570. [doi:10.3390/en13102570](https://doi.org/10.3390/en13102570)
- Dataset: [Northern Hemisphere Horizontal Photovoltaic](https://www.kaggle.com/datasets/saurabhshahane/northern-hemisphere-horizontal-photovoltaic)
- US Department of Energy, *Solar Futures Study* — demand growth and solar share projections.

---

## Author

**Muhammad Cikal Merdeka** — Data Analyst / Data Scientist

[GitHub](https://github.com/mcikalmerdeka) · [LinkedIn](https://www.linkedin.com/in/mcikalmerdeka)
