# MATH840: Time Series Forecasting & Master Analytics Pipeline

**Course:** MATH840 — Advanced Time Series Forecasting  
**Term:** Fall 2026  
**Student:** Mykhailo Kapranov (`mkapranov@kse.org.ua`)  

---

## 1. Project Overview

This repository contains the complete analytical and forecasting pipeline for hourly electricity generation portfolio forecasting ($H = 48$ hours ahead) with calibrated **95% prediction intervals** ($\alpha = 0.05$).

The project implements and benchmarks across four model families:
1. **Classical Statistical:** Seasonal Naive benchmark, AutoARIMA, AutoETS, and Multi-Seasonal STL (MSTL) with per-component AutoETS.
2. **Deep Neural Networks:** LSTM, MLP, DeepAR, and TiDE (trained with and without exogenous weather covariates via `NeuralForecast`).
3. **Time Series Foundation Models:** Chronos-2 (Amazon), TimesFM 2.5 (Google Research), and TiRex (NX-AI xLSTM).
4. **Calibrated Ensemble:** An equal-weight combination of **Chronos-2 + LSTM (with weather) + Seasonal Naive**, calibrated with **distribution-free 95% conformal prediction intervals** derived from out-of-fold rolling validation errors.

---

## 2. Deliverables & Repository Structure

Only the core project artifacts are retained in the working directory:

| File | Type | Description |
|---|---|---|
| **`KAPRANOV_combo.ipynb`** | Jupyter Notebook | **Master Pipeline**: Contains full EDA, STL/MSTL diagnostics, validation runs of all models (with and without weather), combination experiments, forecast error correlation heatmaps, submission generation, and 6-panel residual diagnostics. |
| **`KAPRANOV_combo.ipynb - Colab.pdf`** | PDF Document | **Executed Notebook Export**: Complete compiled PDF showing all executed cells, interactive visualizations, scoreboard tables, and residual plots. |
| **`ultimate_time_series_analysis.ipynb`** | Jupyter Notebook | **In-Depth Analytics**: Exploratory analysis notebook featuring calendar interrogation, variance stabilization, Guerrero Box-Cox estimation, seasonal subseries, weather correlation matrices, KPSS stationarity tests, and differencing plans. |
| **`_project.ipynb`** | Jupyter Notebook | **Original Project Framework**: Baseline reference notebook defining the course scoring protocol, validation splits, and format assertion functions. |
| **`README.md`** | Markdown Document | Complete documentation on execution, environment configuration, model methodology, and explicit AI usage disclosure. |

*(Note: All historical intermediate drafts, temporary backups, and raw datasets are archived in `.backup/` and `/tmp/backup_knoppers/`).*

---

## 3. How to Run the Code

### 3.1 Data Requirements
Place the following two CSV files in the project root directory (downloadable from Moodle):
* `train.csv`: Historical electricity generation (`ds`, `y`) and 7 exogenous weather features (`cloud_cover`, `temperature`, `wind_speed`, `pressure`, `humidity`, `daylight`, `air_density`) across 33,480 hourly observations.
* `weather_forecast.csv`: 48-hour future weather forecast for the hidden test horizon.

### 3.2 Environment Setup & Dependencies
The code is designed to run in **Google Colab** (recommended with T4/V100 GPU) or a **local Linux/macOS Python 3.10–3.12** environment.

Run the following installation command in your terminal or in the first notebook cell:
```bash
# Clean potential package collisions and install required libraries
pip uninstall -y tirex 2>/dev/null
pip install -q statsmodels \
               statsforecast==2.1.1 \
               neuralforecast==3.3.0 \
               chronos-forecasting==2.3.2 \
               "timesfm[torch]==3.0.2" \
               tirex-ts==1.4.2 \
               coreforecast \
               utilsforecast
```

> **Note on TiRex:** The NX-AI time-series foundation model is packaged under `tirex-ts==1.4.2`. Do not install `tirex` (which is an unrelated information retrieval package). If switching packages in Colab, restart the session via **Runtime $\rightarrow$ Restart Session**.

### 3.3 Execution Workflow
Open [**`KAPRANOV_combo.ipynb`**](file:///home/stargazer/knoppers/KAPRANOV_combo.ipynb) and run cells sequentially from top to bottom:
1. **Sections 1–2:** Installs packages, sets global parameters ($M=24, H=48, \text{Level}=95\%$), and executes `interrogate_and_repair_calendar` to ensure a gap-free regular hourly grid.
2. **Sections 3–7:** Executes exploratory analysis: full-series time plot, Box-Cox variance stabilization, 24h & 12m seasonal subseries, exogenous weather correlation heatmaps, KPSS stationarity tests, and full-history STL & MSTL (Daily $m=24$ + Yearly $m=8760$).
3. **Sections 8–9:** Initializes scoring metrics (RMSE, MAPE %, Winkler-95, Coverage %), benchmark functions, ridge-stabilized weather regression, and wrappers for classic, neural, and foundation models.
4. **Sections 10–14:** Executes rolling-origin backtesting across 4 validation windows ($4 \times 48 = 192$ steps):
   - Models without weather (AutoARIMA, AutoETS, MSTL, MLP, DeepAR, LSTM, TiDE, Chronos-2, TimesFM, TiRex, Benchmark).
   - Models with weather (AutoARIMA, AutoETS, MSTL, MLP, DeepAR, LSTM, TiDE, Chronos-2).
   - Baseline / Bad Attempt: Multi-Seasonal STL with AutoETS fitted individually per component.
   - Multi-family combination without weather: `TimesFM 2.5 + Chronos-2 + MLP + DeepAR + AutoETS`.
   - Winning ensemble with weather: `Chronos-2 + LSTM + Benchmark`.
5. **Sections 15–16:** Displays the comprehensive scoreboard table and plots the pairwise forecast error correlation heatmap to inspect residual orthogonality across models.
6. **Sections 17–18:** Calls the winning ensemble on the **entire series** $y$, computes 95% conformal bounds ($\pm 280.45$), validates output integrity via `check_submission`, and exports `KAPRANOV_project.csv`.
7. **Section 19:** Generates the 6-panel residual diagnostics figure and prints summary statistics (mean bias, skewness, kurtosis, Kolmogorov-Smirnov, Ljung-Box test).

---

## 4. AI Usage & Academic Integrity Disclosure (`ai_usage`)

In accordance with academic integrity guidelines, full transparency is provided regarding the use of AI assistance during this project.

### 4.1 AI Tool Utilized
* **Model:** Antigravity AI Assistant (Google DeepMind)
* **Mode:** Pair-programming conversational agent

### 4.2 Breakdown of AI Assistance vs. Human Decision-Making

#### A. AI Contributions (Code Architecture & Scaffolding)
1. **Automated Notebook Modernization:** Refactored legacy manual parameter scripts into automated forecasting pipelines (`AutoARIMA`, `AutoETS`, and MSTL wrappers).
2. **Metric & Interval Formulation:** Fixed evaluation metrics to standard percentage MAPE, 95% Winkler score ($\alpha=0.05$), and implemented finite-sample out-of-fold Conformal Prediction (`conformal_radius`) to handle neural models lacking analytical variance.
3. **Weather Alignment Engine:** Built the ridge-stabilized regression helper `weather_regression` and aligned future weather covariates for neural models (`futr_exog_list`) and foundation models (`future_covariates`).
4. **Notebook Structuring:** Built the clean, title-only combo notebook [**`KAPRANOV_combo.ipynb`**](file:///home/stargazer/knoppers/KAPRANOV_combo.ipynb) combining exploratory analytics with rolling-origin benchmarking.
5. **Troubleshooting & Diagnostics:** Diagnosed package namespace collisions (`tirex-ts` vs. `tirex`), resolved Python kernel in-memory module caching, and built the publication-grade 6-panel residual diagnostic layout.

#### B. Human Decisions & Critical Oversight
1. **Model Selection & Combination:** Formulated the hypothesis that combining an autoregressive deep network (`LSTM` with weather), a zero-shot pre-trained foundation model (`Chronos-2`), and a non-parametric persistence model (`Seasonal Naive`) would minimize out-of-sample variance due to orthogonal error profiles.
2. **Decomposition Architecture:** Identified that the baseline "MSTL" approach should not simply be an average of two models, but rather an explicit decomposition where `AutoETS` is fitted individually to each decomposed component (trend, daily season, weekly season, and remainder).
3. **Rejection of Artificial Bias Correction:** After observing a mean residual error of $\bar{e} \approx -37.5$ in backtest residuals, critically evaluated and **rejected** adding a constant scalar shift to the final forecast. Recognized that electricity generation errors are diurnal (varying between night and day) rather than a flat drift, and that adding a scalar would distort near-zero nighttime generation.
4. **Multi-Scale Seasonality:** Formulated the requirement to evaluate multi-seasonal decomposition across the full dataset spanning nearly 4 years using Daily ($m=24$) and Annual/Yearly ($m=8760$) periods rather than short weekly windows.
5. **Log-Transformation Reasoning:** Investigated and verified that while classical linear Gaussian models require $\log(y)$ to stabilize variance, neural and foundation models must **not** be log-transformed because they rely on native physical scaling, robust batch scalers (`scaler_type="robust"`), and pre-trained zero-shot priors.

---

## 5. Summary of AI Interaction Turns

A total of **19 interaction turns** were conducted during the project lifecycle. A complete verbatim transcript is preserved in the project archive (`.backup/ai_usage.txt`). Key milestones include:

* **Turns 1–3:** Diagnostic analysis of MAPE scaling, Winkler score formulas, and mathematical feasibility of incorporating future weather forecasts.
* **Turns 4–5:** Implementation of 95% confidence intervals on foundation models (`Chronos-2`, `TimesFM 2.5`, `TiRex`) and universal `USE_WEATHER` parameter switching across validation folds.
* **Turns 6–7:** Optimization of backtest computation to prevent CPU exhaustion on long time-series fits; creation of the project submission pipeline and residual analysis blocks.
* **Turn 8:** Clarification of the Bates-Granger simple equal-weight combination principle ($1/M$) utilized in Lab 09.
* **Turns 9–10:** Creation of the unified master combo notebook ([`KAPRANOV_combo.ipynb`](file:///home/stargazer/knoppers/KAPRANOV_combo.ipynb)) with strictly title-only headers, followed by pairwise forecast error correlation heatmaps.
* **Turn 11:** Methodological inquiry regarding residual bias correction; established that constant bias addition should be avoided in cyclical energy time series.
* **Turn 12:** Audit of final submission cells in `KAPRANOV_lab09 (2).ipynb`, verifying that LSTM and Chronos-2 were trained on full data with weather, and catching accidental filename syntax (`_project_.csv`).
* **Turns 13–14:** Identification and resolution of PyPI package collision (`tirex-ts` vs. `tirex`) and Python in-memory `sys.modules` cache invalidation.
* **Turn 15:** Implementation of the component-wise MSTL decomposition with AutoETS fitted on each individual component.
* **Turn 16:** Implementation of ACF & PACF on the raw undifferenced series with seasonal multiples highlighted.
* **Turn 17:** Implementation of full-history STL and multi-seasonal MSTL decomposition with Daily ($m=24$) and Yearly ($m=8760$) seasonality across the ~4-year dataset.
* **Turn 18:** Methodological analysis of variance stabilization; establishing why foundation models and neural networks should not be log-transformed.
* **Turn 19:** Comprehensive analysis of how the code selects between `"own interval"` and `"conformal"` prediction intervals.

---

## 6. Final Results Summary

| Model / Pipeline | Validation RMSE (Mean) | MAPE (%) | 95% Winkler Score | 95% Empirical Coverage | Interval Method |
|---|---|---|---|---|---|
| AutoETS | 517.930 | 225.81% | 10,645.456 | 88.02% | Analytical ("own") |
| MSTL (Per-Component AutoETS) | 599.545 | 207.27% | 18,769.468 | 90.10% | Analytical ("own") |
| AutoARIMA (with weather) | 231.787 | 402.98% | 16,214.380 | 87.50% | Analytical ("own") |
| DeepAR (with weather) | 363.563 | 422.92% | 1,615.819 | 93.75% | Conformal |
| TiDE (with weather) | 252.897 | 321.72% | 1,302.197 | 92.71% | Conformal |
| LSTM (with weather) | 204.188 | 981.45% | 1,592.140 | 89.06% | Conformal |
| Chronos-2 (with weather) | 177.478 | 270.18% | 826.023 | 100.00% | Quantile ("own") |
| TimesFM 2.5 | 222.189 | 218.51% | 773.339 | 100.00% | Quantile ("own") |
| TiRex | 335.970 | 385.26% | 836.920 | 96.35% | Quantile ("own") |
| Benchmark (Seasonal Naive) | 196.525 | 667.07% | 1,063.742 | 89.58% | Random Walk ("own") |
| **Ensemble: Chronos-2 + LSTM + Benchmark** | **127.461** | **548.36%** | **596.832** | **93.75%** | **Conformal (Radius = ±280.45)** |

*The winning ensemble achieved the lowest mean RMSE across all 4 rolling validation windows (127.461) and the best interval quality (Winkler score of 596.832 with ~94% coverage), successfully balancing foundation zero-shot priors, weather-driven neural dynamics, and seasonal persistence.*
