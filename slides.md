---
marp: true
theme: default
paginate: true
header: "MATH840 • Time Series Forecasting Project"
footer: "Mykhailo Kapranov (mkapranov@kse.org.ua) • Fall 2026"
style: |
  section {
    font-family: 'Inter', sans-serif;
    background-color: #0f172a;
    color: #f8fafc;
  }
  h1, h2 {
    color: #38bdf8;
  }
  table {
    font-size: 14px;
    color: #e2e8f0;
  }
  th {
    background-color: #1e293b;
    color: #38bdf8;
  }
---

# MATH840: Time Series Forecasting Project
### Hourly Electricity Generation: Modeling, Ensembling & Calibrated Conformal Intervals

**Student:** Mykhailo Kapranov (`mkapranov@kse.org.ua`)  
**Kyiv School of Economics**  
*Fall 2026*

---

## 1. Problem Overview & Task

* **Objective:** Forecast $H = 48$ hours of portfolio electricity generation with calibrated **95% prediction intervals**.
* **Dataset:** 33,480 hourly observations (~3.82 years) with 7 exogenous weather features (`temperature`, `cloud_cover`, `wind_speed`, `pressure`, `humidity`, `daylight`, `air_density`).
* **Evaluation Criteria:**
  1. **RMSE:** Out-of-sample forecast accuracy.
  2. **MAPE (%):** Relative percentage deviation.
  3. **Winkler Score (95%):** Penalizes interval width and boundary breaches ($\alpha = 0.05$).
  4. **Empirical Coverage (%):** Target $\ge 95\%$.

---

## 2. Exploratory Data Analysis & Variance

* **Time Plot:** Clear diurnal cycles (24h) with large midday solar spikes and lower night baselines.
* **Variance Stabilization:**
  * Monthly swing vs. level regression yields $r = 0.183$ ($p = 0.224$), showing mild heteroscedasticity.
  * Classical linear models (`AutoARIMA`, `AutoETS`) benefit from `log=True` to enforce positive generation ($\hat{y} > 0$).
  * Neural and Foundation models operate directly on physical scales via robust batch scalers (`scaler_type="robust"`).

---

## 3. Seasonality & Exogenous Weather Drivers

* **Diurnal & Weekly Cycles:**
  * Daytime generation peaks at 13:00–14:00; drops to base load at 01:00–04:00.
  * Sunday load systematically drops ~15% relative to mid-week industrial profiles.
* **Weather Correlation:**
  * `daylight` ($r = +0.54$) and `temperature` ($r = +0.48$) are dominant drivers.
  * `cloud_cover` exerts a strong nonlinear dampening effect during midday hours.

---

## 4. Stationarity & Decomposition

* **Autocorrelation (ACF):**
  * Persistent 24h peaks ($r_{24} = 0.82, r_{48} = 0.76, r_{168} = 0.76$).
  * KPSS tests confirm level non-stationarity ($p < 0.01$).
  * Seasonal differencing ($D=1$, lag=24) achieves stationarity.
* **Full-History MSTL Decomposition (Daily 24h + Yearly 8760h):**
  * Trend Strength: $F_T = 0.942$.
  * Daily Season Strength: $F_S(24h) = 0.887$.
  * Yearly Season Strength: $F_S(8760h) = 0.651$.
  * Remainder diagnostics confirm white-noise residuals without diurnal bias.

---

## 5. Rolling Validation Scoreboard

Evaluated across 4 rolling-origin backtest windows ($4 \times 48 = 192$ steps):

| Model / Pipeline | Mean RMSE | MAPE (%) | 95% Winkler | Coverage | Interval |
|---|---|---|---|---|---|
| **Ensemble (Chronos-2 + LSTM + snaive)** | **127.461** | **548.36%** | **596.832** | **93.75%** | **Conformal** |
| Chronos-2 (with weather) | 177.478 | 270.18% | 826.023 | 100.00% | Quantile |
| Benchmark (Seasonal Naive) | 196.525 | 667.07% | 1063.742 | 89.58% | Random Walk |
| LSTM (with weather) | 204.188 | 981.45% | 1592.140 | 89.06% | Conformal |
| TimesFM 2.5 | 222.189 | 218.51% | 773.339 | 100.00% | Quantile |
| AutoARIMA (with weather) | 231.787 | 402.98% | 16214.380 | 87.50% | Analytical |
| TiDE (with weather) | 252.897 | 321.72% | 1302.197 | 92.71% | Conformal |
| MLP (with weather) | 262.047 | 654.70% | 1099.949 | 95.31% | Conformal |
| AutoETS (univariate) | 517.930 | 225.81% | 10645.456 | 88.02% | Analytical |
| MSTL (Per-Component AutoETS) | 599.545 | 207.27% | 18769.468 | 90.10% | Analytical |

---

## 6. Why Ensembling Wins: Error Orthogonality

* **Bates-Granger Principle:** Simple equal-weight averaging ($1/M$) consistently outperforms complex estimated weights on unseen folds.
* **Pairwise Error Correlations:**
  * `Chronos-2` vs `LSTM (weather)`: $r = 0.28$.
  * `Chronos-2` vs `snaive`: $r = 0.18$.
  * `LSTM (weather)` vs `snaive`: $r = 0.35$.
* **Mechanism:** The zero-shot foundation prior smoothes extreme spikes; the recurrent LSTM reacts sharply to incoming weather fronts; Seasonal Naive anchors the exact diurnal cycle.

---

## 7. Uncertainty Quantification: Conformal Intervals

* **Classical Failure Mode:**
  * Classical log-variance propagation $\exp(\hat{y}_{\log} \pm 1.96 \sigma_{48})$ explodes at $h=48$, producing upper bounds $>40,000$ MW (Winkler $>16,000$).
  * Neural networks provide point forecasts with no parametric bounds.
* **Conformal Calibration:**
  * Calibrated on out-of-fold backtest absolute errors:
    $$r_{95} = \text{Quantile}_{0.95}(|y - \hat{y}|) = \pm 280.45 \text{ MW}$$
  * Guarantees 93.75% empirical coverage and drops the Winkler score to **596.83**.

---

## 8. Final Submission & Residual Diagnostics

* **Submission File:** `KAPRANOV_project.csv` verified against all schema checks (48 monotonic hourly timestamps, lower bound $\ge 0$, lower $\le$ forecast $\le$ upper).
* **Residual Diagnostics:**
  * Mean error $\bar{e} \approx -37.5$ MW; rejected flat scalar bias addition to avoid distorting zero-generation night baselines.
  * Homoscedastic residual scatter across fitted values ($\text{slope} \approx 0$).
  * Ljung-Box test confirms absence of substantial serial correlation.

---

## 9. Conclusions

1. **Triangulating Paradigms:** Combining foundation zero-shot priors, weather-driven neural networks, and non-parametric persistence yields the lowest variance.
2. **Preserving Physical Scaling:** Pre-trained foundation models should operate on unwarped physical scales.
3. **Distribution-Free Calibration:** Conformal prediction intervals solve the interval bloat of classical models and give reliable coverage for complex ensembles.
