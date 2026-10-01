# Time-Series Diagnostics Layer (P1.0.1)

This document provides a technical guide to the production-oriented time-series diagnostics layer implemented in the Smart Sales Forecasting System (`ml/analysis/time_series_diagnostics.py`).

---

## 1. Purpose

The objective of the diagnostics layer is to quantitatively evaluate demand time series **prior** to selecting classical statistical or autoregressive models (such as ARIMA, SARIMA, or seasonal decomposition). 

It answers ten fundamental questions:
1. Is the series stationary around a constant mean?
2. Is there a measurable linear or deterministic trend?
3. Is variance changing over time (heteroscedasticity)?
4. Does first differencing or seasonal differencing induce stationarity?
5. What autocorrelation exists across short and medium lags?
6. What direct lag effects exist after controlling for intermediate lags (PACF)?
7. What weekly (7-day) seasonal autocorrelation is present?
8. Are there indications of longer seasonal cycles (monthly or annual)?
9. Are there sufficient observations for classical time-series estimation?
10. Are there extreme observations or distributional skewness requiring variance-stabilizing transformations?

> [!NOTE]
> This module is strictly a diagnostic assessment tool. It does not train forecasting models, rank algorithms, or declare a "best model".

---

## 2. Input Contract & Canonical Continuity

The diagnostics layer accepts a canonical daily DataFrame conforming to the P0.2 time-series continuity contract:

- **Required Columns**:
  - `Date`: Daily calendar timestamps (normalized to midnight).
  - `Quantity`: Numeric demand observations.
- **Optional Columns**:
  - `Promotions`, `Holiday_Flag`, `Sales_Amount`, `Profit`.
- **Integrity Invariants**:
  - Missing calendar dates between `min(Date)` and `max(Date)` are rejected with an explicit `ValueError`.
  - Rows with `is_missing=True` are rejected with an explicit `ValueError`.
  - Zero-demand days (`Quantity == 0.0`) are valid observed demand and are preserved.
  - The input DataFrame is never mutated during diagnostic execution.

---

## 3. Diagnostic Tests & Statistical Methods

### 3.1 Basic Profile
Computes baseline parametric and non-parametric summary statistics:
- Observation count ($N$), start date, end date.
- Mean, median, standard deviation, minimum, maximum.
- Count of zero-demand days.
- Coefficient of Variation ($\text{CV} = \sigma / \mu$ for $\mu > 0$).

### 3.2 Trend Diagnostics
Estimates deterministic linear progression:
- OLS regression of $y_t$ on $t \in [0, N-1]$ yielding slope ($\beta_1$), intercept ($\beta_0$), and goodness-of-fit coefficient of determination ($R^2$).
- Half-split analysis: compares mean demand in the first half ($t < N/2$) vs. the second half ($t \ge N/2$), reporting percentage change without subjective labeling.

### 3.3 Stationarity Diagnostics (Augmented Dickey-Fuller Test)
Evaluates the null hypothesis ($H_0$) that a unit root is present in the autoregressive polynomial:

$$\Delta y_t = c + \gamma y_{t-1} + \sum_{j=1}^p \delta_j \Delta y_{t-j} + \epsilon_t$$

- **Implementation**: Primary execution via `statsmodels.tsa.stattools.adfuller` with Schwert (1989) maximum lag selection ($12 \times (N/100)^{1/4}$) and AIC lag optimization, with finite-sample MacKinnon response-surface fallback.
- **Test Statistic**: $t_{\gamma} = \hat{\gamma} / \text{SE}(\hat{\gamma})$.
- **Finite-Sample Critical Values**: MacKinnon (1994, 2010) regression surfaces for Case 2 (constant, no trend) evaluated at sample size $T^* = N - 1 - p$.
- **Factual Assessment**:
  - `stationary_at_5_percent`: $t_{\gamma} < \text{CV}_{5\%}$ ($p \le 0.05$), rejecting unit root.
  - `non_stationary_at_5_percent`: $t_{\gamma} \ge \text{CV}_{5\%}$ ($p > 0.05$), failing to reject unit root.
  - `inconclusive`: Sample size $N < 8$ or singular system.

### 3.4 Differencing Diagnostics
Runs ADF evaluations on three representations of the series:
1. Levels: $y_t$.
2. First differences: $\Delta y_t = y_t - y_{t-1}$.
3. Seasonal differences (lag 7): $\Delta_7 y_t = y_t - y_{t-7}$.
- **Differencing Signal**: Factual measurement of whether first differencing successfully converts a non-stationary level series into a stationary series (`first_difference_changed_stationarity = True`). Does not speculate on detrending without explicit differenced or detrended testing.

### 3.5 Autocorrelation Function (ACF) & Partial Autocorrelation (PACF)
- **ACF**: Sample autocorrelation coefficients for lags $k \in [1, 35]$:
  $$\hat{\rho}_k = \frac{\sum_{t=k+1}^N (y_t - \bar{y})(y_{t-k} - \bar{y})}{\sum_{t=1}^N (y_t - \bar{y})^2}$$
  Explicitly tracks weekly multiples: lags 1, 7, 14, 21, and 28.
- **PACF**: Measures direct correlation between $y_t$ and $y_{t-k}$ adjusting for intervening lags via OLS regressions.
  - **Sample Size Guard**: Safely restricts maximum evaluated lag to $\min(k_{\max}, \lfloor N/2 \rfloor - 1)$ to prevent over-parameterization on short sequences.

### 3.6 Weekly & Long-Cycle Seasonality
- **Weekly Seasonality**:
  - Tracks autocorrelations at lags 7, 14, 21, and 28.
  - Computes Wang-Smith-Hyndman seasonal strength metric ($F_s \in [0.0, 1.0]$) using moving average detrending and day-of-week decomposition:
    $$F_s = \max\left(0, 1 - \frac{\text{Var}(R)}{\text{Var}(S + R)}\right)$$
- **Longer Cycles**:
  - **Monthly Cycle**: Evaluated at lag 30 only if $N \ge 60$ (at least 2 full cycles).
  - **Annual Cycle**: Evaluated at lag 364 (52 exact weeks) only if $N \ge 730$ (at least 2 full annual cycles).
  - Flags candidate periodic structures only when sufficient data exists.

### 3.7 Rolling Statistics & Variance Stability
- Computes 7-day rolling standard deviations.
- Summarizes `mean_rolling_std`, `max_rolling_std`, `min_rolling_std`.
- Ratio $\text{Max} / \text{Min} > 3.0$ signals changing variance across time (heteroscedasticity).

### 3.8 Variance / Transformation Diagnostics
- Calculates skewness and excess kurtosis.
- **Factual Recommendation**:
  - `consider_log_transform`: Triggered when skewness $> 1.0$ and $\min(y) \ge 0$.
  - Indicates $\log(1 + y)$ must be used when zero-demand days are present.

### 3.9 Statistical Outlier Summary
- Computes IQR fences: $Q1 - 1.5 \times \text{IQR}$ and $Q3 + 1.5 \times \text{IQR}$.
- Calculates outlier count, outlier ratio, and extreme ratios relative to the median.
- Explicitly documented as statistical distribution diagnostics, not operational business anomalies.

### 3.10 Observation Sufficiency
- Verifies sample sizes against classical time-series thresholds:
  - Minimum 28 days for general baseline diagnostics.
  - Minimum 14 days for 7-day weekly seasonality.
  - Minimum 730 days for annual cycle analysis.

---

## 4. Reproducibility & CLI Execution

Execute the diagnostics pipeline on the canonical dataset:

```bash
python ml/analysis/time_series_diagnostics.py
```

### Generated Artifacts
1. **JSON Summary**:
   `ml/artifacts/time_series_diagnostics.json`
   Contains complete structured numerical results, critical values, and diagnostic conclusions. All floats are JSON-sanitized (no unquoted NaNs or Infinities).
2. **Visual Diagnostics (Optional)**:
   Stored in `ml/artifacts/time_series_diagnostics/`:
   - `original_and_rolling.png`: Daily demand with 7-day rolling mean and $\pm 1$ std band.
   - `acf_pacf.png`: Sample ACF and PACF stems with 95% Bartlett confidence bounds ($\pm 1.96 / \sqrt{N}$).
   - `differenced_series.png`: First differenced and lag-7 seasonal differenced series.

---

## 5. Limitations & Caveats

1. **Correlation vs. Causality**: Autocorrelations, trend regressions, and seasonal patterns measure statistical dependence; they do not establish causal business drivers.
2. **Structural Breaks**: The Augmented Dickey-Fuller test can have reduced power in the presence of sudden permanent shifts in baseline demand.
3. **Data Requirements**: High-confidence detection of annual cycles requires multiple complete calendar years ($N \ge 730$).
