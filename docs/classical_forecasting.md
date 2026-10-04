# Classical Forecasting Models: Exponential Smoothing (P1.0.2a)

This document provides a technical guide to the classical statistical forecasting layer implemented in the Smart Sales Forecasting System (`ml/models/classical_forecasters.py`) and its evaluation via the formal rolling-origin framework (`ml/evaluation/rolling_evaluator.py`).

---

## 1. Executive Purpose & Context

> [!IMPORTANT]
> **Candidate Status**: Classical models are candidate forecasting models until validated against the established rolling-origin benchmark. No model is declared superior prior to empirical evaluation across multiple horizons.

The goal of Phase P1.0.2a is to introduce a robust, interpretable classical time-series baseline layer based on exponential smoothing methods (ETS) to complement the system's feature-based machine learning models (Random Forest, HistGradientBoosting, Linear Regression).

Classical exponential smoothing methods offer several key advantages:
- **Direct Historical Forecasting**: They model level, trend, and seasonal components directly from historical demand without requiring feature engineering or external exogenous drivers.
- **Dynamic Parameter Optimization**: Smoothing parameters ($\alpha, \beta, \gamma, \phi$) are fitted via maximum likelihood / numerical optimization on history available prior to each origin.
- **Strong Weekly Periodicity Handling**: They natively capture the 7-day cyclical demand pattern quantitatively identified in Phase P1.0.1 diagnostics ($F_s = 0.8248$).

---

## 2. Model Formulations

### 2.1 Holt Linear Trend Model (`HoltAdditiveModel`)

The Holt method extends simple exponential smoothing to accommodate series with a local deterministic linear trend.

For historical observations $y_t$:

$$\begin{aligned}
\text{Level:}\quad & l_t = \alpha y_t + (1 - \alpha)(l_{t-1} + b_{t-1}) \\
\text{Trend:}\quad & b_t = \beta (l_t - l_{t-1}) + (1 - \beta) b_{t-1} \\
\text{Forecast:}\quad & \hat{y}_{t+h} = \max\left(0.0, l_t + h \cdot b_t\right)
\end{aligned}$$

Where:
- $\alpha \in (0, 1)$ is the level smoothing parameter.
- $\beta \in (0, 1)$ is the trend smoothing parameter.
- $h$ is the forecast horizon step.
- Non-negative clipping enforces demand physical constraints.

### 2.2 Holt-Winters Additive Weekly Model (`HoltWintersAdditiveModel`)

The Holt-Winters additive model incorporates a cyclical seasonal component alongside level and trend.

For a weekly seasonal cycle of length $m = 7$:

$$\begin{aligned}
\text{Level:}\quad & l_t = \alpha (y_t - s_{t-m}) + (1 - \alpha)(l_{t-1} + b_{t-1}) \\
\text{Trend:}\quad & b_t = \beta (l_t - l_{t-1}) + (1 - \beta) b_{t-1} \\
\text{Seasonal:}\quad & s_t = \gamma (y_t - l_{t-1} - b_{t-1}) + (1 - \gamma) s_{t-m} \\
\text{Forecast:}\quad & \hat{y}_{t+h} = \max\left(0.0, l_t + h \cdot b_t + s_{t+h - m(k+1)}\right)
\end{aligned}$$

Where $\gamma \in (0, 1)$ is the seasonal smoothing parameter, and $k = \lfloor (h - 1) / m \rfloor$.

### 2.3 Damped Holt-Winters Additive Model (`DampedHoltWintersModel`)

Linear trend projections can overestimate growth over extended horizons (e.g., 30 and 90 days). The damped trend variant introduces a damping parameter $\phi \in (0, 1)$ that attenuates trend growth over time:

$$\begin{aligned}
\text{Level:}\quad & l_t = \alpha (y_t - s_{t-m}) + (1 - \alpha)(l_{t-1} + \phi b_{t-1}) \\
\text{Trend:}\quad & b_t = \beta (l_t - l_{t-1}) + (1 - \beta) \phi b_{t-1} \\
\text{Seasonal:}\quad & s_t = \gamma (y_t - l_{t-1} - \phi b_{t-1}) + (1 - \gamma) s_{t-m} \\
\text{Forecast:}\quad & \hat{y}_{t+h} = \max\left(0.0, l_t + \sum_{i=1}^h \phi^i b_t + s_{t+h - m(k+1)}\right)
\end{aligned}$$

---

## 3. Design Decisions & Rationale

1. **Why Additive Seasonality**:
   Multiplicative seasonality divides by past level values ($y_t / l_t$), which creates numerical instability or undefined divisions when demand approaches zero or on zero-demand days. Additive seasonality operates on differences, remaining fully robust to zero-demand observations.
2. **Why Seasonal Period $m = 7$**:
   Phase P1.0.1 time-series diagnostics confirmed significant weekly autocorrelation ($\hat{\rho}_7 = 0.8850$, $\hat{\rho}_{14} = 0.8625$, $\hat{\rho}_{21} = 0.8481$, $\hat{\rho}_{28} = 0.8348$) and high Wang-Smith-Hyndman 7-day seasonal strength ($F_s = 0.8248$). A 7-day period aligns exactly with the retail sales calendar.
3. **Non-Negative Clipping**:
   Demand forecasts are constrained by physical reality: sales volume cannot be negative. All raw model outputs are clipped using $\max(0.0, \hat{y})$.
4. **Estimated Initialization**:
   Model state variables ($l_0, b_0, s_0$) are fitted via `initialization_method="estimated"` through statsmodels maximum likelihood estimation.

---

## 4. Evaluation Methodology

### 4.1 Rolling-Origin Framework
All classical models are evaluated using the single production evaluation framework: [`RollingEvaluator`](file:///c:/Users/prasa/Prasad%20Shetty/Project/Smart%20Sales%20Forecasting%20System/ml/evaluation/rolling_evaluator.py):
- **Canonical Dataset**: `data/processed_daily_forecasting_features.csv` (2,894 continuous daily observations).
- **Holdout Period**: 2025-01-01 to 2025-12-31 (365 test days).
- **Horizons**: 7-day, 30-day, and 90-day multi-step blocks.
- **Origins**:
  - 7-day: 53 rolling origins.
  - 30-day: 13 rolling origins.
  - 90-day: 5 rolling origins.

### 4.2 Leakage Prevention Invariants
1. **Strict Origin Isolation**: At each origin date $T$, the classical model is refitted **strictly** on observations where $\text{Date} < T$.
2. **Zero Forward Leakage**: Future actual demand observations after $T$ are never exposed to the model during fitting.
3. **Adversarial Verification**: Modifying future actuals after $T$ produces zero change in predictions at or prior to $T$.

### 4.3 Benchmark Comparison
Every classical model is evaluated alongside the exact same deterministic baseline:
- **Seasonal Naive**: Predicts the value observed 7 days prior ($F_{t+h} = y_{t+h-7}$), recursively fed forward for horizons $> 7$.
- Relative metrics are reported as $\Delta = \text{Metric}_{\text{Model}} - \text{Metric}_{\text{Baseline}}$. Negative $\Delta\text{WAPE}$ indicates superior performance over baseline.

---

## 5. Metric Definitions

- **WAPE (Weighted Absolute Percentage Error)**:
  $$\text{WAPE} = \frac{\sum |y_t - \hat{y}_t|}{\sum y_t}$$
  Scale-independent and unaffected by individual zero-demand days (unlike MAPE).
- **MAE (Mean Absolute Error)**:
  $$\text{MAE} = \frac{1}{N} \sum |y_t - \hat{y}_t|$$
  Measures average magnitude of forecast errors in sales units.
- **RMSE (Root Mean Squared Error)**:
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum (y_t - \hat{y}_t)^2}$$
  Penalizes larger outliers more heavily than MAE.
- **Bias (Mean Forecast Error)**:
  $$\text{Bias} = \frac{1}{N} \sum (y_t - \hat{y}_t)$$
  Positive bias indicates systematic under-prediction ($\text{Actual} > \text{Forecast}$); negative bias indicates systematic over-prediction ($\text{Forecast} > \text{Actual}$).

---

## 6. Execution & Artifacts

### 6.1 Running the Classical Evaluation Runner

Execute the evaluation script:

```bash
python -m ml.evaluation.evaluate_classical_models
```

### 6.2 Output Artifacts

Artifacts are persisted in a dedicated directory to prevent overwriting existing production model evaluations:

- `ml/artifacts/classical_model_evaluation/classical_model_evaluation.csv`: Summary metrics, origin distributions, and baseline deltas by horizon.
- `ml/artifacts/classical_model_evaluation/classical_model_predictions.csv`: Granular daily predictions and errors for all models across all origins.
- `ml/artifacts/classical_model_evaluation/classical_model_origin_metrics.csv`: Origin-by-origin metrics and bias distributions.
