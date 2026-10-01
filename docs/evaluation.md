# Forecasting Model Evaluation Framework (P0.3)

This document provides a comprehensive technical guide to the formal rolling-origin forecasting evaluation framework implemented in the Smart Sales Forecasting System (`ml/evaluation/rolling_evaluator.py`).

---

## 1. Motivation: Rolling-Origin vs. Static Evaluation

Traditional single train/validation/test splits suffer from two major flaws in production forecasting:
1. **Target Leakage / Unrealistic Assumptions**: In static row-wise testing, lag features (such as `quantity_lag_1` or `quantity_lag_7`) for day $t+k$ are computed using ground-truth observations from days $t+1 \dots t+k-1$. In real-world operations, future actuals are never available during a multi-day forecast horizon.
2. **Horizon Instability**: Model performance can vary widely depending on the chosen origin date (e.g., peak holiday periods vs. off-season troughs). A single split produces high variance in error metrics and masks temporal instability.

The **Rolling-Origin Evaluation Framework** simulates real-world operations by repeatedly forecasting multi-step blocks $[O_k, O_k + H)$ from successive origin points $O_k$, strictly using only data available prior to each origin date.

---

## 2. Recursive Multi-Step Forecasting Protocol

For any forecast origin $O$ and horizon $H$:
1. **Strict Historical Boundary**: Only observations strictly before $O$ (`Date < O`) form the initial historical series.
2. **Recursive Autoregressive Steps**:
   - For day $t = O$: All lag and rolling features are derived from genuine historical observations.
   - For day $t = O + 1 \dots O + H - 1$: Recent lag values (e.g., lag 1, rolling window statistics) are derived from the model's own prior predictions.
   - Predictions are constrained to non-negative demand: $\hat{y} = \max(0.0, \hat{y}_{\text{raw}})$.
3. **Delayed Ground Truth Reveal**: True target observations for $[O, O + H)$ are revealed **only after** the entire forecast block is generated and evaluated.
4. **Origin Advance**: The ground truth is incorporated into history, the origin advances, and the process repeats.

---

## 3. Seasonal Naive Deterministic Baseline

To rigorously evaluate whether machine learning models add genuine business value, the framework evaluates every horizon against a deterministic **Seasonal Naive Baseline** (`SeasonalNaiveModel`):

$$\hat{y}_{t} = \max(0.0, y_{t - s}) \quad \text{where } s = 7 \text{ days}$$

- For $t \le 7$: Forecasts repeat the observation from 7 days prior.
- For $t > 7$: Forecasts recursively repeat the earlier predictions from step $t - 7$.
- If an ML model cannot beat the Seasonal Naive baseline on WAPE/MAE/RMSE, it does not warrant deployment for that horizon.

---

## 4. Evaluation Metrics & Conventions

### 4.1 Weighted Absolute Percentage Error (WAPE)
The primary business accuracy metric across all horizons:

$$\text{WAPE} = \frac{\sum_{i=1}^N |y_i - \hat{y}_i|}{\sum_{i=1}^N |y_i|}$$

Unlike standard MAPE, WAPE does not divide by zero on zero-demand days and properly weights high-volume days over low-volume days.

### 4.2 Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

### 4.3 Directional Forecast Bias
Bias measures whether the model systematically over-predicts or under-predicts demand:

$$\text{Bias} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)$$

- **Positive Bias ($> 0$)**: Actual demand exceeded prediction $\implies$ **under-prediction** (risk of stockouts).
- **Negative Bias ($< 0$)**: Prediction exceeded actual demand $\implies$ **over-prediction** (risk of excess inventory and holding costs).

### 4.4 Pooled vs. Origin-Level Metrics
- **Pooled Metrics** (`MAE`, `RMSE`, `MAPE`, `WAPE`, `Bias`): Calculated globally across all $N$ prediction-actual pairs in the holdout period.
- **Origin-Level Metrics** (`mean_origin_WAPE`, `median_origin_WAPE`, `std_origin_WAPE`, `min_origin_WAPE`, `max_origin_WAPE`): Computed per forecast origin block. Standard deviation and range quantify forecast volatility and risk across changing seasonal regimes.
- **Baseline Deltas** (`WAPE_delta_vs_baseline`, `MAE_delta_vs_baseline`, `RMSE_delta_vs_baseline`):
  $$\Delta_{\text{WAPE}} = \text{WAPE}_{\text{model}} - \text{WAPE}_{\text{baseline}}$$
  Negative deltas indicate superiority over the baseline.

---

## 5. Calendar Continuity & Data Quality Guard

The framework integrates directly with the P0.2 canonical time-series quality layer:
- Input datasets must be strictly continuous daily time series.
- Missing calendar dates or explicit missing indicators (`is_missing == True`) raise an immediate `ValueError`.
- Evaluators never silently interpret calendar gaps as zero demand.

---

## 6. How to Run the Production Evaluation

Run the evaluation script from the project root:

```bash
python ml/evaluation/evaluate_production_models.py
```

### Generated Artifacts (`ml/artifacts/`)

| Artifact File | Description | Key Columns |
|---|---|---|
| `production_model_evaluation.csv` | High-level model summary and baseline deltas across evaluated horizons | `model`, `horizon_days`, `number_of_origins`, `WAPE`, `MAE`, `RMSE`, `Bias`, `std_origin_WAPE`, `WAPE_delta_vs_baseline` |
| `production_model_predictions.csv` | Granular day-by-day recursive predictions and individual error metrics | `origin_date`, `forecast_date`, `horizon_days`, `model`, `actual_quantity`, `predicted_quantity`, `error`, `absolute_error` |
| `production_model_origin_metrics.csv` | Metrics calculated individually for each rolling forecast origin block | `origin_date`, `horizon_days`, `model`, `test_observations`, `MAE`, `RMSE`, `WAPE`, `Bias` |

---

## 7. Production Model Benchmark (2025 Holdout Period)

Results evaluated on the 365-day holdout period (`2025-01-01` to `2025-12-31`):

| Model | Horizon | Origins | Days | WAPE | Baseline WAPE | $\Delta_{\text{WAPE}}$ | Bias |
|---|---|---|---|---|---|---|---|
| **Random Forest** | 7-day | 53 | 365 | **2.83%** | 3.64% | **-0.81%** | +57.04 |
| **HistGradientBoosting** | 30-day | 13 | 365 | **2.93%** | 4.82% | **-1.89%** | +105.91 |
| **Linear Regression** | 90-day | 5 | 365 | **4.41%** | 6.56% | **-2.15%** | -221.23 |
