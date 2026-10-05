# Unified Model Comparison & Horizon-Specific Selection (P1.0.2e)

## 1. Executive Summary & Core Principle

Phase **P1.0.2e — Unified Model Comparison and Horizon-Specific Selection** consolidates the empirical forecasting evidence produced across Phase P0.3 (Production ML), Phase P1.0.2a (Classical ETS), Phase P1.0.2b (ARIMA), Phase P1.0.2c (SARIMA), and Phase P1.0.2d (Residual Diagnostics) into a unified, deterministic, and auditable model-selection layer.

### Core Architectural Principle
> **"Forecast accuracy determines model selection; residual diagnostics provide structural-risk context and model-selection rationale."**

The selection framework separates **model ranking** from **risk assessment**:
1. **Model Selection**: Governed deterministically by out-of-sample forecast accuracy metrics on the established 2025 holdout ($N = 365$ calendar days across multiple rolling origins).
2. **Structural Risk Flag (`LOW`, `MEDIUM`, `HIGH`)**: Evaluated independently from econometric residual diagnostics (variance ratio, serial autocorrelation, systematic bias, and outlier frequency) to alert downstream systems to structural fragility without silently overriding empirical accuracy.

---

## 2. Why Model Selection is Horizon-Specific

Forecasting requirements vary substantially across operational planning horizons:
- **7-Day Horizon (Short-Term Operations & Daily Fulfillment)**: High-frequency weekly cycles ($s = 7$) and immediate calendar effects dominate. Models must capture day-of-week demand oscillations without overreacting to daily white-noise fluctuations.
- **30-Day Horizon (Medium-Term Replenishment & Purchasing)**: Weekly cycles combine with monthly promotional campaigns, day-of-month surges, and non-linear interactions. Machine learning tree ensembles excel by leveraging non-linear feature interactions across multiple calendar dimensions.
- **90-Day Horizon (Quarterly Strategic Planning & Capacity Allocation)**: Low-frequency macro trends, quarterly seasonality, and multi-month growth drift dominate. Multi-step recursive projections compound forecast errors, leading to high serial persistence and potential divergence.

Because no single model family dominates all time scales, selecting models per horizon provides the highest empirical accuracy while bounding operational risk.

---

## 3. Designated Primary Candidate Models

Based on previous phases, each horizon evaluates 5 primary candidate models:

| Horizon | Candidate Models | Benchmark Baseline | Excluded Non-Primary Models |
| :---: | :--- | :---: | :--- |
| **7 Days** | Random Forest, Holt-Winters Additive Weekly, Damped Holt-Winters Additive Weekly, SARIMA(1,1,1)(0,1,1,7), Seasonal Naive | Seasonal Naive | Holt Additive, ARIMA(2,1,2) |
| **30 Days** | HistGradientBoosting, Damped Holt-Winters Additive Weekly, Holt-Winters Additive Weekly, SARIMA(1,1,1)(0,1,1,7), Seasonal Naive | Seasonal Naive | Holt Additive, ARIMA(2,1,2) |
| **90 Days** | Linear Regression, Damped Holt-Winters Additive Weekly, Holt-Winters Additive Weekly, SARIMA(1,1,1)(0,1,1,7), Seasonal Naive | Seasonal Naive | Holt Additive, ARIMA(2,1,2) |

*Rationale for Excluded Models*: Holt Additive and ARIMA(2,1,2) omit weekly seasonal parameters ($s=7$). Across all rolling-origin evaluations in P1.0.2a, P1.0.2b, and P1.0.2d, both models were heavily dominated by seasonal candidates (WAPE $> 5.0\%$, lag-7 residual autocorrelation $> 0.48$). Their artifacts are preserved in the repository for historical traceability, but they are excluded from primary production contention.

---

## 4. Metrics & Deterministic Selection Policy

### Metric Definitions
- **Primary Metric**: **WAPE** (Weighted Absolute Percentage Error):
  $$\text{WAPE} = \frac{\sum_{t=1}^N |y_t - \hat{y}_t|}{\sum_{t=1}^N y_t}$$
  WAPE is volume-weighted, scale-independent, and robust to low-volume or volatile days. Lower WAPE indicates superior forecasting performance.
- **Supporting Context**: $\Delta\text{WAPE}_{\text{vs baseline}} = \text{WAPE}_{\text{model}} - \text{WAPE}_{\text{Seasonal Naive}}$ (negative values reflect percentage-point accuracy improvements over the naive seasonal benchmark).
- **Secondary Accuracy Metrics**: MAE, RMSE, MAPE, Bias ($\frac{1}{N}\sum (y_t - \hat{y}_t)$).

### Deterministic Tie-Breaking Policy
To eliminate subjective scoring and arbitrary weightings, candidate models are ranked using a total ordering comparator:
1. **Lowest WAPE**: Primary ranking metric.
2. **WAPE Tie-Breaking**: If the WAPE difference between two models is $\le 0.10$ percentage points ($0.0010$ in decimal ratio), they are considered **effectively tied on WAPE**; selection proceeds to MAE.
3. **MAE Tie-Breaking**: If the MAE difference is $\le 0.10$ units, they are considered **effectively tied on MAE**; selection proceeds to RMSE.
4. **RMSE Tie-Breaking**: If the RMSE difference is $\le 0.10$ units, they are considered **effectively tied on RMSE**; selection proceeds to absolute Bias ($|\text{Bias}|$).
5. **Bias Tie-Breaking**: If the difference in absolute Bias is $\le 0.10$ units; selection proceeds to variance stability (prefer model whose variance ratio is closer to $1.0$).
6. **Simplicity Hierarchy**: If all accuracy, bias, and variance metrics remain tied within tolerances, the simpler model is selected according to the documented hierarchy:
   $$\text{Seasonal Naive (1)} > \text{Linear Regression (2)} > \text{Holt-Winters (3)} > \text{Damped Holt-Winters (4)} > \text{SARIMA (5)} > \text{Random Forest (6)} > \text{HistGradientBoosting (7)}$$

---

## 5. Structural-Risk Assessment Concept

Residual diagnostics evaluate whether forecast errors contain systematic patterns, variance instability, or persistent autocorrelation. However, **residual diagnostics are secondary evidence** and do not automatically override superior empirical forecast accuracy.

The framework computes a separate **`structural_risk`** flag (`LOW`, `MEDIUM`, `HIGH`) based on explicit, published thresholds:

| Risk Level | Trigger Criteria (Any of the following) | Practical Interpretation |
| :---: | :--- | :--- |
| **HIGH** | 1. Extreme variance instability: $\text{VR} > 3.0$ or $\text{VR} < 0.33$<br>2. Large systematic bias: $|\text{Bias}| \ge 150.0$ units<br>3. Severe persistent autocorrelation: $|\hat{\rho}_1| \ge 0.90$ AND $|\hat{\rho}_7| \ge 0.70$<br>4. Extreme 3-sigma outlier frequency: $\ge 7.0\%$ | High structural fragility. The model achieves empirical holdout accuracy, but its error process displays severe drift, heteroskedasticity, or persistent systematic offset. |
| **MEDIUM** | 1. Moderate variance expansion: $1.5 \le \text{VR} \le 3.0$ or $0.33 \le \text{VR} \le 0.67$<br>2. Meaningful residual persistence: $|\hat{\rho}_1| \ge 0.70$ OR $|\hat{\rho}_7| \ge 0.50$<br>3. Moderate systematic bias: $50.0 \le |\text{Bias}| < 150.0$ units<br>4. Moderate outlier frequency: $3.0\% \le \text{outlier} < 7.0\%$ | Typical out-of-sample multi-step behavior. Moderate error clustering during seasonal demand surges, but bounded variance. |
| **LOW** | Meets all baseline bounds:<br>1. Stable variance: $0.67 < \text{VR} < 1.5$<br>2. Low residual persistence: $|\hat{\rho}_1| < 0.70$ AND $|\hat{\rho}_7| < 0.50$<br>3. Low systematic bias: $|\text{Bias}| < 50.0$ units<br>4. Low outlier frequency: $< 3.0\%$ | Highly stable error process. Errors resemble homoskedastic, low-correlation innovations. |

---

## 6. Empirical Selection Results (2025 Holdout)

### A. Horizon-Specific Selection Table

| Horizon | Selected Model | WAPE | MAE | RMSE | Bias | Structural Risk | Selection Reason |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **7 Days** | **Damped Holt-Winters Additive Weekly** | **2.62%** | 150.93 | 316.76 | +22.58 | **MEDIUM** | Tied on WAPE within 0.10 pp (2.622% vs 2.622%) and MAE within 0.10 units (150.93 vs 150.93); selected by lower RMSE (316.76 vs 316.94; diff = -0.18). |
| **30 Days** | **HistGradientBoosting** | **2.93%** | 168.88 | 238.85 | +105.91 | **MEDIUM** | Selected by lowest WAPE (2.93% vs 4.82% for Seasonal Naive; delta = -1.89 pp). |
| **90 Days** | **Linear Regression** | **4.41%** | 253.74 | 376.30 | -221.23 | **HIGH** | Selected by lowest WAPE (4.41% vs 6.23% for Damped Holt-Winters Additive Weekly; delta = -1.82 pp). |

---

### B. Detailed Candidate Rankings per Horizon

#### 7-Day Horizon
1. **Damped Holt-Winters Additive Weekly** [SELECTED]: WAPE = 2.622%, MAE = 150.93, RMSE = 316.76, Bias = +22.58, Risk = MEDIUM
2. **Holt-Winters Additive Weekly**: WAPE = 2.622%, MAE = 150.93, RMSE = 316.94, Bias = +20.01, Risk = MEDIUM
3. **Random Forest** (Production): WAPE = 2.834%, MAE = 163.14, RMSE = 240.36, Bias = +57.04, Risk = MEDIUM
4. **SARIMA(1,1,1)(0,1,1,7)**: WAPE = 2.866%, MAE = 164.98, RMSE = 312.95, Bias = +3.24, Risk = MEDIUM
5. **Seasonal Naive** (Baseline): WAPE = 3.640%, MAE = 209.55, RMSE = 397.12, Bias = +4.15, Risk = MEDIUM

*Key Observation*: Classical Holt-Winters and Damped Holt-Winters achieve the lowest overall WAPE (2.62%) and MAE (150.93), outperforming Random Forest by 0.21 pp WAPE. SARIMA delivers the lowest RMSE (312.95) and near-zero bias (+3.24), but slightly higher WAPE (2.87%).

#### 30-Day Horizon
1. **HistGradientBoosting** (Production) [SELECTED]: WAPE = 2.934%, MAE = 168.88, RMSE = 238.85, Bias = +105.91, Risk = MEDIUM
2. **Seasonal Naive** (Baseline): WAPE = 4.825%, MAE = 277.74, RMSE = 429.94, Bias = -20.60, Risk = MEDIUM
3. **Damped Holt-Winters Additive Weekly**: WAPE = 4.860%, MAE = 279.79, RMSE = 430.81, Bias = -83.04, Risk = MEDIUM
4. **Holt-Winters Additive Weekly**: WAPE = 4.927%, MAE = 283.64, RMSE = 435.29, Bias = -93.48, Risk = MEDIUM
5. **SARIMA(1,1,1)(0,1,1,7)**: WAPE = 4.960%, MAE = 285.54, RMSE = 412.00, Bias = -83.54, Risk = MEDIUM

*Key Observation*: HistGradientBoosting decisively dominates the 30-day horizon, outperforming all classical and benchmark models by over 1.89 percentage points in WAPE and over 108 units in MAE.

#### 90-Day Horizon
1. **Linear Regression** (Production) [SELECTED]: WAPE = 4.408%, MAE = 253.74, RMSE = 376.30, Bias = -221.23, Risk = HIGH
2. **Damped Holt-Winters Additive Weekly**: WAPE = 6.227%, MAE = 358.46, RMSE = 528.47, Bias = +81.61, Risk = MEDIUM
3. **Holt-Winters Additive Weekly**: WAPE = 6.273%, MAE = 361.14, RMSE = 529.70, Bias = +53.76, Risk = MEDIUM
4. **Seasonal Naive** (Baseline): WAPE = 6.560%, MAE = 377.63, RMSE = 541.72, Bias = +79.09, Risk = MEDIUM
5. **SARIMA(1,1,1)(0,1,1,7)**: WAPE = 7.100%, MAE = 408.72, RMSE = 583.32, Bias = +42.97, Risk = MEDIUM

*Key Observation*: Linear Regression achieves the lowest out-of-sample WAPE (4.41%) and MAE (253.74), but triggers multiple HIGH structural risk criteria (extreme variance ratio of $6.14$, persistent negative bias $-221.23$, and lag-1 autocorrelation $\hat{\rho}_1 = 0.922$). In accordance with governance principles, Linear Regression is selected on accuracy, but transparently flagged as HIGH structural risk.

---

## 7. Model Interpretation: Accuracy Winner vs. Structural Risk

A central achievement of Phase P1.0.2e is transparently reporting both dimensions:

> **"Linear Regression has the lowest observed 90-day WAPE and MAE, but its residual diagnostics show high structural risk."**

- We do **not** state that Linear Regression is invalid or defective.
- We do **not** silently replace Linear Regression with a worse-performing model.
- Instead, downstream inventory, budgeting, or replenishment pipelines receive both the model prediction and its structural risk profile, allowing risk managers to apply hedging or scenario bands appropriately.

---

## 8. Artifacts & Reproducibility

All model comparison and selection artifacts are saved in:
[`ml/artifacts/model_selection/`](file:///c:/Users/prasa/Prasad%20Shetty/Project/Smart%20Sales%20Forecasting%20System/ml/artifacts/model_selection/)

### Artifact Descriptions:
1. [`unified_model_comparison.csv`](file:///c:/Users/prasa/Prasad%20Shetty/Project/Smart%20Sales%20Forecasting%20System/ml/artifacts/model_selection/unified_model_comparison.csv): Full table combining accuracy metrics, residual diagnostics, structural risk, accuracy ranks, and selection indicators across all evaluated candidate models.
2. [`horizon_model_selection.csv`](file:///c:/Users/prasa/Prasad%20Shetty/Project/Smart%20Sales%20Forecasting%20System/ml/artifacts/model_selection/horizon_model_selection.csv): Concise, production-facing table of the winning model for each horizon, its core metrics, structural risk flag, and explicit selection reason.
3. [`model_selection_rationale.json`](file:///c:/Users/prasa/Prasad%20Shetty/Project/Smart%20Sales%20Forecasting%20System/ml/artifacts/model_selection/model_selection_rationale.json): Machine-readable metadata document capturing candidate sets, policy rules, tie-breaking thresholds, per-horizon rankings, and structural risk rationales.

### Execution Command:
```powershell
python -m ml.analysis.run_model_selection
```
