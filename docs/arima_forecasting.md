# Classical Forecasting Models: ARIMA (P1.0.2b)

This document provides a technical guide to the Classical Autoregressive Integrated Moving Average (ARIMA) forecasting layer implemented in the Smart Sales Forecasting System (`ml/models/arima_forecaster.py`), its pre-holdout order selection methodology (`ml/evaluation/select_arima_order.py`), and its empirical evaluation via the formal rolling-origin framework (`ml/evaluation/evaluate_arima.py`).

---

## 1. Executive Purpose & Context

> [!IMPORTANT]
> **Core Methodological Principle**:
> "Model order selection is performed strictly on pre-holdout data; the 2025 evaluation period is used only for final out-of-sample assessment."

The objective of Phase P1.0.2b is to introduce classical Autoregressive Integrated Moving Average (ARIMA) models into the system's evaluation framework. Following the statistical foundations laid out in Phase P1.0.1 (Time-Series Diagnostics) and Phase P1.0.2a (Exponential Smoothing / ETS), ARIMA models provide a benchmark rooted in the Box-Jenkins methodology.

Unlike heuristic or feature-engineered models, ARIMA models stochastic autocorrelation structures directly through differenced stationary series.

---

## 2. Mathematical Formulation

An $\text{ARIMA}(p, d, q)$ process models a time series $y_t$ by taking $d$ differences to achieve stationarity, then fitting autoregressive (AR) and moving average (MA) polynomials:

$$\left(1 - \sum_{i=1}^p \phi_i B^i\right) (1 - B)^d y_t = c + \left(1 + \sum_{j=1}^q \theta_j B^j\right) \varepsilon_t$$

Where:
- $p \ge 0$: Order of the autoregressive (AR) polynomial, capturing persistence from past differenced values.
- $d \ge 0$: Degree of non-seasonal differencing, converting non-stationary levels into stationary increments.
- $q \ge 0$: Order of the moving average (MA) polynomial, capturing shocks from past forecast errors.
- $B$: Backshift lag operator ($B^k y_t = y_{t-k}$).
- $(1 - B)^d$: Differencing operator. For $d = 1$: $\Delta y_t = y_t - y_{t-1}$.
- $\phi_1, \dots, \phi_p$: Autoregressive coefficients.
- $\theta_1, \dots, \theta_q$: Moving average coefficients.
- $\varepsilon_t \sim \text{WN}(0, \sigma^2)$: White-noise error process.
- $c$: Deterministic trend/constant (omitted when trend is None to avoid drift exaggeration).

Demand forecasts for horizon $h \ge 1$ are generated recursively via state-space Kalman filtering and clipped to enforce physical realism:

$$\hat{y}_{t+h} = \max\left(0.0, \mathbb{E}[y_{t+h} \mid \mathcal{F}_t]\right)$$

Where $\mathcal{F}_t = \{y_1, \dots, y_t\}$ denotes information available strictly up to time $t$.

---

## 3. Why $d = 1$ Differencing Was Selected

Phase P1.0.1 quantitative time-series diagnostics established two definitive properties on the canonical daily demand series:
1. **Level Series Non-Stationarity**: Augmented Dickey-Fuller (ADF) test yielded $t = -0.7160, p = 0.8425$, failing to reject the null hypothesis of a unit root at any conventional significance level ($\alpha = 0.05, \text{critical value} = -2.8623$).
2. **First Difference Stationarity**: Taking the first difference $\Delta y_t = y_t - y_{t-1}$ yielded an ADF statistic of $t = -11.9069, p = 5.37 \times 10^{-22}$, decisively rejecting the unit root null hypothesis.

Consequently, setting $d = 1$ is statistically justified and mathematically necessary to ensure stationarity prior to ARMA modeling.

---

## 4. Candidate Order Grid & Selection Protocol

To avoid combinatorial explosion, severe overfitting, and unbounded compute, an automatic order search over arbitrary ranges was rejected. Instead, a bounded candidate grid of 9 low-order models with $d = 1$ was evaluated:

$$\mathcal{C} = \{(0,1,0), (0,1,1), (1,1,0), (1,1,1), (2,1,0), (0,1,2), (2,1,1), (1,1,2), (2,1,2)\}$$

### 4.1 Strict Pre-Holdout Isolation
- **Holdout Period**: 2025-01-01 through 2025-12-31 (365 days).
- **Selection Cutoff Date**: Strictly $\le$ **2024-12-31** ($N = 2,529$ daily observations).
- The 2025 holdout data was completely blinded and excluded from order selection.

### 4.2 Objective Function: AIC & BIC
Each candidate model was fit on the pre-2025 training series using exact maximum likelihood via statsmodels. Goodness-of-fit was measured by:

$$\text{AIC} = 2k - 2\ln(\hat{L})$$
$$\text{BIC} = k\ln(N) - 2\ln(\hat{L})$$

Where $k$ is the number of estimated parameters ($p + q + 1$), $\hat{L}$ is the maximized likelihood, and $N = 2,529$.

### 4.3 Order Selection Results

| Order | $p$ | $d$ | $q$ | AIC | BIC | Log-Likelihood | Converged |
| :---: | :-: | :-: | :-: | :-: | :-: | :---: | :-: |
| $(0, 1, 0)$ | 0 | 1 | 0 | 36,615.60 | 36,621.43 | -18,306.80 | True |
| $(0, 1, 1)$ | 0 | 1 | 1 | 36,590.82 | 36,602.49 | -18,293.41 | True |
| $(1, 1, 0)$ | 1 | 1 | 0 | 36,612.31 | 36,623.98 | -18,304.16 | True |
| $(1, 1, 1)$ | 1 | 1 | 1 | 36,107.48 | 36,124.98 | -18,050.74 | True |
| $(2, 1, 0)$ | 2 | 1 | 0 | 36,186.28 | 36,203.78 | -18,090.14 | True |
| $(0, 1, 2)$ | 0 | 1 | 2 | 35,872.45 | 35,889.96 | -17,933.23 | True |
| $(2, 1, 1)$ | 2 | 1 | 1 | 35,904.34 | 35,927.68 | -17,948.17 | True |
| $(1, 1, 2)$ | 1 | 1 | 2 | 35,863.67 | 35,887.01 | -17,927.84 | True |
| **$(2, 1, 2)$** | **2** | **1** | **2** | **35,857.65** | **35,886.83** | **-17,923.83** | **True** |

**Selected Model**: **$\text{ARIMA}(2, 1, 2)$** achieves the minimum AIC (35,857.65) and minimum BIC (35,886.83) across all evaluated candidates.

---

## 5. Order Locking & Rolling-Origin Evaluation

Once selected on pre-2025 data, the order $(2, 1, 2)$ was **locked**. It was not re-selected or modified at any point during rolling evaluation.

At each rolling origin $T$ in 2025:
1. The $\text{ARIMA}(2, 1, 2)$ model was refit strictly on observations available prior to $T$ ($\{y_t \mid t < T\}$).
2. Out-of-sample predictions were generated for horizon $h \in \{7, 30, 90\}$.
3. Predictions were compared against actual demand to record error metrics.

This procedure was executed across:
- **7-day horizon**: 53 rolling origins (365 test days).
- **30-day horizon**: 13 rolling origins (365 test days).
- **90-day horizon**: 5 rolling origins (365 test days).

---

## 6. Empirical Holdout Evaluation Results (2025)

The table below summarizes the out-of-sample performance of locked $\text{ARIMA}(2, 1, 2)$ benchmarked against the deterministic `Seasonal Naive` baseline:

| Horizon | Model | Origins | Days | MAE | RMSE | MAPE | WAPE | Bias | $\Delta\text{WAPE}$ vs Baseline |
| :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **7 Days** | Seasonal Naive | 53 | 365 | 209.55 | 397.12 | 3.54% | **3.64%** | +4.15 | 0.00% |
| | **ARIMA(2, 1, 2)** | 53 | 365 | 342.32 | 433.80 | 5.83% | **5.95%** | -2.53 | **+2.31%** |
| **30 Days** | Seasonal Naive | 13 | 365 | 277.74 | 429.94 | 4.78% | **4.82%** | -20.60 | 0.00% |
| | **ARIMA(2, 1, 2)** | 13 | 365 | 358.63 | 460.92 | 6.16% | **6.23%** | +4.71 | **+1.41%** |
| **90 Days** | Seasonal Naive | 5 | 365 | 377.63 | 541.72 | 6.46% | **6.56%** | +79.09 | 0.00% |
| | **ARIMA(2, 1, 2)** | 5 | 365 | 489.03 | 640.23 | 8.40% | **8.50%** | +56.42 | **+1.94%** |

---

## 7. Comparative Analysis: ARIMA vs. Seasonal Naive vs. ETS

Across all three forecast horizons, non-seasonal $\text{ARIMA}(2, 1, 2)$ demonstrates higher forecast errors than both `Seasonal Naive` and `Holt-Winters` (P1.0.2a):

| Horizon | Model | WAPE | MAE | RMSE | $\Delta\text{WAPE}$ vs Baseline |
| :--- | :--- | :-: | :-: | :-: | :-: |
| **7 Days** | Damped Holt-Winters | **2.62%** | 150.93 | 316.76 | **-1.02%** |
| | Holt-Winters Additive | **2.62%** | 150.93 | 316.94 | **-1.02%** |
| | Seasonal Naive | 3.64% | 209.55 | 397.12 | 0.00% |
| | Holt Additive (Non-seasonal) | 5.09% | 293.11 | 467.76 | +1.45% |
| | **ARIMA(2, 1, 2) (Non-seasonal)** | **5.95%** | 342.32 | 433.80 | **+2.31%** |
| **30 Days** | Seasonal Naive | **4.82%** | 277.74 | 429.94 | 0.00% |
| | Damped Holt-Winters | 4.86% | 279.79 | 430.81 | +0.04% |
| | Holt-Winters Additive | 4.93% | 283.64 | 435.29 | +0.10% |
| | **ARIMA(2, 1, 2) (Non-seasonal)** | **6.23%** | 358.63 | 460.92 | **+1.41%** |
| | Holt Additive (Non-seasonal) | 7.76% | 446.54 | 593.55 | +2.93% |
| **90 Days** | Damped Holt-Winters | **6.23%** | 358.46 | 528.47 | **-0.33%** |
| | Holt-Winters Additive | 6.27% | 361.14 | 529.70 | -0.29% |
| | Seasonal Naive | 6.56% | 377.63 | 541.72 | 0.00% |
| | **ARIMA(2, 1, 2) (Non-seasonal)** | **8.50%** | 489.03 | 640.23 | **+1.94%** |
| | Holt Additive (Non-seasonal) | 10.25% | 590.26 | 749.17 | +3.69% |

### Key Methodological Insights:
1. **The Critical Role of Weekly Seasonality**:
   The primary driver of demand in retail sales is the 7-day cyclical shopping pattern ($F_s = 0.8248$, established in P1.0.1).
   Non-seasonal models (both non-seasonal Holt at WAPE 5.09% and non-seasonal ARIMA at WAPE 5.95%) lack a 7-day periodic component and consequently smooth through weekly peaks and troughs.
2. **ARIMA Aggregate Forecast Bias**:
   While non-seasonal ARIMA cannot capture the intra-week cyclic pattern, ARIMA shows relatively low aggregate forecast bias despite higher absolute error ($-2.53$ at 7 days, $+4.71$ at 30 days).
3. **Implication for Phase P1.0.2c (SARIMA)**:
   The empirical underperformance of non-seasonal ARIMA confirms the theoretical prediction of time-series diagnostics: modeling retail demand requires seasonal differencing or seasonal lag terms ($P, D, Q)_7$. This directly motivates **SARIMA**, where seasonal autoregressive and moving average operators with period $s = 7$ will be introduced.

---

## 8. Verification & Test Suite

The ARIMA forecasting module and evaluation harness are verified by 59 automated tests across two test suites:
- [`tests/test_arima_forecaster.py`](file:///c:/Users/prasa/Prasad%20Shetty/Project/Smart%20Sales%20Forecasting%20System/tests/test_arima_forecaster.py): 52 unit tests verifying input validation, minimum history boundaries, non-negative clipping, constant/zero/trending histories, parameter immutability, and determinism.
- [`tests/test_arima_evaluation.py`](file:///c:/Users/prasa/Prasad%20Shetty/Project/Smart%20Sales%20Forecasting%20System/tests/test_arima_evaluation.py): 7 integration tests verifying rolling evaluation integration across 7/14/30-day horizons, coexistence with ETS and ML models, and **adversarial holdout leakage tests** confirming that modifying 2025 actuals produces zero change in pre-2025 order selection.
