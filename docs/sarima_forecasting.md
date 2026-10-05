# Classical Forecasting Models: SARIMA / Seasonal ARIMA (P1.0.2c)

This document details the Seasonal Autoregressive Integrated Moving Average (SARIMA) forecasting layer implemented in the Smart Sales Forecasting System (`ml/models/sarima_forecaster.py`), its pre-holdout model selection methodology (`ml/evaluation/select_sarima_order.py`), and its empirical evaluation using the formal rolling-origin framework (`ml/evaluation/evaluate_sarima.py`).

---

## 1. Executive Purpose & Context

> [!IMPORTANT]
> **Core Methodological Principle**:
> "SARIMA specification selection is performed strictly on pre-holdout data; the 2025 period is reserved exclusively for out-of-sample evaluation."

Phase P1.0.2b revealed that while non-seasonal $\text{ARIMA}(2, 1, 2)$ removed non-stationary levels ($d=1$), it could not capture the strong 7-day cyclical demand pattern ($F_s = 0.8248$), yielding higher forecast error ($5.95\%$ WAPE at 7 days) than both the `Seasonal Naive` baseline ($3.64\%$) and `Holt-Winters` ($2.62\%$).

The objective of Phase P1.0.2c is to introduce classical **Seasonal ARIMA (SARIMA)** to model both non-seasonal increments and the 7-day weekly cyclical demand structure directly within the Box-Jenkins state-space framework.

---

## 2. Mathematical Formulation

A multiplicative seasonal autoregressive integrated moving average process $\text{SARIMA}(p, d, q)(P, D, Q)_s$ with seasonal period $s = 7$ is formulated as:

$$\Phi_P(B^s) \phi_p(B) (1 - B)^d (1 - B^s)^D y_t = c + \Theta_Q(B^s) \theta_q(B) \varepsilon_t$$

Where:
- **Non-Seasonal Operators**:
  - $\phi_p(B) = 1 - \sum_{i=1}^p \phi_i B^i$: Non-seasonal autoregressive polynomial of order $p$.
  - $(1 - B)^d$: Non-seasonal differencing operator of degree $d$. For $d = 1$: $\Delta y_t = y_t - y_{t-1}$.
  - $\theta_q(B) = 1 + \sum_{j=1}^q \theta_j B^j$: Non-seasonal moving average polynomial of order $q$.
- **Seasonal Operators ($s = 7$)**:
  - $\Phi_P(B^s) = 1 - \sum_{i=1}^P \Phi_i B^{i \cdot s}$: Seasonal autoregressive polynomial of order $P$.
  - $(1 - B^s)^D$: Seasonal differencing operator of degree $D$. For $D = 1$: $\Delta_7 y_t = y_t - y_{t-7}$.
  - $\Theta_Q(B^s) = 1 + \sum_{j=1}^Q \Theta_j B^{j \cdot s}$: Seasonal moving average polynomial of order $Q$.
- $\varepsilon_t \sim \text{WN}(0, \sigma^2)$: White noise innovation sequence.
- $c$: Deterministic trend/drift term (omitted to avoid linear drift extrapolation).

Demand forecasts for horizon $h \ge 1$ are generated recursively via state-space Kalman filtering and clipped to enforce non-negative physical demand:

$$\hat{y}_{t+h} = \max\left(0.0, \mathbb{E}[y_{t+h} \mid \mathcal{F}_t]\right)$$

Where $\mathcal{F}_t = \{y_1, \dots, y_t\}$ represents historical observations strictly prior to origin time $t$.

---

## 3. Stationarity, Seasonality & Differencing Rationale

1. **Why $d = 1$**:
   Phase P1.0.1 time-series diagnostics established that the raw demand series is non-stationary (ADF test: $t = -0.7160, p = 0.8425$), while first differencing yields decisive stationarity ($t = -11.9069, p < 10^{-20}$).
2. **Why $s = 7$**:
   Phase P1.0.1 established strong weekly periodicity in the autocorrelation function ($\hat{\rho}_7 = 0.8850, \hat{\rho}_{14} = 0.8625, \hat{\rho}_{21} = 0.8481, \hat{\rho}_{28} = 0.8348$) and high Wang-Smith-Hyndman seasonal strength ($F_s = 0.8248$). A period of $s = 7$ maps exactly to weekly retail trading cycles.
3. **Selection of $D$ ($D \in \{0, 1\}$)**:
   Rather than assuming $D = 1$ a priori, candidate models both with $D = 0$ (seasonal AR/MA without seasonal differencing) and $D = 1$ (seasonal differencing) were evaluated on pre-holdout data. Empirical evaluation confirmed that $D = 1$ substantially improves likelihood and information criteria.

---

## 4. Controlled Candidate Grid & Pre-Holdout Selection

To prevent unbounded compute and over-parameterization, an unrestricted search was rejected. A bounded candidate set of 16 parsimonious specifications combining established low-order structures was evaluated:

### 4.1 Candidate Grid Evaluation (Cutoff: `2024-12-31`, $N = 2,529$)

| Candidate Order | Seasonal Order | $p$ | $d$ | $q$ | $P$ | $D$ | $Q$ | $s$ | AIC | BIC | Log-Likelihood | Converged |
| :---: | :---: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| $(0, 1, 0)$ | $(1, 0, 0, 7)$ | 0 | 1 | 0 | 1 | 0 | 0 | 7 | 34,571.16 | 34,582.82 | -17,283.58 | True |
| $(0, 1, 0)$ | $(0, 1, 1, 7)$ | 0 | 1 | 0 | 0 | 1 | 1 | 7 | 32,665.45 | 32,677.11 | -16,330.73 | True |
| $(0, 1, 1)$ | $(1, 0, 0, 7)$ | 0 | 1 | 1 | 1 | 0 | 0 | 7 | 34,496.62 | 34,514.12 | -17,245.31 | True |
| $(1, 1, 0)$ | $(1, 0, 0, 7)$ | 1 | 1 | 0 | 1 | 0 | 0 | 7 | 34,491.23 | 34,508.73 | -17,242.62 | True |
| $(1, 1, 1)$ | $(1, 0, 0, 7)$ | 1 | 1 | 1 | 1 | 0 | 0 | 7 | 34,234.02 | 34,257.35 | -17,113.01 | True |
| $(0, 1, 1)$ | $(0, 0, 1, 7)$ | 0 | 1 | 1 | 0 | 0 | 1 | 7 | 35,575.95 | 35,593.44 | -17,784.97 | True |
| $(1, 1, 0)$ | $(0, 0, 1, 7)$ | 1 | 1 | 0 | 0 | 0 | 1 | 7 | 35,612.69 | 35,630.19 | -17,803.35 | True |
| $(1, 1, 1)$ | $(0, 0, 1, 7)$ | 1 | 1 | 1 | 0 | 0 | 1 | 7 | 35,194.11 | 35,217.44 | -17,593.06 | True |
| $(0, 1, 1)$ | $(1, 0, 1, 7)$ | 0 | 1 | 1 | 1 | 0 | 1 | 7 | 32,618.22 | 32,641.54 | -16,305.11 | True |
| $(1, 1, 0)$ | $(1, 0, 1, 7)$ | 1 | 1 | 0 | 1 | 0 | 1 | 7 | 32,642.98 | 32,666.31 | -16,317.49 | True |
| $(1, 1, 1)$ | $(1, 0, 1, 7)$ | 1 | 1 | 1 | 1 | 0 | 1 | 7 | 32,663.01 | 32,692.17 | -16,326.50 | True |
| $(0, 1, 1)$ | $(0, 1, 1, 7)$ | 0 | 1 | 1 | 0 | 1 | 1 | 7 | 32,545.66 | 32,563.15 | -16,269.83 | True |
| $(1, 1, 0)$ | $(0, 1, 1, 7)$ | 1 | 1 | 0 | 0 | 1 | 1 | 7 | 32,561.87 | 32,579.36 | -16,277.94 | True |
| **$(1, 1, 1)$** | **$(0, 1, 1, 7)$** | **1** | **1** | **1** | **0** | **1** | **1** | **7** | **32,519.87** | **32,543.18** | **-16,255.93** | **True** |
| $(0, 1, 1)$ | $(1, 1, 0, 7)$ | 0 | 1 | 1 | 1 | 1 | 0 | 7 | 33,544.93 | 33,562.42 | -16,769.47 | True |
| $(1, 1, 0)$ | $(1, 1, 0, 7)$ | 1 | 1 | 0 | 1 | 1 | 0 | 7 | 33,533.65 | 33,551.14 | -16,763.82 | True |

**Selected Specification**: **$\text{SARIMA}(1, 1, 1)(0, 1, 1, 7)$** achieved the minimum AIC (**$32,519.87$**) and minimum BIC (**$32,543.18$**) across all 16 candidates.

---

## 5. Minimum History Validation Rule

SARIMA requires adequate historical depth to perform compound differencing $(1 - B)^d (1 - B^s)^D$ and observe recurring seasonal cycles.

The defensible minimum history rule is defined as:

$$N_{\min} = \max\left(2 \cdot s + d + D \cdot s, \; d + D \cdot s + p + q + (P + Q) \cdot s + 2\right)$$

For $\text{SARIMA}(1, 1, 1)(0, 1, 1, 7)$ ($s=7$):
- $2 \cdot s + d + D \cdot s = 14 + 1 + 7 = 22$
- $d + D \cdot s + p + q + (P + Q) \cdot s + 2 = 1 + 7 + 1 + 1 + 7 + 2 = 19$
- $N_{\min} = \max(22, 19) = 22$ observations.

The evaluation configuration enforces `min_history_days = 28`, well exceeding $N_{\min}$.

---

## 6. Empirical Holdout Evaluation Results (2025)

The selected specification $\text{SARIMA}(1, 1, 1)(0, 1, 1, 7)$ was **locked** and evaluated through [`RollingEvaluator`](file:///c:/Users/prasa/Prasad Shetty/Project/Smart%20Sales%20Forecasting%20System/ml/evaluation/rolling_evaluator.py) across all 71 rolling origins in 2025:

| Horizon | Model | Origins | Days | MAE | RMSE | MAPE | WAPE | Bias | $\Delta\text{WAPE}$ vs Baseline |
| :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **7 Days** | **SARIMA(1, 1, 1)(0, 1, 1, 7)** | 53 | 365 | **164.98** | **312.95** | **2.77%** | **2.87%** | +3.24 | **-0.77%** |
| | Seasonal Naive | 53 | 365 | 209.55 | 397.12 | 3.54% | 3.64% | +4.15 | 0.00% |
| **30 Days** | Seasonal Naive | 13 | 365 | 277.74 | 429.94 | 4.78% | **4.82%** | -20.60 | 0.00% |
| | **SARIMA(1, 1, 1)(0, 1, 1, 7)** | 13 | 365 | 285.54 | **412.00** | 4.91% | 4.96% | -83.54 | +0.14% |
| **90 Days** | Seasonal Naive | 5 | 365 | **377.63** | **541.72** | **6.46%** | **6.56%** | +79.09 | 0.00% |
| | **SARIMA(1, 1, 1)(0, 1, 1, 7)** | 5 | 365 | 408.72 | 583.32 | 7.04% | 7.10% | +42.97 | +0.54% |

---

## 7. Comparative Benchmark Analysis Across Model Families

| Horizon | Model Family | Model Name | WAPE | MAE | RMSE | $\Delta\text{WAPE}$ vs Baseline |
| :--- | :--- | :--- | :-: | :-: | :-: | :-: |
| **7 Days** | Classical ETS | Damped Holt-Winters | **2.62%** | 150.93 | 316.76 | **-1.02%** |
| | Classical ETS | Holt-Winters Additive | **2.62%** | 150.93 | 316.94 | **-1.02%** |
| | **Classical SARIMA** | **SARIMA(1, 1, 1)(0, 1, 1, 7)** | **2.87%** | **164.98** | **312.95** | **-0.77%** |
| | Deterministic Baseline | Seasonal Naive | 3.64% | 209.55 | 397.12 | 0.00% |
| | Classical ETS | Holt Additive (Non-seasonal) | 5.09% | 293.11 | 467.76 | +1.45% |
| | Classical ARIMA | ARIMA(2, 1, 2) (Non-seasonal) | 5.95% | 342.32 | 433.80 | +2.31% |
| **30 Days** | Deterministic Baseline | Seasonal Naive | **4.82%** | 277.74 | 429.94 | 0.00% |
| | Classical ETS | Damped Holt-Winters | 4.86% | 279.79 | 430.81 | +0.04% |
| | Classical ETS | Holt-Winters Additive | 4.93% | 283.64 | 435.29 | +0.10% |
| | **Classical SARIMA** | **SARIMA(1, 1, 1)(0, 1, 1, 7)** | **4.96%** | **285.54** | **412.00** | **+0.14%** |
| | Classical ARIMA | ARIMA(2, 1, 2) (Non-seasonal) | 6.23% | 358.63 | 460.92 | +1.41% |
| | Classical ETS | Holt Additive (Non-seasonal) | 7.76% | 446.54 | 593.55 | +2.93% |
| **90 Days** | Classical ETS | Damped Holt-Winters | **6.23%** | 358.46 | 528.47 | **-0.33%** |
| | Classical ETS | Holt-Winters Additive | 6.27% | 361.14 | 529.70 | -0.29% |
| | Deterministic Baseline | Seasonal Naive | 6.56% | 377.63 | 541.72 | 0.00% |
| | **Classical SARIMA** | **SARIMA(1, 1, 1)(0, 1, 1, 7)** | **7.10%** | **408.72** | **583.32** | **+0.54%** |
| | Classical ARIMA | ARIMA(2, 1, 2) (Non-seasonal) | 8.50% | 489.03 | 640.23 | +1.94% |
| | Classical ETS | Holt Additive (Non-seasonal) | 10.25% | 590.26 | 749.17 | +3.69% |

### Key Methodological Findings:
1. **Dramatic Error Reduction Over Non-Seasonal ARIMA**:
   Adding seasonal differencing ($D=1$) and seasonal moving average ($Q=1, s=7$) reduced 7-day WAPE from **$5.95\%$** to **$2.87\%$** (a $>50\%$ relative error reduction), outperforming Seasonal Naive by **$-0.77\%$**.
2. **Lowest RMSE on 7-Day and 30-Day Horizons**:
   SARIMA produced the lowest RMSE among the evaluated models at the 7-day and 30-day horizons, indicating comparatively lower sensitivity to large squared errors in this holdout (RMSE of **$312.95$** at 7 days and **$412.00$** at 30 days).
3. **Weekly Seasonality is the Decisive Factor**:
   Both seasonal families (Holt-Winters and SARIMA) dramatically outperform their non-seasonal counterparts (Holt and ARIMA), proving that explicit 7-day modeling is essential for sales forecasting in this domain.
