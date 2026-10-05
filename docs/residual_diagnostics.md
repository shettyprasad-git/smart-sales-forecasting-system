# Residual Diagnostics Report (P1.0.2d)

## 1. Executive Summary & Diagnostic Scope

In Phase P1.0.2d, we conducted a rigorous econometric and statistical residual diagnostics analysis across all production and candidate out-of-sample forecast-error series evaluated on the 2025 holdout period ($N = 365$ calendar days).

Residuals are defined under the standard sign convention:
$$e_t = y_t - \hat{y}_t \quad (\text{Actual Quantity} - \text{Predicted Quantity})$$
Under this convention:
- A positive residual ($e_t > 0$) represents an **under-forecast** (actual demand exceeded projection).
- A negative residual ($e_t < 0$) represents an **over-forecast** (projection exceeded actual demand).

Diagnostics were calculated across **21 distinct (model, horizon) series** covering 8 models evaluated at 7-day, 30-day, and 90-day horizons:
1. **Production Models**: Random Forest (7d), HistGradientBoosting (30d), Linear Regression (90d).
2. **Classical Exponential Smoothing**: Holt Additive (7d, 30d, 90d), Holt-Winters Additive Weekly (7d, 30d, 90d), Damped Holt-Winters Additive Weekly (7d, 30d, 90d).
3. **ARIMA & SARIMA**: ARIMA(2,1,2) (7d, 30d, 90d), SARIMA(1,1,1)(0,1,1,7) (7d, 30d, 90d).
4. **Baseline Benchmark**: Seasonal Naive (7d, 30d, 90d).

All residual and forecast-error calculations are based strictly on pre-computed holdout predictions without retraining or modifying prior artifacts.

---

## 2. Residual Diagnostics Summary Table

The table below summarizes the statistical moments, error metrics, serial correlation, variance stability, and outlier frequency across all 21 out-of-sample forecast-error series on the 2025 holdout ($N = 365$ per series).

| Model | Horizon | MAE | RMSE | Bias (Mean $e_t$) | Median $e_t$ | Std $e_t$ | Skew | Kurt | $\hat{\rho}_1$ | $\hat{\rho}_7$ | $\hat{\rho}_{14}$ | $\hat{\rho}_{28}$ | Var Ratio ($s_2^2/s_1^2$) | 3$\sigma$ Outliers | Outlier % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **7-Day Horizon** | | | | | | | | | | | | | | | |
| Damped Holt-Winters Additive | 7d | 150.93 | 316.76 | +22.58 | +2.25 | 316.39 | 0.56 | 9.04 | 0.751 | -0.360 | +0.035 | +0.014 | 2.095 | 23 | 6.30% |
| Holt-Winters Additive Weekly | 7d | 150.93 | 316.94 | +20.01 | +0.29 | 316.75 | 0.55 | 9.04 | 0.751 | -0.360 | +0.035 | +0.014 | 2.092 | 23 | 6.30% |
| Random Forest (Production) | 7d | 163.14 | 240.36 | +57.04 | +23.89 | 233.82 | 0.62 | 2.18 | 0.476 | +0.563 | +0.508 | +0.460 | 0.749 | 3 | 0.82% |
| SARIMA(1,1,1)(0,1,1,7) | 7d | 164.98 | 312.95 | +3.24 | -24.41 | 313.37 | 1.21 | 8.03 | 0.755 | -0.243 | -0.025 | -0.009 | 1.780 | 22 | 6.03% |
| Seasonal Naive | 7d | 209.55 | 397.12 | +4.15 | +6.00 | 397.64 | -0.05 | 4.63 | 0.758 | -0.397 | -0.049 | +0.026 | 1.821 | 10 | 2.74% |
| Holt Additive | 7d | 293.11 | 467.76 | +169.96 | +11.57 | 436.39 | 0.82 | 2.86 | 0.531 | +0.283 | +0.481 | +0.454 | 1.341 | 9 | 2.47% |
| ARIMA(2,1,2) | 7d | 342.32 | 433.80 | -2.53 | -146.50 | 434.39 | 1.21 | 2.13 | 0.553 | +0.485 | +0.422 | +0.489 | 1.320 | 6 | 1.64% |
| **30-Day Horizon** | | | | | | | | | | | | | | | |
| HistGradientBoosting (Production) | 30d | 168.88 | 238.85 | +105.91 | +70.18 | 214.38 | 0.72 | 1.08 | 0.649 | +0.494 | +0.357 | +0.275 | 0.855 | 3 | 0.82% |
| Seasonal Naive | 30d | 277.74 | 429.94 | -20.60 | -25.00 | 430.04 | 0.49 | 2.71 | 0.734 | +0.304 | +0.085 | -0.120 | 1.926 | 3 | 0.82% |
| Damped Holt-Winters Additive | 30d | 279.79 | 430.81 | -83.04 | -45.13 | 423.31 | 0.20 | 2.28 | 0.873 | +0.356 | +0.070 | -0.181 | 1.657 | 3 | 0.82% |
| Holt-Winters Additive Weekly | 30d | 283.64 | 435.29 | -93.48 | -51.05 | 425.71 | 0.18 | 2.22 | 0.874 | +0.357 | +0.071 | -0.179 | 1.645 | 3 | 0.82% |
| SARIMA(1,1,1)(0,1,1,7) | 30d | 285.54 | 411.99 | -83.54 | -67.49 | 403.99 | 0.69 | 2.74 | 0.859 | +0.317 | +0.051 | -0.174 | 1.115 | 8 | 2.19% |
| ARIMA(2,1,2) | 30d | 358.63 | 460.92 | +4.71 | -96.82 | 461.53 | 1.02 | 1.64 | 0.585 | +0.544 | +0.431 | +0.403 | 1.625 | 5 | 1.37% |
| Holt Additive | 30d | 446.54 | 593.55 | -69.73 | -92.96 | 590.25 | 0.26 | 0.77 | 0.734 | +0.583 | +0.355 | +0.073 | 1.070 | 3 | 0.82% |
| **90-Day Horizon** | | | | | | | | | | | | | | | |
| Linear Regression (Production) | 90d | 253.74 | 376.30 | -221.23 | -136.94 | 304.82 | -1.38 | 1.34 | 0.922 | +0.769 | +0.634 | +0.416 | 6.144 | 0 | 0.00% |
| Damped Holt-Winters Additive | 90d | 358.46 | 528.47 | +81.61 | +13.37 | 522.85 | 0.74 | 1.63 | 0.930 | +0.640 | +0.538 | +0.380 | 1.149 | 7 | 1.92% |
| Holt-Winters Additive Weekly | 90d | 361.14 | 529.70 | +53.76 | -13.63 | 527.69 | 0.66 | 1.57 | 0.931 | +0.645 | +0.543 | +0.388 | 1.184 | 7 | 1.92% |
| Seasonal Naive | 90d | 377.63 | 541.72 | +79.09 | +37.00 | 536.65 | 0.60 | 1.40 | 0.898 | +0.659 | +0.562 | +0.417 | 1.227 | 6 | 1.64% |
| SARIMA(1,1,1)(0,1,1,7) | 90d | 408.72 | 583.32 | +42.97 | -17.88 | 582.54 | 0.48 | 1.15 | 0.941 | +0.695 | +0.600 | +0.446 | 1.207 | 7 | 1.92% |
| ARIMA(2,1,2) | 90d | 489.03 | 640.23 | +56.42 | -62.35 | 638.61 | 0.53 | 0.73 | 0.798 | +0.758 | +0.685 | +0.570 | 1.113 | 2 | 0.55% |
| Holt Additive | 90d | 590.26 | 749.17 | -78.67 | -123.19 | 746.05 | 0.33 | 0.31 | 0.848 | +0.809 | +0.744 | +0.629 | 1.064 | 2 | 0.55% |

---

## 3. Ljung-Box Portmanteau Tests

The Ljung-Box portmanteau test evaluates whether a finite sequence of autocorrelations is jointly zero:
$$Q(m) = N(N + 2) \sum_{k=1}^m \frac{\hat{\rho}_k^2}{N - k} \sim \chi^2(m)$$
Under the null hypothesis ($H_0$), the residuals are independently distributed white noise up to lag $m$. A $p$-value $< 0.05$ indicates rejection of $H_0$, indicating that statistically detectable serial correlation remains.

We evaluated portmanteau statistics at lags $m \in \{7, 14, 28\}$:

| Model | Horizon | $Q(7)$ Stat | $Q(7)$ p-val | $Q(14)$ Stat | $Q(14)$ p-val | $Q(28)$ Stat | $Q(28)$ p-val | Rejection ($H_0$ White Noise) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Damped Holt-Winters | 7d | 433.33 | $< 10^{-88}$ | 499.58 | $< 10^{-96}$ | 501.19 | $< 10^{-87}$ | **Rejected ($p < 0.001$)** |
| Holt-Winters Additive | 7d | 433.57 | $< 10^{-88}$ | 499.85 | $< 10^{-97}$ | 501.48 | $< 10^{-87}$ | **Rejected ($p < 0.001$)** |
| Random Forest | 7d | 239.16 | $< 10^{-47}$ | 382.78 | $< 10^{-72}$ | 637.67 | $< 10^{-115}$ | **Rejected ($p < 0.001$)** |
| SARIMA(1,1,1)(0,1,1,7) | 7d | 408.27 | $< 10^{-83}$ | 444.91 | $< 10^{-85}$ | 448.99 | $< 10^{-76}$ | **Rejected ($p < 0.001$)** |
| Seasonal Naive | 7d | 480.84 | $< 10^{-98}$ | 615.39 | $< 10^{-121}$ | 621.38 | $< 10^{-112}$ | **Rejected ($p < 0.001$)** |
| Holt Additive | 7d | 171.21 | $< 10^{-32}$ | 369.47 | $< 10^{-69}$ | 660.57 | $< 10^{-120}$ | **Rejected ($p < 0.001$)** |
| ARIMA(2,1,2) | 7d | 227.20 | $< 10^{-44}$ | 396.37 | $< 10^{-75}$ | 717.69 | $< 10^{-132}$ | **Rejected ($p < 0.001$)** |
| HistGradientBoosting | 30d | 449.40 | $< 10^{-92}$ | 550.57 | $< 10^{-107}$ | 669.81 | $< 10^{-122}$ | **Rejected ($p < 0.001$)** |
| Seasonal Naive | 30d | 717.27 | $< 10^{-149}$ | 750.65 | $< 10^{-150}$ | 769.21 | $< 10^{-143}$ | **Rejected ($p < 0.001$)** |
| Damped Holt-Winters | 30d | 1117.52 | $< 10^{-236}$ | 1209.02 | $< 10^{-248}$ | 1306.16 | $< 10^{-256}$ | **Rejected ($p < 0.001$)** |
| Holt-Winters Additive | 30d | 1119.73 | $< 10^{-236}$ | 1212.12 | $< 10^{-249}$ | 1308.53 | $< 10^{-257}$ | **Rejected ($p < 0.001$)** |
| SARIMA(1,1,1)(0,1,1,7) | 30d | 1036.05 | $< 10^{-218}$ | 1101.80 | $< 10^{-225}$ | 1196.16 | $< 10^{-233}$ | **Rejected ($p < 0.001$)** |
| ARIMA(2,1,2) | 30d | 334.82 | $< 10^{-67}$ | 447.90 | $< 10^{-86}$ | 656.92 | $< 10^{-119}$ | **Rejected ($p < 0.001$)** |
| Holt Additive | 30d | 700.15 | $< 10^{-146}$ | 826.53 | $< 10^{-166}$ | 943.95 | $< 10^{-179}$ | **Rejected ($p < 0.001$)** |
| Linear Regression | 90d | 1799.12 | $0.000$ | 2987.48 | $0.000$ | 4307.44 | $0.000$ | **Rejected ($p < 0.001$)** |
| Damped Holt-Winters | 90d | 1604.89 | $0.000$ | 2462.95 | $0.000$ | 3656.76 | $0.000$ | **Rejected ($p < 0.001$)** |
| Holt-Winters Additive | 90d | 1616.12 | $0.000$ | 2488.22 | $0.000$ | 3705.01 | $0.000$ | **Rejected ($p < 0.001$)** |
| Seasonal Naive | 90d | 1496.47 | $0.000$ | 2317.88 | $0.000$ | 3488.74 | $0.000$ | **Rejected ($p < 0.001$)** |
| SARIMA(1,1,1)(0,1,1,7) | 90d | 1740.90 | $0.000$ | 2794.34 | $0.000$ | 4323.57 | $0.000$ | **Rejected ($p < 0.001$)** |
| ARIMA(2,1,2) | 90d | 1059.47 | $< 10^{-223}$ | 1721.26 | $0.000$ | 2721.63 | $0.000$ | **Rejected ($p < 0.001$)** |
| Holt Additive | 90d | 1356.31 | $< 10^{-287}$ | 2333.63 | $0.000$ | 3891.03 | $0.000$ | **Rejected ($p < 0.001$)** |

### Key Econometric Findings from Portmanteau Tests:
1. **White Noise is Universally Rejected**: Across every single model and horizon, $p < 0.0001$. No model produces mathematically independent white-noise innovations on this daily holdout.
2. **Out-of-Sample Rolling Dynamics**: All evaluated out-of-sample forecast-error series exhibit statistically detectable autocorrelation. Because these are multi-step rolling forecast errors rather than one-step-ahead in-sample innovations, the Ljung–Box results should be interpreted as evidence of remaining temporal structure in forecast errors, not as proof of model invalidity.
3. **Calendar Demand Shocks**: Abrupt demand spikes during promotional and holiday periods create localized runs of positive or negative errors that portmanteau tests readily detect.

---

## 4. Seasonal Structure Removal: SARIMA vs. ARIMA vs. Holt-Winters

A critical question is whether models that explicitly incorporate weekly seasonal terms ($s = 7$) successfully remove the 7-day cyclical structure from the demand series.

### Autocorrelation at Multiples of the Seasonal Lag ($s=7, 14, 28$):
- **At the 7-Day Horizon**:
  - **Non-Seasonal Models**:
    - **ARIMA(2,1,2)** exhibits strong positive weekly autocorrelation: $\hat{\rho}_7 = +0.485$, $\hat{\rho}_{14} = +0.422$, $\hat{\rho}_{28} = +0.489$. Because ARIMA differences non-seasonally without seasonal lag terms, weekly demand cycles are entirely preserved in its residuals.
    - **Holt Additive** similarly fails to remove weekly cycles: $\hat{\rho}_7 = +0.283$, $\hat{\rho}_{14} = +0.481$, $\hat{\rho}_{28} = +0.454$.
  - **Seasonal Models**:
    - **SARIMA(1,1,1)(0,1,1,7)** drives weekly residual autocorrelation negative: $\hat{\rho}_7 = -0.243$, with higher seasonal lags dropping to zero ($\hat{\rho}_{14} = -0.025$, $\hat{\rho}_{28} = -0.009$). SARIMA effectively cleans out the multi-week cyclical persistence.
    - **Holt-Winters Additive Weekly** similarly eliminates positive seasonal persistence: $\hat{\rho}_7 = -0.360$, $\hat{\rho}_{14} = +0.035$, $\hat{\rho}_{28} = +0.014$.
    - **Seasonal Naive** shows a similar pattern: $\hat{\rho}_7 = -0.397$, $\hat{\rho}_{14} = -0.049$, $\hat{\rho}_{28} = +0.026$.
    - **Random Forest** exhibits moderate residual persistence: $\hat{\rho}_7 = +0.563$, $\hat{\rho}_{14} = +0.508$, $\hat{\rho}_{28} = +0.460$, demonstrating that regression trees without recursive feedback leave substantial weekly correlation unmodeled.

- **At the 30-Day and 90-Day Horizons**:
  - At 30 days, SARIMA's seasonal autocorrelation remains moderate ($\hat{\rho}_7 = +0.317, \hat{\rho}_{14} = +0.051$), whereas non-seasonal ARIMA remains higher ($\hat{\rho}_7 = +0.544, \hat{\rho}_{14} = +0.431$).
  - At 90 days, all models experience heavy autocorrelation accumulation ($\hat{\rho}_1 > 0.80, \hat{\rho}_7 > 0.60$), reflecting low-frequency drift and quarterly trend divergence rather than unresolved weekly seasonality.

---

## 5. Variance Stability & Homoskedasticity Analysis

Variance stability was evaluated by computing the ratio of residual variance between the second half (Jul 3, 2025 – Dec 31, 2025, $N_2 = 182$) and the first half (Jan 1, 2025 – Jul 2, 2025, $N_1 = 183$):
$$\text{Variance Ratio} = \frac{\hat{\sigma}_{\text{H2}}^2}{\hat{\sigma}_{\text{H1}}^2}$$

### Findings:
1. **Classical ETS & SARIMA Models (7-Day Horizon)**:
   - Damped Holt-Winters: Variance Ratio = $2.095$ ($s_1 = 254.29 \to s_2 = 368.04$).
   - Holt-Winters Additive: Variance Ratio = $2.092$ ($s_1 = 254.71 \to s_2 = 368.40$).
   - SARIMA(1,1,1)(0,1,1,7): Variance Ratio = $1.780$ ($s_1 = 265.19 \to s_2 = 353.83$).
   - **Economic Interpretation**: The variance ratio of $\approx 1.8 - 2.1$ reflects the genuine seasonal retail demand structure, where Q3/Q4 festival and holiday shopping induces substantially higher daily sales variance than Q1/Q2. The variance expansion is moderate and bounded.
2. **Machine Learning Models**:
   - Random Forest (7d): Variance Ratio = $0.749$ ($s_1 = 241.21 \to s_2 = 208.74$). Random forest demonstrates stable error dispersion across the holdout.
   - HistGradientBoosting (30d): Variance Ratio = $0.855$ ($s_1 = 209.30 \to s_2 = 193.54$). Shows consistent variance across halves.
   - Linear Regression (90d): Variance Ratio = **$6.144$** ($s_1 = 150.14 \to s_2 = 372.14$). Linear regression exhibits extreme variance instability due to rigid extrapolation that deviates substantially from late-year seasonal demand levels.

---

## 6. Outlier Diagnostics ($|z| > 3.0$)

Outliers are defined as out-of-sample forecast errors exceeding 3 standard deviations from the sample mean ($|z_t| = |(e_t - \bar{e}) / s| > 3.0$).

### Key Outlier Characteristics:
- **Total Outlier Frequency**:
  - Classical & SARIMA models at 7 days identify $22 - 23$ outliers ($6.0\% - 6.3\%$).
  - Random Forest at 7 days has only 3 outliers ($0.82\%$).
  - HistGradientBoosting at 30 days has only 3 outliers ($0.82\%$).
  - At 30 and 90 days, outlier rates remain low ($0.5\% - 2.2\%$) because the overall forecast-error standard deviation is wider, expanding the $3\sigma$ threshold.

### Analysis of Extreme Outlier Dates:
Inspection of the top outliers across models reveals distinct clustering on specific retail demand events:
1. **Early March Demand Shock (2025-03-01 to 2025-03-04)**:
   - Actual demand surged to $7,664$ units on March 1 and $7,544$ units on March 2.
   - Models projected $\sim 6,255 - 6,293$ units, producing massive positive forecast errors:
     - Damped Holt-Winters: $e = +1371.65$ ($z = +4.26$)
     - SARIMA: $e = +1408.70$ ($z = +4.49$)
2. **Early August Promotional Surge & Subsequent Drop (2025-08-01 to 2025-08-12)**:
   - On 2025-08-02, actual sales hit $7,391$ (vs predicted $\sim 6,121$, forecast error $+1269.69$, $z = +3.94$).
   - A week later on 2025-08-09, actual demand dropped to $5,937$ while models projected $7,278$, yielding extreme negative forecast errors ($e = -1341.92$, $z = -4.30$).
3. **Late November Pre-Holiday Demand Peak (2025-11-20 to 2025-11-24)**:
   - Actual demand exceeded $6,700$ units, generating large positive forecast errors ($e \approx +1120$, $z \approx +3.5$).

**Sign Asymmetry**: Extreme positive out-of-sample forecast errors ($z > +3.5$) reach up to $+1408$ units, whereas extreme negative forecast errors reach down to $-1341$ units. The upside shocks represent calendar events and promotional campaigns where sales surged well above baseline time-series forecasts.

---

## 7. Distributional Normality: Skewness and Kurtosis

Under a standard Gaussian assumption:
- Skewness $= 0$
- Excess Kurtosis $= 0$ (Fisher definition, where standard normal kurtosis is 0)

### Findings:
1. **Leptokurtosis (Fat Tails)**:
   - At the 7-day horizon, Holt-Winters Additive exhibits kurtosis of $9.04$, Damped Holt-Winters $9.04$, and SARIMA $8.03$. These values strongly reject Gaussian normality, indicating heavy tails driven by sudden demand spikes.
   - At the 30-day and 90-day horizons, kurtosis moderates to $1.0 - 2.7$, approaching Gaussian shape as errors aggregate.
2. **Positive Skewness**:
   - Almost all models exhibit positive skewness ($0.20 - 1.21$). Positive skewness confirms that prediction errors are asymmetric: sudden large positive surges (under-forecasting demand spikes) exceed downward over-forecast deviations.
   - Linear Regression at 90 days is a notable exception with negative skewness ($-1.38$), caused by rigid over-forecasting during periods of slower trend growth.

---

## 8. Answers to Executive Diagnostic Questions

### Question 1: Do any models produce white-noise residuals?
**No.** All evaluated out-of-sample forecast-error series exhibit statistically detectable autocorrelation. Because these are multi-step rolling forecast errors rather than one-step-ahead in-sample innovations, the Ljung–Box results should be interpreted as evidence of remaining temporal structure in forecast errors, not as proof of model invalidity. Ljung-Box portmanteau tests at lags 7, 14, and 28 reject the null hypothesis of white noise at $p < 0.0001$ across all 21 evaluated series.

### Question 2: How does SARIMA residual autocorrelation compare to ARIMA and Holt-Winters?
- **Against ARIMA(2,1,2)**: SARIMA is vastly superior at modeling weekly structure. ARIMA leaves a large positive weekly autocorrelation of $\hat{\rho}_7 = +0.485$ and $\hat{\rho}_{14} = +0.422$. SARIMA eliminates this positive persistence, flipping lag 7 to $\hat{\rho}_7 = -0.243$ and reducing lag 14 to near-zero ($\hat{\rho}_{14} = -0.025$).
- **Against Holt-Winters**: SARIMA and Holt-Winters perform comparably in seasonal structure removal. Holt-Winters achieves $\hat{\rho}_7 = -0.360$ and $\hat{\rho}_{14} = +0.035$. SARIMA shows slightly lower absolute seasonal magnitude ($|-0.243|$ vs $|-0.360|$).

### Question 3: Does residual variance remain stable over time?
Residual variance remains **moderately stable** for classical and SARIMA models, with variance ratios between $1.78$ and $2.09$ from the first half to the second half of 2025. This 2-fold variance increase is consistent with seasonal retail patterns (Q3/Q4 holiday volume). Machine learning models (Random Forest and HistGradientBoosting) demonstrate the highest variance stability (ratios $0.75 - 0.85$). Linear Regression at 90 days exhibits severe instability (ratio $6.14$).

### Question 4: Are residual distributions normal?
**No.** Residual distributions are distinctly non-Gaussian:
- They are **leptokurtic** (fat-tailed), with excess kurtosis reaching $8.0 - 9.0$ at the 7-day horizon.
- They are **positively skewed** ($0.5 - 1.2$), indicating that unpredicted upside demand surges occur more frequently and with greater magnitude than downward drops.

### Question 5: What do the 3-sigma outliers represent?
Outliers ($|z| > 3.0$) represent **exogenous demand events**:
- Early March demand surge (March 1–4)
- Early August promotional peak (August 1–5) and subsequent correction (August 8–12)
- Late November holiday shopping ramp-up (November 20–24)
These spikes reflect external promotional and calendar events that univariate time-series history cannot anticipate without promotional features.

### Question 6: How does forecast horizon impact residual autocorrelation?
Residual autocorrelation **increases dramatically with forecast horizon**:
- At 7 days: $\hat{\rho}_1 \approx 0.48 - 0.75$.
- At 30 days: $\hat{\rho}_1 \approx 0.58 - 0.87$.
- At 90 days: $\hat{\rho}_1 > 0.80 - 0.94$.
At long horizons (90 days), forecast errors are dominated by low-frequency cumulative trend drift, causing out-of-sample forecast errors to behave like persistent, highly autocorrelated processes.

### Question 7: Do residuals show systematic bias?
- At the 7-day horizon: **SARIMA shows virtually zero mean bias** (+3.24 units, $0.05\%$ of mean demand), and ARIMA shows -2.53 units. Holt Additive exhibits severe positive bias (+169.96 units). Random Forest exhibits moderate under-forecast bias (+57.04 units).
- At the 30-day horizon: HistGradientBoosting exhibits moderate under-forecast bias (+105.91 units), while Holt-Winters and SARIMA exhibit slight over-forecast bias ($-83$ to $-93$ units).
- At the 90-day horizon: Linear Regression exhibits massive negative bias ($-221.23$ units), persistently over-forecasting long-run demand.

### Question 8: What are the model recommendation implications?
1. **Forecast Metric Primacy**: Residual diagnostics provide structural insight but do not override empirical forecast metrics (MAE, RMSE, WAPE).
2. **7-Day Recommendation**: Random Forest and Holt-Winters Additive offer the lowest MAE ($150.93 - 163.14$), with SARIMA providing the lowest RMSE ($312.95$) and near-zero aggregate bias (+3.24). SARIMA is the strongest classical seasonal candidate.
3. **30-Day Recommendation**: HistGradientBoosting remains the top production model (MAE $168.88$, RMSE $238.85$), strongly outperforming all classical models (MAEs $> 277$).
4. **90-Day Recommendation**: While Linear Regression achieves low MAE ($253.74$), its residual diagnostics reveal severe structural deficiencies (variance ratio $6.14$, massive bias $-221.23$, persistence $\hat{\rho}_1 = 0.922$). Holt-Winters Additive or Damped Holt-Winters offer far more stable long-term structural characteristics despite higher holdout MAE.

---

## 9. Generated Artifacts Reference

All generated artifacts are located in:
`ml/artifacts/residual_diagnostics/`

### Tabular & JSON Artifacts:
- [`residual_summary.csv`](file:///c:/Users/prasa/Prasad%20Shetty/Project/Smart%20Sales%20Forecasting%20System/ml/artifacts/residual_diagnostics/residual_summary.csv): Full statistical moment, correlation, variance, and outlier summary for all 21 series.
- [`residual_ljung_box.csv`](file:///c:/Users/prasa/Prasad%20Shetty/Project/Smart%20Sales%20Forecasting%20System/ml/artifacts/residual_diagnostics/residual_ljung_box.csv): Portmanteau test statistics, $p$-values, and significance flags at lags 7, 14, and 28.
- [`residual_autocorrelation.csv`](file:///c:/Users/prasa/Prasad%20Shetty/Project/Smart%20Sales%20Forecasting%20System/ml/artifacts/residual_diagnostics/residual_autocorrelation.csv): Full ACF and PACF values from lag 0 through lag 28 for all 21 series.
- [`residual_outliers.csv`](file:///c:/Users/prasa/Prasad%20Shetty/Project/Smart%20Sales%20Forecasting%20System/ml/artifacts/residual_diagnostics/residual_outliers.csv): Granular table of all detected $|z| > 3.0$ observations with forecast date, actual, prediction, residual, and $z$-score.
- [`residual_diagnostics.json`](file:///c:/Users/prasa/Prasad%20Shetty/Project/Smart%20Sales%20Forecasting%20System/ml/artifacts/residual_diagnostics/residual_diagnostics.json): Machine-readable summary for pipeline consumption.

### Publication-Quality Diagnostic Plots:
- **7-Day Horizon**:
  - `residual_timeseries_7d.png`: Residual trajectories over the 2025 holdout calendar.
  - `residual_acf_7d.png`: Autocorrelation function with 95% white-noise bounds.
  - `residual_pacf_7d.png`: Partial autocorrelation function with 95% white-noise bounds.
- **30-Day Horizon**:
  - `residual_timeseries_30d.png`
  - `residual_acf_30d.png`
  - `residual_pacf_30d.png`
- **90-Day Horizon**:
  - `residual_timeseries_90d.png`
  - `residual_acf_90d.png`
  - `residual_pacf_90d.png`
