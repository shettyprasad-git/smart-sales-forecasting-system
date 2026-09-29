# Company-Specific Demand Sensitivity & Elasticity Model Architecture

## 1. Architectural Distinction: Pure Volume Forecasting vs. Demand Sensitivity

In the Smart Sales Forecasting System, forecasting demand and estimating price/discount sensitivity are treated as two distinct econometric/machine learning tasks:

1. **Volume Forecasting Pipeline (7D / 30D / 90D Horizons)**:
   - Primary Goal: Predict baseline future sales quantities based on historical autoregressive patterns, day-of-week seasonality, monthly cycles, and calendar indicators (holidays, promotion events).
   - Algorithm Family: HistGradientBoosting, LightGBM, Random Forest, ARIMA/AutoARIMA.
   - Capability: Forecasts volume when commercial terms remain constant or follow historical calendar cadences. It does **not** estimate cross-sectional or longitudinal price elasticity curves.

2. **Demand Sensitivity / Elasticity Model (`CompanyElasticityModel`)**:
   - Primary Goal: Quantify customer responsiveness to changes in unit selling price (\(\beta_p\)) and discount depth (\(\beta_d\)).
   - Algorithm: Regularized log-log multivariate Ridge regression (\(\alpha = 1.0\)).
   - Capability: Enables "Price Change" and "Discount Depth" scenario simulations with mathematically grounded counterfactual quantity and dynamic effective price revenue projections.

---

## 2. Mathematical Specification & Regression Formulation

The dedicated demand-sensitivity model estimates log-transformed quantity response using the following formulation:

\[
\ln(Q_i + \epsilon) = \beta_0 + \beta_p \ln(P_i) + \beta_d \left(\frac{D_i}{100}\right) + \beta_{\text{promo}} \cdot \text{Promo}_i + \beta_{\text{hol}} \cdot \text{Holiday}_i + \sum_{k} \gamma_k \cdot \text{Calendar}_k + \sum_{j} \delta_j \cdot \text{Category}_j
\]

Where:
- \(Q_i\): Historical daily sales quantity for observation \(i\). \(\epsilon = 1.0\) prevents \(\ln(0)\) undefined values.
- \(P_i\): Unit selling price (\(P_i > 0\)).
- \(D_i\): Discount percentage (\(D_i \in [0, 100]\)).
- \(\beta_p\): **Price Elasticity Coefficient**. Reflects the percentage change in quantity demanded per 1% change in price:
  \[
  \beta_p = \frac{\partial \ln Q}{\partial \ln P} \approx \frac{\% \Delta Q}{\% \Delta P}
  \]
  *(Typically negative for normal goods, indicating downward-sloping demand).*
- \(\beta_d\): **Discount Sensitivity Coefficient**. Reflects the percentage change in quantity demanded per percentage point increase in discount depth:
  \[
  \beta_d = 100 \times \frac{\partial \ln Q}{\partial D}
  \]
  *(Typically positive, indicating promotional markdown lift).*
- Regularization: \(\text{Ridge}(\alpha = 1.0)\) with feature standardization avoids unstable or explosive coefficients when price and discount correlate with promotions or seasonal peaks.

---

## 3. Revenue Invariant Mathematics

Simulations enforce strict revenue consistency across all horizons and scenarios:

### 3.1 Baseline Revenue
\[
\text{Baseline Effective Price} = P_0 \times \left(1 - \frac{d_{\text{base}}}{100}\right)
\]
\[
\text{Baseline Revenue} = \sum_{t=1}^{H} Q_{\text{base}, t} \times P_{\text{base\_eff}}
\]

### 3.2 Price Change Counterfactual
For a hypothetical price shift of \(\Delta p\%\):
\[
P_{\text{scen}} = P_0 \times \left(1 + \frac{\Delta p}{100}\right)
\]
\[
\text{Quantity Multiplier} = \left(\frac{P_{\text{scen}}}{P_0}\right)^{\beta_p} = \left(1 + \frac{\Delta p}{100}\right)^{\beta_p}
\]
\[
Q_{\text{scen}, t} = \max\left(0, Q_{\text{base}, t} \times \text{Multiplier}\right)
\]
\[
\text{Scenario Revenue} = \sum_{t=1}^{H} Q_{\text{scen}, t} \times P_{\text{scen}}
\]

### 3.3 Discount Depth Counterfactual
For a hypothetical discount depth shift of \(\Delta d\) percentage points:
\[
d_{\text{scen}} = \min\left(100, \max\left(0, d_{\text{base}} + \Delta d\right)\right)
\]
\[
P_{\text{scen\_eff}} = P_0 \times \left(1 - \frac{d_{\text{scen}}}{100}\right)
\]
\[
\text{Quantity Multiplier} = \exp\left(\beta_d \times \frac{d_{\text{scen}} - d_{\text{base}}}{100}\right)
\]
\[
Q_{\text{scen}, t} = \max\left(0, Q_{\text{base}, t} \times \text{Multiplier}\right)
\]
\[
\text{Scenario Revenue} = \sum_{t=1}^{H} Q_{\text{scen}, t} \times P_{\text{scen\_eff}}
\]

### 3.4 Invariants Enforced
1. **Scenario Unit Price > 0**: Price decreases cannot reach or drop below zero.
2. **Discount Rate \(\in [0, 100]\)**: Clamped within physical limits.
3. **Scenario Quantity \(\ge 0\)**: Demand cannot be negative.
4. **Finite Numerical Outputs**: Zero `NaN`, `Inf`, or division-by-zero artifacts.
5. **Exact Daily Sum**: \(\text{Total Revenue} = \sum_{t=1}^H \text{Daily Revenue}_t\).

---

## 4. Model Boundary & Fail-Closed Safeguards

To prevent ungrounded, fabricated, or commercially deceptive projections, the simulation engine operates on a strict **fail-closed** architecture:

| Condition | Status | UI Behavior | API Response |
|---|---|---|---|
| Active model `READY` with price/discount variation | `ready` | Sliders enabled; live simulation executable | `HTTP 200` with simulation payload & model provenance |
| Total observations < 30 rows | `insufficient_data` | "Model Boundary Restriction: Insufficient Data (< 30 rows)" | `HTTP 422` with exact diagnostic message |
| Constant unit price (std < 1e-4) | `ready` (price disabled) | "Model Boundary Restriction: Price Variation Required" | `HTTP 422` with "insufficient price variation" |
| Constant discount rate (std < 1e-4) | `ready` (discount disabled) | "Model Boundary Restriction: Discount Variation Required" | `HTTP 422` with "insufficient discount variation" |
| No active dataset or model | `unavailable` | "Model Boundary Restriction: Dedicated Elasticity Model Required" | `HTTP 422` with model requirement notice |

---

## 5. Multi-Tenant Isolation & Storage Separation

1. **Database Isolation**:
   - `CompanyElasticityModel` enforces foreign key `user_id` mapped to the authenticated tenant.
   - Partial unique index `uq_company_elasticity_models_user_active` guarantees at most one active elasticity model per tenant.
   - All query paths filter explicitly by `user_id == current_user.id`.

2. **Storage Separation**:
   - Production (`ENVIRONMENT=production`): Artifacts are written to the private Supabase Storage bucket `company-models` under path `models/{user_id}/{dataset_id}/v{version}/elasticity/model.joblib`. Public anonymous access is disabled.
   - Development/Testing: Artifacts are written to local disk under `data/models/{user_id}/`.

---

## 6. Non-Causal Interpretation & Management Notice

> [!WARNING]
> **Observational Data Disclaimer**:
> Historical price elasticity and discount sensitivity estimates reflect observational associations in the company's uploaded sales records. They do not constitute guaranteed causal outcomes. Unobserved confounding factors—such as competitor promotional campaigns, macroeconomic shifts, advertising spend, stockouts, and consumer trend realignments—can influence historical demand.
>
> All what-if scenario simulations are analytical planning tools. They do **not** automatically update live pricing, purchase orders, or ERP inventories. Every commercial decision requires managerial verification and review through the Executive Decision Center.
