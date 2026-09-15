# Phase 2 — Baseline Forecasting

Primary target: Quantity / demand.
Secondary target: Sales_Amount / revenue.
Validation: 2024. Final test: 2025.
Horizons: 7, 30, and 90 days.
Naive repeats the last observed value.
Seasonal Naive recursively repeats the value from 7 days earlier.

## Test results

| Target       |   Horizon_Days | Evaluation   | Model             |         MAE |        RMSE |   MAPE_% |   WAPE_% |
|:-------------|---------------:|:-------------|:------------------|------------:|------------:|---------:|---------:|
| Quantity     |              7 | Test_2025    | Seasonal_Naive_7D |     139.286 |     195.685 |    2.672 |    2.650 |
| Quantity     |             30 | Test_2025    | Seasonal_Naive_7D |     243.367 |     295.125 |    4.423 |    4.507 |
| Quantity     |             90 | Test_2025    | Seasonal_Naive_7D |     620.300 |     779.000 |   10.067 |   10.614 |
| Sales_Amount |              7 | Test_2025    | Seasonal_Naive_7D |  534776.842 |  687806.990 |    2.987 |    2.977 |
| Sales_Amount |             30 | Test_2025    | Seasonal_Naive_7D |  900048.624 | 1079851.759 |    4.769 |    4.875 |
| Sales_Amount |             90 | Test_2025    | Seasonal_Naive_7D | 2077428.393 | 2421832.531 |   10.101 |   10.465 |

## Interpretation

These baselines are the minimum performance bar. Phase 3 must beat them using feature-based machine-learning models.

## Next phase
Scikit-learn feature-based forecasting models: Linear Regression, Random Forest and Gradient Boosting.