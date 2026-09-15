# Phase 3 — Feature-Based Machine Learning

## Objective
Test whether supervised ML models using lag, rolling, calendar and known promotion/holiday features can beat the 7-day seasonal-naive benchmark.

## Models
- Linear Regression
- Random Forest
- HistGradientBoosting
- Seasonal Naive benchmark

## Primary target
Quantity / daily demand.

## Evaluation
- Validation: 2024
- Final test: 2025
- Horizons: 7, 30, 90 days
- Recursive one-step forecasting for ML models
- Metrics: MAE, RMSE, MAPE, WAPE
- No random shuffling
- Future promotion/holiday calendar variables are treated as known inputs.
- Future demand is never used as an input.

## Best models on 2025 test data

|   Horizon_Days | Evaluation   | Model                |     MAE |    RMSE |   MAPE_% |   WAPE_% |
|---------------:|:-------------|:---------------------|--------:|--------:|---------:|---------:|
|              7 | Test_2025    | Random Forest        |  56.350 |  63.657 |    1.071 |    1.072 |
|             30 | Test_2025    | HistGradientBoosting |  85.628 | 116.286 |    1.544 |    1.586 |
|             90 | Test_2025    | Linear Regression    | 132.514 | 154.442 |    2.283 |    2.267 |

## Next step
Use the strongest feature-based ML model as a benchmark for the TensorFlow LSTM. The LSTM should only be accepted as the production candidate if it provides a meaningful out-of-sample improvement.
