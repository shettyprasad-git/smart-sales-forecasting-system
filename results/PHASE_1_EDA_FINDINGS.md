# Phase 1 — EDA Findings

## Dataset
- Daily dataset: 2,922 rows
- Date range: 2018-01-01 to 2025-12-31
- Calendar coverage: complete (2,922 expected days)
- Missing values: 0
- Duplicate dates: 0
- Product-level dataset: 1,168,800 product-day records across 400 products

## Target decision
- Primary: `Quantity` (daily demand)
- Secondary: `Sales_Amount` (daily revenue)
- Horizons: 7, 30, 90 days

## Key findings
- Average daily demand: 4,937.8 units
- Average daily revenue: ₹16,811,224
- Saturday demand is about 1.12x Monday demand, indicating weekly structure.
- Promotion days average 1.16x the non-promotion demand, so promotion is a useful explanatory feature.
- Holiday effect is comparatively small in this generated environment.
- Demand autocorrelation is high at multiple lags; trend/seasonality must be handled carefully and baselines are mandatory.
- Annual revenue increases from approximately ₹5.19B in 2018 to ₹7.12B in 2025, so the generated data contains a clear upward trend.

## Modeling features
Lag features: 1, 7, 14, 28 days.
Rolling features: 7, 14, 28 day mean/std.
Calendar features: day-of-week, day-of-month, ISO week, month, quarter, weekend, trend index.
Exogenous features: promotions and holiday flag.

## Split
- Train: before 2024-01-01
- Validation: 2024
- Test: 2025
- No random shuffling.

## Next milestone
Phase 2: implement Naive and Seasonal-Naive baselines, then compare Scikit-learn models before building the TensorFlow LSTM.
