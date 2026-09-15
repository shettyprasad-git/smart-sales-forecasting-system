def add_lag_features(df, target="Quantity", lags=(1, 7, 14, 28)):
    x = df.copy()
    for lag in lags:
        x[f"lag_{lag}"] = x[target].shift(lag)
    return x
