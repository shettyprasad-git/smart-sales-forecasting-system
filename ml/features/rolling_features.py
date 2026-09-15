def add_rolling_features(df, target="Quantity", windows=(7, 14, 28)):
    x = df.copy()
    shifted = x[target].shift(1)

    for window in windows:
        x[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        x[f"rolling_std_{window}"] = shifted.rolling(window).std()

    return x
