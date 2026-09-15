import numpy as np
import pandas as pd

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["day_of_week"] = x.index.dayofweek
    x["day_of_month"] = x.index.day
    x["week_of_year"] = x.index.isocalendar().week.astype(int)
    x["month"] = x.index.month
    x["quarter"] = x.index.quarter
    x["is_weekend"] = (x.index.dayofweek >= 5).astype(int)

    if "trend" not in x.columns:
        x["trend"] = np.arange(len(x))

    return x
