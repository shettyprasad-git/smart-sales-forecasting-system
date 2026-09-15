import pandas as pd
from .data_loader import build_daily_aggregate

def prepare_daily_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert product-day records into the project daily forecasting dataset."""
    return build_daily_aggregate(df)
