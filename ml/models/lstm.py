from pathlib import Path

def get_lstm_artifact(path="ml/artifacts/sales_demand_lstm.keras"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LSTM artifact not found: {path}")
    return path
