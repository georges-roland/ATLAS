import pandas as pd


def make_candles(rows, start="2024-01-02 00:00", freq="60min"):
    """rows = liste de (open, high, low, close)."""
    idx = pd.date_range(start=start, periods=len(rows), freq=freq, name="timestamp")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 100.0
    return df
