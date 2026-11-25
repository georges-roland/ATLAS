"""Indicateurs techniques (calculs vectorisés, sans regarder le futur :
la valeur à l'instant t n'utilise que les données jusqu'à t)."""

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Moyenne mobile exponentielle (alpha = 2 / (period + 1))."""
    if period < 1:
        raise ValueError("period doit être >= 1")
    return series.ewm(span=period, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range : max(high-low, |high-close_prec|, |low-close_prec|).
    Pour la première bougie (pas de close précédent), TR = high - low."""
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)  # max ignore les NaN -> high-low sur la 1re ligne


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range, lissage de Wilder (alpha = 1 / period)."""
    if period < 1:
        raise ValueError("period doit être >= 1")
    return true_range(df).ewm(alpha=1.0 / period, adjust=False).mean()
