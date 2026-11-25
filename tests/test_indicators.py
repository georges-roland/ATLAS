import pandas as pd
import pytest

from atlas.indicators import atr, ema, true_range
from tests.helpers import make_candles


def test_ema_constant_series_is_constant():
    s = pd.Series([5.0] * 10)
    assert (ema(s, 3) == 5.0).all()


def test_ema_known_values():
    # period 3 -> alpha = 0.5 : [1, 1.5, 2.25]
    out = ema(pd.Series([1.0, 2.0, 3.0]), 3)
    assert out.tolist() == pytest.approx([1.0, 1.5, 2.25])


def test_ema_uses_no_future_data():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    full = ema(s, 3)
    partial = ema(s.iloc[:3], 3)
    assert full.iloc[:3].tolist() == pytest.approx(partial.tolist())


def test_ema_invalid_period():
    with pytest.raises(ValueError):
        ema(pd.Series([1.0]), 0)


def test_true_range_first_candle_is_high_minus_low():
    df = make_candles([(100, 104, 98, 102)])
    assert true_range(df).iloc[0] == 6


def test_true_range_includes_gap():
    # close précédent 102, la bougie suivante ouvre bien plus haut : TR > high-low
    df = make_candles([(100, 104, 98, 102), (110, 112, 109, 111)])
    assert true_range(df).iloc[1] == 10  # |112 - 102|


def test_atr_constant_range_equals_range():
    rows = [(100, 102, 100, 101)] * 30  # range 2, pas de gap
    df = make_candles(rows)
    assert atr(df, 14).iloc[-1] == pytest.approx(2.0)


def test_atr_uses_no_future_data():
    rows = [(100 + i, 102 + i, 99 + i, 101 + i) for i in range(30)]
    df = make_candles(rows)
    assert atr(df, 14).iloc[:20].tolist() == pytest.approx(atr(df.iloc[:20], 14).tolist())
