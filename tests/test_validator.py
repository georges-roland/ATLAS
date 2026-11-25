import pandas as pd
import pytest

from atlas.data import validate_candles
from tests.helpers import make_candles

GOOD = [(100, 102, 99, 101), (101, 103, 100, 102), (102, 104, 101, 103)]


def test_clean_data_is_ok():
    report = validate_candles(make_candles(GOOD), timeframe_minutes=60)
    assert report.ok
    assert report.errors == 0
    assert report.unexpected_gaps == []


def test_invalid_ohlc_high_below_close():
    rows = GOOD + [(100, 100.5, 99, 101)]  # high < close
    report = validate_candles(make_candles(rows), timeframe_minutes=60)
    assert report.invalid_ohlc == 1
    assert not report.ok


def test_invalid_ohlc_low_above_open():
    rows = GOOD + [(100, 102, 100.5, 101)]  # low > open
    report = validate_candles(make_candles(rows), timeframe_minutes=60)
    assert report.invalid_ohlc == 1


def test_non_positive_price():
    rows = GOOD + [(100, 102, 0, 101)]
    report = validate_candles(make_candles(rows), timeframe_minutes=60)
    assert report.non_positive == 1
    assert not report.ok


def test_missing_value():
    df = make_candles(GOOD)
    df.iloc[1, df.columns.get_loc("close")] = float("nan")
    report = validate_candles(df, timeframe_minutes=60)
    assert report.missing_values == 1
    assert not report.ok


def test_duplicate_timestamps():
    df = make_candles(GOOD)
    df = pd.concat([df, df.iloc[[0]]])
    report = validate_candles(df, timeframe_minutes=60)
    assert report.duplicates == 1
    assert not report.ok


def test_non_monotonic_timestamps():
    df = make_candles(GOOD).iloc[::-1]
    report = validate_candles(df, timeframe_minutes=60)
    assert report.non_monotonic == 2
    assert not report.ok


def test_unexpected_gap_midweek_is_warning_not_error():
    a = make_candles(GOOD, start="2024-01-02 00:00")  # mardi
    b = make_candles(GOOD, start="2024-01-02 12:00")  # trou de 9 h
    report = validate_candles(pd.concat([a, b]), timeframe_minutes=60)
    assert len(report.unexpected_gaps) == 1
    assert report.ok  # un trou n'est pas une erreur bloquante


def test_weekend_gap_is_not_flagged_as_unexpected():
    a = make_candles(GOOD, start="2024-01-05 20:00")  # vendredi soir
    b = make_candles(GOOD, start="2024-01-08 00:00")  # lundi
    report = validate_candles(pd.concat([a, b]), timeframe_minutes=60)
    assert len(report.gaps) == 1
    assert report.unexpected_gaps == []


def test_empty_dataframe():
    report = validate_candles(make_candles([]), timeframe_minutes=60)
    assert report.n_rows == 0
    assert report.ok


def test_invalid_timeframe_raises():
    with pytest.raises(ValueError):
        validate_candles(make_candles(GOOD), timeframe_minutes=0)
