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


# --- Détection des bougies fabriquées -------------------------------------


def make_weekday_candles(n, start="2024-01-01 00:00"):
    """n bougies H1 uniques, valides, sans week-end (lundi-vendredi)."""
    idx = pd.date_range(start=start, periods=n * 2, freq="60min", name="timestamp")
    idx = idx[idx.day_of_week < 5][:n]
    base = [100 + i * 0.01 for i in range(n)]
    df = pd.DataFrame(
        {
            "open": base,
            "high": [b + 1 for b in base],
            "low": [b - 1 for b in base],
            "close": [b + 0.5 for b in base],
            "volume": 100.0,
        },
        index=idx,
    )
    return df


def copy_rows_from_previous(df, every):
    """Remplace une ligne sur `every` par une copie exacte de la précédente."""
    df = df.copy()
    for i in range(every, len(df), every):
        df.iloc[i, :4] = df.iloc[i - 1, :4].to_numpy()
    return df


def test_saturday_candle_is_blocking():
    df = make_candles(GOOD, start="2024-01-06 10:00")  # samedi
    report = validate_candles(df, timeframe_minutes=60)
    assert report.weekend_candles == 3
    assert not report.ok
    assert report.fabricated_suspected


def test_sunday_evening_candle_is_normal():
    df = make_candles(GOOD, start="2024-01-07 23:00")  # dimanche 23:00 : réouverture
    report = validate_candles(df.iloc[:1], timeframe_minutes=60)
    assert report.weekend_candles == 0
    assert report.ok


def test_sunday_morning_candle_is_blocking():
    df = make_candles(GOOD, start="2024-01-07 10:00")  # dimanche 10:00
    report = validate_candles(df.iloc[:1], timeframe_minutes=60)
    assert report.weekend_candles == 1
    assert not report.ok


def test_many_copied_candles_are_blocking():
    df = copy_rows_from_previous(make_weekday_candles(600), every=10)  # ~10 %
    report = validate_candles(df, timeframe_minutes=60)
    assert report.n_rows == 600
    assert report.repeated_candles >= 55
    assert report.weekend_candles == 0
    assert not report.ok


def test_a_few_copied_candles_are_tolerated():
    df = make_weekday_candles(600)
    df.iloc[300, :4] = df.iloc[299, :4].to_numpy()  # 1 ligne sur 600 (~0,17 %)
    report = validate_candles(df, timeframe_minutes=60)
    assert report.repeated_candles == 1
    assert report.ok


def test_many_flat_candles_are_blocking():
    df = make_weekday_candles(600)
    for i in range(0, 600, 20):  # 5 % de bougies plates
        v = df.iloc[i, 0]
        df.iloc[i, :4] = [v, v, v, v]
    report = validate_candles(df, timeframe_minutes=60)
    assert report.flat_candles == 30
    assert not report.ok


def test_flat_and_copied_candles_informative_under_500_rows():
    df = make_weekday_candles(100)
    df.iloc[10, :4] = [100.0, 100.0, 100.0, 100.0]
    df.iloc[20, :4] = df.iloc[19, :4].to_numpy()
    report = validate_candles(df, timeframe_minutes=60)
    assert report.flat_candles >= 1
    assert report.repeated_candles >= 1
    assert report.ok  # sous 500 lignes : information seulement


def test_zero_volume_is_ignored_when_whole_column_is_zero():
    df = make_candles(GOOD)
    df["volume"] = 0.0
    report = validate_candles(df, timeframe_minutes=60)
    assert report.zero_volume == 0
    assert report.ok


def test_zero_volume_is_informational_never_blocking():
    df = make_candles(GOOD)
    df.iloc[1, df.columns.get_loc("volume")] = 0.0
    report = validate_candles(df, timeframe_minutes=60)
    assert report.zero_volume == 1
    assert report.ok


def test_summary_warns_about_fabricated_candles():
    df = make_candles(GOOD, start="2024-01-06 10:00")
    text = validate_candles(df, timeframe_minutes=60).summary()
    assert "fabriquées" in text
    assert "ERREURS A CORRIGER" in text


def test_fabricated_file_is_rejected_by_backtest_pipeline(tmp_path):
    from atlas.backtest import DataValidationError, backtest_from_csv

    df = make_candles(GOOD, start="2024-01-06 10:00")  # samedi
    f = tmp_path / "filled.csv"
    df.reset_index().rename(columns={"timestamp": "time"}).to_csv(f, index=False)
    with pytest.raises(DataValidationError):
        backtest_from_csv(f, timeframe_minutes=60)
