import pandas as pd

from atlas.backtest import rollover_units


def test_rollover_units_monday_to_tuesday_is_one():
    assert rollover_units(
        pd.Timestamp("2024-01-01 10:00"), pd.Timestamp("2024-01-02 10:00")
    ) == 1


def test_rollover_units_wednesday_to_thursday_is_three():
    assert rollover_units(
        pd.Timestamp("2024-01-03 10:00"), pd.Timestamp("2024-01-04 10:00")
    ) == 3


def test_rollover_units_friday_to_monday_is_one():
    assert rollover_units(
        pd.Timestamp("2024-01-05 10:00"), pd.Timestamp("2024-01-08 10:00")
    ) == 1


def test_rollover_units_same_day_is_zero():
    assert rollover_units(
        pd.Timestamp("2024-01-02 10:00"), pd.Timestamp("2024-01-02 18:00")
    ) == 0
