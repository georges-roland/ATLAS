import pandas as pd
import pytest

from atlas.backtest import BacktestConfig, DataValidationError
from atlas.validation import (
    ParamRow,
    TrainTestStudy,
    select_best,
    train_test_from_csv,
    train_test_study,
)
from tests.test_backtest import cfg, two_day_scenario, write_csv


def row(train_profit, train_trades, test_profit=0.0, test_trades=0, ema=20, sl=1.5):
    return ParamRow(ema, sl, train_profit, train_trades, test_profit, test_trades)


def test_select_best_picks_highest_train_profit_among_eligible():
    rows = [row(500, 40, ema=10), row(900, 12, ema=20), row(700, 35, ema=50)]
    best = select_best(rows, min_trades=30)
    assert best.ema_span == 50  # 900 $ ignoré : seulement 12 trades


def test_select_best_returns_none_when_nobody_has_enough_trades():
    assert select_best([row(500, 5), row(900, 29)], min_trades=30) is None


def study_with(rows, bh=0.0):
    return TrainTestStudy("2024-12-31", 100, 50, rows, bh)


def test_verdict_alert_when_best_setting_loses_out_of_sample():
    s = study_with([row(1000, 50, test_profit=-200, test_trades=40)])
    assert s.verdict().startswith("ALERTE")


def test_verdict_not_significant_with_few_test_trades():
    s = study_with([row(1000, 50, test_profit=900, test_trades=10)])
    assert "non significatif" in s.verdict()


def test_verdict_profitable_but_worse_than_buy_and_hold():
    s = study_with([row(1000, 50, test_profit=300, test_trades=40)], bh=800)
    assert "moins bien que l'achat-conservation" in s.verdict()


def test_verdict_positive_is_cautious():
    s = study_with([row(1000, 50, test_profit=900, test_trades=40)], bh=100)
    assert "première vérification" in s.verdict()


def test_verdict_nothing_to_conclude_without_enough_train_trades():
    assert "rien à conclure" in study_with([row(1000, 3)]).verdict()


def test_profitable_in_test_count():
    s = study_with([row(1, 50, 10), row(1, 50, -5), row(1, 50, 3)])
    assert s.profitable_in_test == 2


def test_split_is_disjoint_and_complete():
    df = two_day_scenario()
    s = train_test_study(df, "2024-01-02", cfg(), emas=(2, 3), sl_mults=(1.0,))
    assert s.train_rows + s.test_rows == len(df)
    assert s.train_rows == 24  # toute la journée du 02/01 (00:00 à 23:00)
    assert len(s.rows) == 2


def test_tiny_data_has_no_significant_best():
    s = train_test_study(two_day_scenario(), "2024-01-02", cfg(), emas=(2,), sl_mults=(1.0,))
    assert s.best is None
    assert "rien à conclure" in s.summary()


def test_empty_period_raises():
    with pytest.raises(ValueError, match="Séparation impossible"):
        train_test_study(two_day_scenario(), "2030-01-01", cfg())
    with pytest.raises(ValueError, match="Séparation impossible"):
        train_test_study(two_day_scenario(), "2020-01-01", cfg())


def test_train_and_test_are_backtested_separately():
    df = two_day_scenario()
    s = train_test_study(df, "2024-01-02", cfg(), emas=(2,), sl_mults=(1.0,))
    r = s.rows[0]
    assert r.train_trades == 1  # le trade du 02/01
    assert r.test_trades == 1  # le trade du 04/01, indicateurs repartis de zéro


def test_refuses_invalid_data(tmp_path):
    df = two_day_scenario()
    saturday = pd.DataFrame(
        [(100, 100.5, 99.5, 100)], columns=["open", "high", "low", "close"],
        index=pd.DatetimeIndex(["2024-01-06 10:00"], name="timestamp"),
    )
    saturday["volume"] = 100.0
    f = tmp_path / "bad.csv"
    write_csv(pd.concat([df, saturday]), f)
    with pytest.raises(DataValidationError):
        train_test_from_csv(f, "2024-01-02", cfg())


def test_runs_from_csv_on_valid_data(tmp_path):
    f = tmp_path / "ok.csv"
    write_csv(two_day_scenario(), f)
    s = train_test_from_csv(f, "2024-01-02", cfg(), emas=(2,), sl_mults=(1.0,))
    assert s.train_rows == 24
    assert isinstance(s.summary(), str)


def test_summary_lists_every_combination():
    s = train_test_study(two_day_scenario(), "2024-01-02", cfg(), emas=(2, 3), sl_mults=(1.0, 1.5))
    assert len(s.rows) == 4
    text = s.summary()
    assert text.count("\n") > 8 and "Achat-conservation" in text
