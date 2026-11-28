"""Tests du backtest sur des données construites à la main (aucun aléatoire).

Configuration commune : ema_span=2, atr_period=2, sl_mult=1.0.
Sur les bougies 0-3 (range 1, clôture 100), l'ATR vaut 1 ; la bougie 4 clôture
à 110 avec un range de 11, donc ATR(4) = 0.5*1 + 0.5*11 = 6 -> stop à 6 $.
"""

import pandas as pd
import pytest

from atlas.backtest import (
    BacktestConfig,
    DataValidationError,
    backtest_from_csv,
    max_drawdown,
    run_backtest,
)

FLAT = (100, 100.5, 99.5, 100)
BREAKOUT = (100, 110.5, 99.5, 110)  # bougie 4 : le prix croise l'EMA vers le haut

NO_COST = dict(spread=0.0, slippage=0.0, commission_per_lot=0.0)


def make_df(bars, start="2024-01-02 00:00"):
    idx = pd.date_range(start=start, periods=len(bars), freq="60min", name="timestamp")
    df = pd.DataFrame(bars, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 100.0
    return df


def cfg(**kw):
    base = dict(ema_span=2, atr_period=2, sl_mult=1.0, **NO_COST)
    base.update(kw)
    return BacktestConfig(**base)


def long_scenario(bar6, tail=(111, 111.5, 110.5, 111)):
    """Signal à la clôture de la bougie 4, entrée prévue à l'ouverture de la 5."""
    bars = [FLAT] * 4 + [BREAKOUT, (111, 112, 110.5, 111.5), bar6, tail]
    return make_df(bars)


SL_BY_LOW = (111.5, 112, 104, 111)  # low sous le stop (105), clôture au-dessus


def test_entry_at_next_open_not_at_signal_close():
    df = long_scenario(SL_BY_LOW)
    res = run_backtest(df, cfg())
    t = res.trades[0]
    assert t.entry_time == df.index[5]  # bougie suivant le signal
    assert t.entry_price == 111  # ouverture de la bougie 5, pas la clôture (110)
    assert t.side == "long"


def test_no_trade_during_warmup():
    # Croisement dès la bougie 1 : avant la fin de la période de chauffe
    bars = [FLAT, (100, 110.5, 99.5, 110), (110, 111, 109, 110), (110, 111, 109, 110)]
    res = run_backtest(make_df(bars), cfg(ema_span=3, atr_period=3))
    assert res.n_trades == 0


def test_stop_hit_by_low_while_close_above_stop():
    df = long_scenario(SL_BY_LOW)
    res = run_backtest(df, cfg())
    t = res.trades[0]
    assert df["close"].iloc[6] > 105  # la clôture est AU-DESSUS du stop...
    assert t.exit_reason == "stop_loss"  # ...mais le stop est quand même touché
    assert t.exit_time == df.index[6]
    assert t.exit_price == pytest.approx(105.0)  # prix du stop, pas la clôture


def test_sl_and_tp_in_same_bar_sl_wins():
    both = (111.5, 118, 104, 111)  # TP à 117 et SL à 105 touchés dans la même bougie
    res = run_backtest(long_scenario(both), cfg(tp_mult=1.0))
    assert res.trades[0].exit_reason == "stop_loss"


def test_take_profit_when_only_tp_touched():
    only_tp = (111.5, 118, 111, 117.5)
    res = run_backtest(long_scenario(only_tp), cfg(tp_mult=1.0))
    t = res.trades[0]
    assert t.exit_reason == "take_profit"
    assert t.exit_price == pytest.approx(117.0)


def test_gap_beyond_stop_exits_at_open_not_at_stop():
    gap = (103, 104, 102, 103.5)  # ouvre à 103, sous le stop de 105
    df = long_scenario(gap)
    t = run_backtest(df, cfg()).trades[0]
    assert t.exit_reason == "stop_loss_gap"
    assert t.exit_price == pytest.approx(103.0)  # pire que le stop : le gap est subi


def test_sizing_comes_from_lots_for_risk_and_risk_is_capped():
    df = long_scenario(SL_BY_LOW)
    t = run_backtest(df, cfg(risk_pct=1.0)).trades[0]
    # 10 000 x 1 % = 100 $ ; stop à 6 $ -> 600 $/lot -> 0.1666 -> 0.16 lot (arrondi bas)
    assert t.lots == pytest.approx(0.16)
    assert t.net_pnl == pytest.approx(-96.0)  # (105-111) x 0.16 x 100
    assert -t.net_pnl <= 100.0  # jamais plus que le budget de risque


def test_risk_includes_costs_in_sizing():
    c = cfg(spread=0.3, slippage=0.05, commission_per_lot=7.0)
    t = run_backtest(long_scenario(SL_BY_LOW), c).trades[0]
    # Perte de prix au stop (spread + slippage inclus, hors commission) <= 100 $
    price_loss = -(t.net_pnl + 7.0 * t.lots)
    assert price_loss <= 100.0 + 1e-9


def test_costs_zero_pnl_matches_manual_calculation():
    t = run_backtest(long_scenario(SL_BY_LOW), cfg()).trades[0]
    assert t.costs == 0.0
    assert t.net_pnl == pytest.approx(t.gross_pnl)
    assert t.gross_pnl == pytest.approx((105.0 - 111.0) * t.lots * 100)


def test_costs_make_net_pnl_strictly_worse_per_lot():
    free = run_backtest(long_scenario(SL_BY_LOW), cfg()).trades[0]
    paid = run_backtest(
        long_scenario(SL_BY_LOW), cfg(spread=0.3, slippage=0.05, commission_per_lot=7.0)
    ).trades[0]
    assert paid.costs > 0
    assert paid.net_pnl == pytest.approx(paid.gross_pnl - paid.costs)
    # coûts = (spread + 2 x slippage) x 100 oz x lots + commission x lots
    assert paid.costs == pytest.approx((0.3 + 0.1) * 100 * paid.lots + 7.0 * paid.lots)
    assert paid.net_pnl / paid.lots < free.net_pnl / free.lots


def test_executed_prices_include_spread_and_slippage():
    t = run_backtest(
        long_scenario(SL_BY_LOW), cfg(spread=0.3, slippage=0.05)
    ).trades[0]
    assert t.entry_price == pytest.approx(111 + 0.15 + 0.05)  # achat : on paie plus cher
    assert t.exit_price == pytest.approx(105 - 0.15 - 0.05)  # vente : on reçoit moins


def test_trade_skipped_when_lots_is_zero():
    res = run_backtest(long_scenario(SL_BY_LOW), cfg(initial_capital=100.0))
    assert res.n_trades == 0
    assert res.skipped_trades == 1
    assert res.final_equity == pytest.approx(100.0)


def test_final_equity_equals_initial_plus_sum_of_net_pnl():
    c = cfg(spread=0.3, slippage=0.05, commission_per_lot=7.0)
    res = run_backtest(long_scenario(SL_BY_LOW), c)
    assert res.final_equity == pytest.approx(
        c.initial_capital + sum(t.net_pnl for t in res.trades)
    )


def test_max_drawdown_known_curve():
    eq = pd.Series([10000, 9900, 9800, 10100, 9700, 10200], dtype=float)
    usd, pct = max_drawdown(eq)
    assert usd == pytest.approx(400.0)  # pic 10 100 -> creux 9 700
    assert pct == pytest.approx(400.0 / 10100 * 100)


def test_max_drawdown_is_zero_only_when_curve_never_falls():
    assert max_drawdown(pd.Series([100.0, 101.0, 105.0]))[0] == 0.0
    assert max_drawdown(pd.Series([100.0, 90.0, 120.0]))[0] == pytest.approx(10.0)


def test_drawdown_not_zero_after_a_losing_trade():
    res = run_backtest(long_scenario(SL_BY_LOW), cfg())
    loss = -res.trades[0].net_pnl
    assert loss > 0
    usd, pct = res.drawdown
    assert usd >= loss - 1e-9  # la perte réalisée est visible dans le drawdown
    assert pct > 0


def test_equity_curve_follows_open_position():
    df = long_scenario(SL_BY_LOW)
    res = run_backtest(df, cfg())
    assert len(res.equity_curve) == len(df)
    assert res.equity_curve.iloc[0] == 10_000.0
    assert res.equity_curve.iloc[5] > 10_000.0  # clôture 111.5 > entrée 111


# --- Sortie sur signal, ventes, données invalides -------------------------

SIGNAL_EXIT_BARS = [FLAT] * 4 + [
    BREAKOUT,
    (111, 112, 110.5, 111.5),  # 5 : entrée long à 111
    (111.5, 112, 111, 111.8),
    (111.8, 111.9, 106, 107),  # 7 : close sous l'EMA, low (106) au-dessus du stop (105)
    (106.5, 106.8, 105.5, 106),  # 8 : exécution du signal à l'ouverture
    (106, 106.5, 105.5, 106),
]


def test_exit_on_signal_executes_at_next_open():
    df = make_df(SIGNAL_EXIT_BARS)
    t = run_backtest(df, cfg()).trades[0]
    assert t.exit_reason == "signal"
    assert t.exit_time == df.index[8]
    assert t.exit_price == pytest.approx(106.5)  # ouverture de la bougie 8, pas la clôture 7


def test_short_flag_adds_sells_and_keeps_buys():
    df = make_df(SIGNAL_EXIT_BARS)
    only_long = run_backtest(df, cfg(allow_short=False))
    both = run_backtest(df, cfg(allow_short=True))

    assert {t.side for t in only_long.trades} == {"long"}
    assert "long" in {t.side for t in both.trades}
    assert "short" in {t.side for t in both.trades}
    # Les achats sont identiques avec ou sans l'option
    longs_a = [t for t in only_long.trades if t.side == "long"]
    longs_b = [t for t in both.trades if t.side == "long"]
    assert [(t.entry_time, t.exit_time, t.net_pnl) for t in longs_a] == [
        (t.entry_time, t.exit_time, t.net_pnl) for t in longs_b
    ]
    short = next(t for t in both.trades if t.side == "short")
    assert short.entry_time == df.index[8]  # retournement à l'ouverture suivante


def test_input_dataframe_is_not_modified():
    df = long_scenario(SL_BY_LOW)
    before = df.copy()
    run_backtest(df, cfg())
    pd.testing.assert_frame_equal(df, before)


def test_buy_and_hold_benchmark():
    df = long_scenario(SL_BY_LOW)
    res = run_backtest(df, cfg())
    # ouverture 100 -> dernière clôture 111, sans levier : 10 000 x 111/100
    assert res.buy_hold_final_equity == pytest.approx(10_000 * 111 / 100)


def test_backtest_from_csv_refuses_invalid_data(tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text(
        "time,open,high,low,close\n"
        "2024-01-02 01:00:00,1,2,0.5,1.5\n"
        "2024-01-02 01:00:00,1,2,0.5,1.5\n",  # doublon de date
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError):
        backtest_from_csv(f, cfg(), timeframe_minutes=60)


def test_backtest_from_csv_runs_on_valid_data(tmp_path):
    df = long_scenario(SL_BY_LOW)
    f = tmp_path / "ok.csv"
    df.reset_index().rename(columns={"timestamp": "time"}).to_csv(f, index=False)
    res = backtest_from_csv(f, cfg(), timeframe_minutes=60)
    assert res.n_trades == 1


# --- Robustesse : profit avant coûts, résultats par année, périodes -------

from atlas.backtest import BacktestResult  # noqa: E402,F401

COSTS = dict(spread=0.3, slippage=0.05, commission_per_lot=7.0)


def test_gross_profit_minus_costs_equals_net_profit():
    res = run_backtest(long_scenario(SL_BY_LOW), cfg(**COSTS))
    assert res.total_costs > 0
    assert res.gross_profit - res.total_costs == pytest.approx(res.net_profit)


def test_yearly_profits_sum_to_net_profit():
    res = run_backtest(long_scenario(SL_BY_LOW), cfg(**COSTS))
    assert sum(res.yearly_net_profit().values()) == pytest.approx(res.net_profit)


def test_yearly_profit_uses_exit_year_not_entry_year():
    # Entrée le 31/12/2024 à 23:00 (bougie 5), sortie le 01/01/2025 à 00:00 (bougie 6)
    df = make_df(
        [FLAT] * 4 + [BREAKOUT, (111, 112, 110.5, 111.5), SL_BY_LOW, (111, 111.5, 110.5, 111)],
        start="2024-12-31 18:00",
    )
    res = run_backtest(df, cfg())
    t = res.trades[0]
    assert t.entry_time.year == 2024
    assert t.exit_time.year == 2025
    assert list(res.yearly_net_profit().keys()) == [2025]


def test_low_sample_warning_and_message():
    res = run_backtest(long_scenario(SL_BY_LOW), cfg())
    assert res.n_trades == 1
    assert res.low_sample_warning is True
    assert "ATTENTION : moins de 30 trades, résultat non significatif" in res.summary()
    assert "Profit avant coûts" in res.summary()


def test_no_low_sample_warning_with_enough_trades():
    res = run_backtest(long_scenario(SL_BY_LOW), cfg())
    res.trades = res.trades * 30  # 30 trades au total
    assert res.low_sample_warning is False
    assert "ATTENTION" not in res.summary()


def two_day_scenario():
    """Un trade le mardi 02/01/2024, un autre le jeudi 04/01/2024 (pont plat entre les deux)."""
    first = [FLAT] * 4 + [BREAKOUT, (111, 112, 110.5, 111.5), SL_BY_LOW, (111, 111.5, 110.5, 111)]
    bridge = [FLAT] * 40
    second = [FLAT] * 4 + [BREAKOUT, (111, 112, 110.5, 111.5), SL_BY_LOW, (111, 111.5, 110.5, 111)]
    return make_df(first + bridge + second)  # démarre le mardi 02/01/2024 à 00:00


def write_csv(df, path):
    df.reset_index().rename(columns={"timestamp": "time"}).to_csv(path, index=False)


def test_period_scenario_has_two_trades_on_two_days(tmp_path):
    df = two_day_scenario()
    res = run_backtest(df, cfg())
    assert res.n_trades == 2
    assert res.trades[0].entry_time.date().isoformat() == "2024-01-02"
    assert res.trades[1].entry_time.date().isoformat() == "2024-01-04"


def test_without_start_end_result_is_unchanged(tmp_path):
    df = two_day_scenario()
    f = tmp_path / "d.csv"
    write_csv(df, f)
    direct = run_backtest(df, cfg())
    via_csv = backtest_from_csv(f, cfg(), timeframe_minutes=60)
    assert [(t.entry_time, t.exit_time, t.net_pnl) for t in via_csv.trades] == [
        (t.entry_time, t.exit_time, t.net_pnl) for t in direct.trades
    ]


def test_start_excludes_earlier_trades(tmp_path):
    f = tmp_path / "d.csv"
    write_csv(two_day_scenario(), f)
    res = backtest_from_csv(f, cfg(), timeframe_minutes=60, start="2024-01-04")
    assert res.n_trades == 1
    assert all(t.entry_time >= pd.Timestamp("2024-01-04") for t in res.trades)


def test_end_is_inclusive_for_the_whole_day(tmp_path):
    f = tmp_path / "d.csv"
    write_csv(two_day_scenario(), f)
    res = backtest_from_csv(f, cfg(), timeframe_minutes=60, end="2024-01-02")
    assert res.n_trades == 1  # le trade du 02/01 (05:00) est inclus, celui du 04/01 non
    assert res.trades[0].entry_time.date().isoformat() == "2024-01-02"


def test_empty_period_raises_clear_error(tmp_path):
    f = tmp_path / "d.csv"
    write_csv(two_day_scenario(), f)
    with pytest.raises(ValueError, match="Aucune donnée"):
        backtest_from_csv(f, cfg(), timeframe_minutes=60, start="2030-01-01")


def test_validation_covers_whole_file_even_outside_the_period(tmp_path):
    df = two_day_scenario()
    saturday = make_df([FLAT], start="2024-01-06 10:00")  # bougie de samedi, hors période
    f = tmp_path / "d.csv"
    write_csv(pd.concat([df, saturday]), f)
    with pytest.raises(DataValidationError):
        backtest_from_csv(f, cfg(), timeframe_minutes=60, end="2024-01-04")


def test_low_sample_threshold_is_exactly_30_trades():
    res = run_backtest(long_scenario(SL_BY_LOW), cfg())
    res.trades = res.trades * 29
    assert res.low_sample_warning is True  # 29 trades : encore trop peu
    res.trades = res.trades + res.trades[:1]
    assert res.n_trades == 30
    assert res.low_sample_warning is False  # 30 trades : plus d'alerte
