import pandas as pd
import pytest

from atlas.signal import SignalConfig, generate_signal


FLAT = (100, 100.5, 99.5, 100)
BREAKOUT = (100, 110.5, 99.5, 110)
SIGNAL_EXIT_BARS = [
    FLAT,
    FLAT,
    FLAT,
    FLAT,
    BREAKOUT,
    (111, 112, 110.5, 111.5),
    (111.5, 112, 111, 111.8),
    (111.8, 111.9, 106, 107),
]


def make_df(bars, start="2024-01-02 00:00"):
    index = pd.date_range(start=start, periods=len(bars), freq="60min", name="timestamp")
    frame = pd.DataFrame(
        bars,
        columns=["open", "high", "low", "close"],
        index=index,
    )
    frame["volume"] = 100.0
    return frame


def test_default_config_keeps_validated_strategy_parameters():
    config = SignalConfig()
    assert config.ema_span == 200
    assert config.atr_period == 14
    assert config.sl_mult == 1.5
    assert config.risk_pct == 1.0


def test_no_signal_before_warmup():
    candles = make_df([FLAT, BREAKOUT])
    config = SignalConfig(ema_span=2, atr_period=2)
    assert generate_signal(candles, 10_000, config=config) is None


def test_upward_cross_proposes_a_long_entry_at_signal_close():
    candles = make_df([FLAT] * 4 + [BREAKOUT])
    config = SignalConfig(
        ema_span=2,
        atr_period=2,
        sl_mult=1.0,
        spread=0.0,
        slippage=0.0,
    )

    signal = generate_signal(candles, 10_000, config=config)

    assert signal is not None
    assert signal.action == "buy"
    assert signal.time == candles.index[-1]
    assert signal.reference_price == 110
    assert signal.stop_price == pytest.approx(104)
    assert signal.atr_value == pytest.approx(6)


def test_position_size_is_rounded_down_and_risk_is_reported():
    candles = make_df([FLAT] * 4 + [BREAKOUT])
    config = SignalConfig(
        ema_span=2,
        atr_period=2,
        sl_mult=1.0,
        spread=0.0,
        slippage=0.0,
    )

    signal = generate_signal(candles, 10_000, config=config)

    assert signal is not None
    assert signal.lots == pytest.approx(0.16)
    assert signal.risk_amount == pytest.approx(96)
    assert signal.risk_amount <= 10_000 * config.risk_pct / 100


def test_signal_is_omitted_when_size_is_below_broker_minimum():
    candles = make_df([FLAT] * 4 + [BREAKOUT])
    config = SignalConfig(
        ema_span=2,
        atr_period=2,
        sl_mult=1.0,
        spread=0.0,
        slippage=0.0,
    )
    assert generate_signal(candles, 100, config=config) is None


def test_downward_cross_does_not_propose_a_short():
    candles = make_df(SIGNAL_EXIT_BARS)
    config = SignalConfig(ema_span=2, atr_period=2, sl_mult=1.0)
    assert generate_signal(candles, 10_000, config=config) is None


def test_downward_cross_proposes_closing_an_open_long():
    candles = make_df(SIGNAL_EXIT_BARS)
    config = SignalConfig(ema_span=2, atr_period=2, sl_mult=1.0)

    signal = generate_signal(
        candles,
        10_000,
        position_side="long",
        config=config,
    )

    assert signal is not None
    assert signal.action == "close_long"
    assert signal.time == candles.index[-1]
    assert signal.reference_price == 107
    assert signal.stop_price is None
    assert signal.lots == 0.0


def test_signal_does_not_repeat_after_the_upward_cross():
    candles = make_df([FLAT] * 4 + [BREAKOUT, (110, 111, 109, 110.5)])
    config = SignalConfig(ema_span=2, atr_period=2, sl_mult=1.0)
    assert generate_signal(candles, 10_000, config=config) is None


def test_input_candles_are_not_modified():
    candles = make_df([FLAT] * 4 + [BREAKOUT])
    before = candles.copy(deep=True)
    config = SignalConfig(ema_span=2, atr_period=2, sl_mult=1.0)

    generate_signal(candles, 10_000, config=config)

    pd.testing.assert_frame_equal(candles, before)
