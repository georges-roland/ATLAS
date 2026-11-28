from types import SimpleNamespace

import pandas as pd

from atlas.config import InstrumentSpec
from atlas.mt5_live import evaluate_mt5
from atlas.signal import SignalConfig


class FakeMT5:
    ACCOUNT_TRADE_MODE_DEMO = 0
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    ORDER_TYPE_BUY = 0
    TIMEFRAME_H4 = 16388

    def __init__(self, *, currency="USD", positions=(), server="Exness-Demo"):
        self.initialized = False
        self.shutdown_called = False
        self._positions = positions
        self._account = SimpleNamespace(
            trade_mode=self.ACCOUNT_TRADE_MODE_DEMO,
            server=server,
            currency=currency,
            equity=10_000.0,
        )
        times = pd.date_range("2026-11-27", periods=5, freq="4h", tz="UTC")
        self.rates = [
            {"time": int(times[i].timestamp()), **values}
            for i, values in enumerate(
                [
                    {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0},
                    {"open": 10.0, "high": 12.0, "low": 10.0, "close": 11.0},
                    {"open": 11.0, "high": 12.0, "low": 9.0, "close": 10.0},
                    {"open": 10.0, "high": 13.0, "low": 10.0, "close": 12.0},
                    {"open": 100.0, "high": 200.0, "low": 90.0, "close": 150.0},
                ]
            )
        ]
        self._tick = SimpleNamespace(
            time=int((times[-1] + pd.Timedelta(hours=2)).timestamp()),
            bid=12.0,
            ask=12.2,
        )
        self._symbol = SimpleNamespace(
            currency_profit="USD",
            trade_contract_size=100.0,
            volume_min=0.01,
            volume_step=0.01,
            volume_max=100.0,
        )

    def initialize(self):
        self.initialized = True
        return True

    def account_info(self):
        return self._account

    def symbol_info(self, symbol):
        return self._symbol if symbol == "XAUUSDm" else None

    def symbol_info_tick(self, symbol):
        return self._tick if symbol == "XAUUSDm" else None

    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        return self.rates

    def positions_get(self, *, symbol):
        return self._positions

    def order_calc_profit(self, action, symbol, volume, price_open, price_close):
        return (price_close - price_open) * 91.46 * volume

    def last_error(self):
        return (0, "no error")

    def shutdown(self):
        self.shutdown_called = True


def make_config():
    return SignalConfig(
        ema_span=2,
        atr_period=2,
        sl_mult=1.5,
        risk_pct=1.0,
        spread=0.30,
        slippage=0.05,
        commission_per_lot=0.0,
        instrument=InstrumentSpec("XAUUSDm", 100.0, 0.01, 0.01, 100.0),
    )


def test_live_reader_uses_closed_h4_bars_and_blocks_unverified_costs():
    mt5 = FakeMT5()

    result = evaluate_mt5(mt5=mt5, config=make_config())

    assert mt5.shutdown_called
    assert result.closed_candle_count == 4
    assert result.last_closed_candle_utc == pd.Timestamp("2026-11-27 12:00:00", tz="UTC")
    assert result.signal is not None
    assert result.signal.action == "buy"
    assert result.proposal is None
    assert result.blocked_reason == "Les coûts du broker n'ont pas été confirmés."


def test_live_reader_creates_pending_proposal_only_after_cost_confirmation():
    result = evaluate_mt5(
        mt5=FakeMT5(),
        config=make_config(),
        costs_verified=True,
        commission_per_lot_account_currency=0.0,
    )

    assert result.proposal is not None
    assert result.proposal.status.value == "pending"
    assert result.proposal.entry_price == 12.2
    assert result.proposal.stop_price < result.proposal.entry_price
    assert result.proposal.risk_currency == "USD"
    assert result.blocked_reason is None


def test_live_reader_sizes_eur_account_using_mt5_account_currency_profit():
    result = evaluate_mt5(
        mt5=FakeMT5(currency="EUR"),
        config=make_config(),
        costs_verified=True,
        commission_per_lot_account_currency=2.0,
    )

    assert result.signal is not None
    assert result.signal.action == "buy"
    assert result.proposal is not None
    assert result.proposal.risk_currency == "EUR"
    assert result.proposal.risk_amount <= result.equity * make_config().risk_pct / 100


def test_live_reader_blocks_when_mt5_profit_calculation_fails():
    class FailedProfitMT5(FakeMT5):
        def order_calc_profit(self, action, symbol, volume, price_open, price_close):
            return None

    result = evaluate_mt5(
        mt5=FailedProfitMT5(currency="EUR"),
        config=make_config(),
        costs_verified=True,
        commission_per_lot_account_currency=0.0,
    )

    assert result.proposal is None
    assert result.blocked_reason == "Calcul MT5 du risque impossible ou non négatif."


def test_live_reader_does_not_propose_with_existing_position():
    position = SimpleNamespace(type=FakeMT5.POSITION_TYPE_BUY)
    result = evaluate_mt5(
        mt5=FakeMT5(positions=(position,)),
        config=make_config(),
        costs_verified=True,
        commission_per_lot_account_currency=0.0,
    )

    assert result.position_count == 1
    assert result.proposal is None
    assert result.blocked_reason == "Une position est déjà ouverte sur ce symbole."
