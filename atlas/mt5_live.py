"""Lecture seule du compte MT5 et préparation prudente d'une proposition."""

from dataclasses import dataclass
import math
from typing import Optional

import pandas as pd

from atlas.proposal import TradeProposal, create_proposal_in_account_currency
from atlas.signal import Signal, SignalConfig, generate_signal


H4_DURATION = pd.Timedelta(hours=4)


@dataclass(frozen=True)
class LiveEvaluation:
    server: str
    symbol: str
    account_currency: str
    equity: float
    position_count: int
    spread: float
    tick_time_utc: pd.Timestamp
    closed_candle_count: int
    last_closed_candle_utc: pd.Timestamp
    signal: Optional[Signal]
    proposal: Optional[TradeProposal]
    blocked_reason: Optional[str]


def evaluate_mt5(
    *,
    symbol: str = "XAUUSDm",
    config: SignalConfig = SignalConfig(),
    costs_verified: bool = False,
    commission_per_lot_account_currency: Optional[float] = None,
    mt5=None,
) -> LiveEvaluation:
    """Lit MT5 et prépare éventuellement une proposition ; n'envoie aucun ordre.

    Une proposition d'entrée reste bloquée tant que les hypothèses de coûts ne
    sont pas explicitement confirmées ou que les caractéristiques du contrat ne
    correspondent pas à celles utilisées par le modèle. Le P/L au stop est
    converti par MT5 dans la devise du compte.
    """
    if commission_per_lot_account_currency is not None and (
        not math.isfinite(commission_per_lot_account_currency)
        or commission_per_lot_account_currency < 0
    ):
        raise ValueError("La commission doit être finie et >= 0 dans la devise du compte")

    if mt5 is None:
        try:
            import MetaTrader5 as mt5_module
        except ImportError as exc:
            raise RuntimeError("Le paquet MetaTrader5 n'est pas installé") from exc
        mt5 = mt5_module

    if not mt5.initialize():
        error_code = mt5.last_error()[0]
        raise RuntimeError(f"Connexion MT5 impossible (code {error_code})")

    try:
        account = mt5.account_info()
        if account is None:
            raise RuntimeError("MT5 ne renvoie pas les informations du compte")
        server = str(getattr(account, "server", ""))
        if account.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
            raise RuntimeError("Le compte MT5 n'est pas un compte démo")
        if "exness" not in server.lower():
            raise RuntimeError("Le compte MT5 connecté n'utilise pas un serveur Exness")
        if not math.isfinite(float(account.equity)) or account.equity <= 0:
            raise RuntimeError("Les fonds propres MT5 sont invalides")

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise RuntimeError(f"Symbole MT5 indisponible : {symbol}")
        tick = mt5.symbol_info_tick(symbol)
        if tick is None or not tick.time or tick.ask <= 0 or tick.bid <= 0:
            raise RuntimeError(f"Cotations MT5 indisponibles : {symbol}")
        tick_time = pd.to_datetime(tick.time, unit="s", utc=True)

        count = max(1000, config.ema_span + 2, config.atr_period + 2)
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, count)
        if rates is None:
            raise RuntimeError(f"Bougies H4 indisponibles (code {mt5.last_error()[0]})")
        bars = pd.DataFrame(rates)
        required = {"time", "open", "high", "low", "close"}
        if bars.empty or not required.issubset(bars.columns):
            raise RuntimeError("MT5 ne renvoie pas de bougies H4 exploitables")
        bars["time"] = pd.to_datetime(bars["time"], unit="s", utc=True)
        bars = bars.loc[bars["time"] + H4_DURATION <= tick_time].sort_values("time")
        if bars.empty:
            raise RuntimeError("Aucune bougie H4 clôturée n'est disponible")
        candles = bars.set_index("time").loc[:, ["open", "high", "low", "close"]]

        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            raise RuntimeError(f"Lecture des positions MT5 impossible (code {mt5.last_error()[0]})")
        position_count = len(positions)
        position_types = {position.type for position in positions}
        short_position = mt5.POSITION_TYPE_SELL in position_types
        position_side = (
            "long"
            if position_types and position_types == {mt5.POSITION_TYPE_BUY}
            else None
        )

        spread = float(tick.ask - tick.bid)
        account_currency = str(account.currency)
        if not short_position:
            signal = generate_signal(
                candles,
                float(account.equity),
                position_side=position_side,
                size_position=False,
                config=config,
            )
        else:
            signal = None

        proposal = None
        blocked_reason = None
        if short_position:
            blocked_reason = "Une position vendeuse est ouverte ; analyse bloquée."
        elif position_count:
            blocked_reason = "Une position est déjà ouverte sur ce symbole."
        elif signal is not None and signal.action == "buy":
            if not _contract_matches(symbol_info, config):
                blocked_reason = "Les spécifications du contrat diffèrent du modèle de risque."
            elif spread > config.spread:
                blocked_reason = (
                    f"Spread courant ({spread:.5f}) supérieur au spread du modèle "
                    f"({config.spread:.5f})."
                )
            elif not costs_verified:
                blocked_reason = "Les coûts du broker n'ont pas été confirmés."
            elif commission_per_lot_account_currency is None:
                blocked_reason = "La commission par lot en devise du compte doit être fournie."
            else:
                def profit_at_stop(volume: float, price_open: float, price_close: float):
                    return mt5.order_calc_profit(
                        mt5.ORDER_TYPE_BUY,
                        symbol,
                        volume,
                        price_open,
                        price_close,
                    )

                try:
                    proposal = create_proposal_in_account_currency(
                        signal,
                        entry_price=float(tick.ask),
                        equity=float(account.equity),
                        profit_at_stop=profit_at_stop,
                        volume_min=float(symbol_info.volume_min),
                        volume_step=float(symbol_info.volume_step),
                        volume_max=float(symbol_info.volume_max),
                        commission_per_lot=commission_per_lot_account_currency,
                        risk_currency=account_currency,
                        config=config,
                    )
                except ValueError as exc:
                    blocked_reason = str(exc)
                if proposal is None and blocked_reason is None:
                    blocked_reason = "La taille calculée est inférieure au lot minimum."

        return LiveEvaluation(
            server=server,
            symbol=symbol,
            account_currency=account_currency,
            equity=float(account.equity),
            position_count=position_count,
            spread=spread,
            tick_time_utc=tick_time,
            closed_candle_count=len(candles),
            last_closed_candle_utc=candles.index[-1],
            signal=signal,
            proposal=proposal,
            blocked_reason=blocked_reason,
        )
    finally:
        mt5.shutdown()


def _contract_matches(symbol_info, config: SignalConfig) -> bool:
    instrument = config.instrument
    actual = (
        getattr(symbol_info, "trade_contract_size", None),
        getattr(symbol_info, "volume_min", None),
        getattr(symbol_info, "volume_step", None),
    )
    expected = (instrument.contract_size, instrument.min_lot, instrument.lot_step)
    try:
        return all(
            math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=1e-9)
            for got, want in zip(actual, expected)
        )
    except (TypeError, ValueError):
        return False
