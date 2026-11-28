"""Signal temps réel pour la stratégie XAU/USD validée.

Le signal est calculé à la clôture de la dernière bougie fournie. Le prix
utilisé pour présenter l'entrée est donc une référence ; l'exécution éventuelle
se fera plus tard et devra recalculer le stop et la taille au prix réel.
"""

from dataclasses import dataclass
import math
from typing import Literal, Optional

import numpy as np
import pandas as pd

from atlas.config import XAUUSD, InstrumentSpec
from atlas.indicators import atr, ema
from atlas.risk import lots_for_risk


@dataclass(frozen=True)
class SignalConfig:
    """Paramètres du signal. Les valeurs par défaut gardent la stratégie validée."""

    ema_span: int = 200
    atr_period: int = 14
    sl_mult: float = 1.5
    risk_pct: float = 1.0
    spread: float = 0.30
    slippage: float = 0.05
    commission_per_lot: float = 0.0
    instrument: InstrumentSpec = XAUUSD

    def __post_init__(self) -> None:
        if self.ema_span < 1:
            raise ValueError("ema_span doit être >= 1")
        if self.atr_period < 1:
            raise ValueError("atr_period doit être >= 1")
        if not math.isfinite(self.sl_mult) or self.sl_mult <= 0:
            raise ValueError("sl_mult doit être > 0")
        if not math.isfinite(self.risk_pct) or not 0 < self.risk_pct <= 100:
            raise ValueError("risk_pct doit être dans ]0, 100]")
        if any(
            not math.isfinite(value) or value < 0
            for value in (self.spread, self.slippage, self.commission_per_lot)
        ):
            raise ValueError("les coûts doivent être finis et >= 0")


@dataclass(frozen=True)
class Signal:
    """Événement de signal ; un signal ne place jamais d'ordre."""

    action: Literal["buy", "close_long"]
    time: pd.Timestamp
    reference_price: float
    ema_value: float
    atr_value: float
    stop_price: Optional[float] = None
    lots: float = 0.0
    risk_amount: float = 0.0


def generate_signal(
    candles: pd.DataFrame,
    equity: float,
    *,
    position_side: Optional[Literal["long"]] = None,
    config: SignalConfig = SignalConfig(),
) -> Optional[Signal]:
    """Calcule le signal sur la dernière bougie clôturée.

    Un achat est proposé lors d'un croisement haussier du cours au-dessus de
    l'EMA. Si une position longue est déjà ouverte, un croisement baissier
    produit un signal de clôture. Les ventes ne sont pas proposées.

    L'entrée affichée est le close de la bougie de signal, pas le futur prix
    d'exécution. Le stop de la proposition est à 1,5 ATR par défaut. Le calcul
    des lots suit le backtest : spread et slippage sont ajoutés au risque de
    prix pour le dimensionnement ; la commission est incluse dans le risque
    affiché, mais pas dans ce dimensionnement.
    """
    if not math.isfinite(equity) or equity <= 0:
        raise ValueError("equity doit être > 0")
    if position_side not in (None, "long"):
        raise ValueError("position_side doit être None ou 'long'")
    if not isinstance(candles, pd.DataFrame):
        raise TypeError("candles doit être un DataFrame pandas")

    required = ("open", "high", "low", "close")
    missing = [column for column in required if column not in candles.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {', '.join(missing)}")
    if not isinstance(candles.index, pd.DatetimeIndex):
        raise ValueError("l'index des bougies doit être un DatetimeIndex")
    if not candles.index.is_monotonic_increasing or candles.index.has_duplicates:
        raise ValueError("les bougies doivent être triées et sans doublons")

    try:
        prices = candles.loc[:, required].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("les prix OHLC doivent être numériques") from exc
    if not np.isfinite(prices).all():
        raise ValueError("les prix OHLC doivent être finis")

    warmup = max(config.ema_span, config.atr_period)
    if len(candles) <= warmup:
        return None

    ema_values = ema(candles["close"], config.ema_span)
    atr_values = atr(candles, config.atr_period)
    previous_close = float(candles["close"].iloc[-2])
    current_close = float(candles["close"].iloc[-1])
    previous_ema = float(ema_values.iloc[-2])
    current_ema = float(ema_values.iloc[-1])
    current_atr = float(atr_values.iloc[-1])
    if not math.isfinite(current_atr) or current_atr <= 0:
        return None

    crossed_up = previous_close <= previous_ema and current_close > current_ema
    crossed_down = previous_close >= previous_ema and current_close < current_ema

    if position_side == "long" and crossed_down:
        return Signal(
            action="close_long",
            time=candles.index[-1],
            reference_price=current_close,
            ema_value=current_ema,
            atr_value=current_atr,
        )
    if position_side is not None or not crossed_up:
        return None

    stop_distance = config.sl_mult * current_atr
    risk_distance = stop_distance + config.spread + 2.0 * config.slippage
    stop_price = current_close - stop_distance
    sizing_stop = current_close - risk_distance
    if stop_price <= 0 or sizing_stop <= 0:
        return None

    lots = lots_for_risk(
        equity,
        config.risk_pct,
        current_close,
        sizing_stop,
        config.instrument,
    )
    if lots <= 0:
        return None

    risk_amount = (
        risk_distance * lots * config.instrument.contract_size
        + config.commission_per_lot * lots
    )
    return Signal(
        action="buy",
        time=candles.index[-1],
        reference_price=current_close,
        ema_value=current_ema,
        atr_value=current_atr,
        stop_price=stop_price,
        lots=lots,
        risk_amount=risk_amount,
    )
