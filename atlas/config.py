"""Configuration centrale : specs de l'instrument et paramètres de risque.

Les valeurs XAU/USD ci-dessous sont des valeurs de référence courantes sur MT5
(1 lot = 100 oz). VERIFIE-LES avec les specs de TON broker avant tout usage réel.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    contract_size: float   # unités par lot (XAU/USD : 100 oz)
    min_lot: float
    lot_step: float
    max_lot: float


XAUUSD = InstrumentSpec(
    symbol="XAU/USD",
    contract_size=100.0,
    min_lot=0.01,
    lot_step=0.01,
    max_lot=100.0,
)

# Risque maximum par trade, en % du capital.
MAX_RISK_PER_TRADE_PCT = 1.0
