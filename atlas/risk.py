"""Gestion du risque : taille de position en LOTS.

Règle d'or : on arrondit TOUJOURS vers le bas. Si le résultat est sous le lot
minimum du broker, on renvoie 0 (pas de trade) au lieu de grossir la position :
le risque réel ne doit jamais dépasser le budget.
"""

import math

from atlas.config import InstrumentSpec


def risk_per_lot(entry: float, stop: float, spec: InstrumentSpec) -> float:
    """Perte en $ pour 1 lot si le stop est touché (sans frais)."""
    return abs(entry - stop) * spec.contract_size


def risk_amount(lots: float, entry: float, stop: float, spec: InstrumentSpec) -> float:
    """Perte en $ pour une taille donnée si le stop est touché (sans frais)."""
    return lots * risk_per_lot(entry, stop, spec)


def lots_for_risk(
    equity: float,
    risk_pct: float,
    entry: float,
    stop: float,
    spec: InstrumentSpec,
) -> float:
    """Taille maximale en lots dont la perte au stop reste <= risk_pct du capital."""
    if equity <= 0:
        raise ValueError("equity doit être > 0")
    if not 0 < risk_pct <= 100:
        raise ValueError("risk_pct doit être dans ]0, 100]")
    if entry <= 0 or stop <= 0:
        raise ValueError("entry et stop doivent être > 0")
    if entry == stop:
        raise ValueError("stop identique au prix d'entrée")

    budget = equity * risk_pct / 100.0
    raw_lots = budget / risk_per_lot(entry, stop, spec)
    raw_lots = min(raw_lots, spec.max_lot)

    # Arrondi vers le bas au lot_step (petite tolérance pour les flottants)
    steps = math.floor(raw_lots / spec.lot_step + 1e-9)
    lots = round(steps * spec.lot_step, 8)

    if lots < spec.min_lot:
        return 0.0
    return lots

# Convenience wrapper used by the back‑test script.
# Mirrors the original expected name ``lot_size_for_trade``.
# Uses the XAUUSD spec by default.
from atlas.config import XAUUSD

def lot_size_for_trade(equity: float, risk_pct: float, entry: float, stop: float, spec: XAUUSD = XAUUSD) -> float:
    """Calculate position size (lots) given equity, risk %, entry price and stop price.
    Thin wrapper around :func:`lots_for_risk` to preserve the historical API.
    """
    return lots_for_risk(equity, risk_pct, entry, stop, spec)
