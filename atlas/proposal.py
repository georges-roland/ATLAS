"""Prépare et fait valider une proposition de trade, sans envoyer d'ordre."""

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Callable, Literal, Optional

import pandas as pd

from atlas.risk import lots_for_risk
from atlas.signal import Signal, SignalConfig


class ProposalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class TradeProposal:
    """Proposition calculée à un prix d'exécution fourni, en attente d'un humain."""

    signal_time: pd.Timestamp
    reference_price: float
    entry_price: float
    stop_price: float
    lots: float
    risk_amount: float
    equity: float
    risk_currency: str = "USD"
    status: ProposalStatus = ProposalStatus.PENDING


def create_proposal(
    signal: Signal,
    entry_price: float,
    equity: float,
    config: SignalConfig = SignalConfig(),
) -> Optional[TradeProposal]:
    """Recalcule stop et lots au prix fourni, sans créer ni envoyer d'ordre.

    ``entry_price`` doit être le prix disponible au moment de préparer
    l'exécution, et non le prix de référence du signal à la clôture précédente.
    Une proposition sous le lot minimum est ignorée.
    """
    if signal.action != "buy":
        raise ValueError("seul un signal d'achat peut créer une proposition")
    if not math.isfinite(entry_price) or entry_price <= 0:
        raise ValueError("entry_price doit être fini et > 0")
    if not math.isfinite(equity) or equity <= 0:
        raise ValueError("equity doit être fini et > 0")

    stop_distance = config.sl_mult * signal.atr_value
    risk_distance = stop_distance + config.spread + 2.0 * config.slippage
    stop_price = entry_price - stop_distance
    sizing_stop = entry_price - risk_distance
    if stop_price <= 0 or sizing_stop <= 0:
        return None

    lots = lots_for_risk(
        equity,
        config.risk_pct,
        entry_price,
        sizing_stop,
        config.instrument,
    )
    if lots <= 0:
        return None

    risk_amount = (
        risk_distance * lots * config.instrument.contract_size
        + config.commission_per_lot * lots
    )
    return TradeProposal(
        signal_time=signal.time,
        reference_price=signal.reference_price,
        entry_price=entry_price,
        stop_price=stop_price,
        lots=lots,
        risk_amount=risk_amount,
        equity=equity,
    )


def create_proposal_in_account_currency(
    signal: Signal,
    entry_price: float,
    equity: float,
    *,
    profit_at_stop: Callable[[float, float, float], Optional[float]],
    volume_min: float,
    volume_step: float,
    volume_max: float,
    commission_per_lot: float,
    risk_currency: str,
    config: SignalConfig = SignalConfig(),
) -> Optional[TradeProposal]:
    """Crée une proposition avec un calcul MT5 du P/L dans la devise du compte.

    ``profit_at_stop`` reçoit volume, prix d'entrée et prix de sortie de risque ;
    il doit renvoyer le P/L dans la devise du compte (donc négatif au stop).
    La commission fournie est un coût total par lot, dans cette même devise.
    """
    if signal.action != "buy":
        raise ValueError("seul un signal d'achat peut créer une proposition")
    if not math.isfinite(entry_price) or entry_price <= 0:
        raise ValueError("entry_price doit être fini et > 0")
    if not math.isfinite(equity) or equity <= 0:
        raise ValueError("equity doit être finie et > 0")
    if not math.isfinite(signal.atr_value) or signal.atr_value <= 0:
        raise ValueError("atr_value doit être finie et > 0")
    if not risk_currency:
        raise ValueError("risk_currency ne peut pas être vide")
    if not math.isfinite(commission_per_lot) or commission_per_lot < 0:
        raise ValueError("commission_per_lot doit être finie et >= 0")
    volume_values = (volume_min, volume_step, volume_max)
    if any(not math.isfinite(value) or value <= 0 for value in volume_values):
        raise ValueError("les limites de volume doivent être finies et > 0")
    if volume_min > volume_max:
        raise ValueError("volume_min ne peut pas dépasser volume_max")

    stop_distance = config.sl_mult * signal.atr_value
    stop_price = entry_price - stop_distance
    sizing_stop = entry_price - (
        stop_distance + config.spread + 2.0 * config.slippage
    )
    if stop_price <= 0 or sizing_stop <= 0:
        return None

    def loss_for_volume(volume: float) -> float:
        profit = profit_at_stop(volume, entry_price, sizing_stop)
        if profit is None or not math.isfinite(float(profit)) or profit >= 0:
            raise ValueError("Calcul MT5 du risque impossible ou non négatif.")
        return -float(profit) + commission_per_lot * volume

    risk_per_lot = loss_for_volume(volume_min) / volume_min
    if risk_per_lot <= 0:
        raise ValueError("Le risque par lot doit être positif.")
    risk_budget = equity * config.risk_pct / 100.0
    affordable_lots = risk_budget / risk_per_lot
    if affordable_lots + 1e-9 < volume_min:
        return None
    steps_by_budget = math.floor(
        (affordable_lots - volume_min) / volume_step + 1e-9
    )
    steps_by_max = math.floor((volume_max - volume_min) / volume_step + 1e-9)
    lots = round(volume_min + min(steps_by_budget, steps_by_max) * volume_step, 8)

    risk_amount = loss_for_volume(lots)
    while risk_amount > risk_budget + 1e-9 and lots - volume_step >= volume_min:
        lots = round(lots - volume_step, 8)
        risk_amount = loss_for_volume(lots)
    if risk_amount > risk_budget + 1e-9:
        return None

    return TradeProposal(
        signal_time=signal.time,
        reference_price=signal.reference_price,
        entry_price=entry_price,
        stop_price=stop_price,
        lots=lots,
        risk_amount=risk_amount,
        equity=equity,
        risk_currency=risk_currency,
    )


def decide_proposal(
    proposal: TradeProposal,
    decision: Literal["approve", "reject"],
) -> TradeProposal:
    """Enregistre une décision humaine explicite ; n'exécute aucun ordre."""
    if proposal.status is not ProposalStatus.PENDING:
        raise ValueError("la proposition a déjà reçu une décision")
    if decision == "approve":
        status = ProposalStatus.APPROVED
    elif decision == "reject":
        status = ProposalStatus.REJECTED
    else:
        raise ValueError("decision doit être 'approve' ou 'reject'")
    return replace(proposal, status=status)
