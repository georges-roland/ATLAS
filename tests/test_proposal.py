import pandas as pd
import pytest

from atlas.proposal import (
    ProposalStatus,
    create_proposal,
    decide_proposal,
)
from atlas.signal import Signal, SignalConfig


def make_signal(action="buy"):
    return Signal(
        action=action,
        time=pd.Timestamp("2024-01-02 08:00"),
        reference_price=100.0,
        ema_value=99.0,
        atr_value=4.0,
    )


def make_config():
    return SignalConfig(
        ema_span=2,
        atr_period=2,
        sl_mult=1.5,
        risk_pct=1.0,
        spread=0.3,
        slippage=0.05,
        commission_per_lot=7.0,
    )


def test_proposal_recalculates_stop_from_execution_price():
    proposal = create_proposal(make_signal(), entry_price=110.0, equity=10_000, config=make_config())
    assert proposal is not None
    assert proposal.reference_price == 100.0
    assert proposal.entry_price == 110.0
    assert proposal.stop_price == pytest.approx(104.0)


def test_proposal_sizes_lots_and_reports_cost_inclusive_risk():
    proposal = create_proposal(make_signal(), entry_price=110.0, equity=10_000, config=make_config())
    assert proposal is not None
    assert proposal.lots == pytest.approx(0.15)
    assert proposal.risk_amount == pytest.approx(97.05)


def test_proposal_is_pending_until_human_decision():
    proposal = create_proposal(make_signal(), entry_price=110.0, equity=10_000, config=make_config())
    assert proposal is not None
    assert proposal.status is ProposalStatus.PENDING


def test_approval_is_explicit_and_does_not_mutate_pending_proposal():
    proposal = create_proposal(make_signal(), entry_price=110.0, equity=10_000, config=make_config())
    assert proposal is not None
    approved = decide_proposal(proposal, "approve")
    assert approved.status is ProposalStatus.APPROVED
    assert proposal.status is ProposalStatus.PENDING


def test_rejection_is_explicit():
    proposal = create_proposal(make_signal(), entry_price=110.0, equity=10_000, config=make_config())
    assert proposal is not None
    assert decide_proposal(proposal, "reject").status is ProposalStatus.REJECTED


def test_proposal_is_skipped_below_broker_minimum_lot():
    assert create_proposal(make_signal(), entry_price=110.0, equity=100, config=make_config()) is None


def test_close_signal_cannot_create_entry_proposal():
    with pytest.raises(ValueError, match="signal d'achat"):
        create_proposal(make_signal("close_long"), entry_price=110.0, equity=10_000, config=make_config())


def test_finalized_proposal_cannot_be_decided_twice():
    proposal = create_proposal(make_signal(), entry_price=110.0, equity=10_000, config=make_config())
    assert proposal is not None
    approved = decide_proposal(proposal, "approve")
    with pytest.raises(ValueError, match="déjà reçu"):
        decide_proposal(approved, "reject")


def test_unknown_decision_is_rejected():
    proposal = create_proposal(make_signal(), entry_price=110.0, equity=10_000, config=make_config())
    assert proposal is not None
    with pytest.raises(ValueError, match="decision doit être"):
        decide_proposal(proposal, "maybe")  # type: ignore[arg-type]
