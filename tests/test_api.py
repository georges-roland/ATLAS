from datetime import datetime, timezone

from fastapi.testclient import TestClient
import pandas as pd

from atlas.api import create_app
from atlas.mt5_live import LiveEvaluation
from atlas.proposal import ProposalStatus, TradeProposal
from atlas.signal import Signal


TOKEN = "test-only-bearer-token"


def make_evaluation():
    now = pd.Timestamp("2026-11-27 12:00:00", tz="UTC")
    signal = Signal(
        action="buy",
        time=now,
        reference_price=2700.0,
        ema_value=2698.0,
        atr_value=10.0,
        stop_price=2685.0,
    )
    proposal = TradeProposal(
        signal_time=now,
        reference_price=2700.0,
        entry_price=2701.0,
        stop_price=2686.0,
        lots=0.02,
        risk_amount=30.0,
        equity=3000.0,
        risk_currency="EUR",
    )
    return LiveEvaluation(
        server="Exness-Demo",
        symbol="XAUUSDm",
        account_currency="EUR",
        equity=3000.0,
        position_count=0,
        spread=0.2,
        tick_time_utc=pd.Timestamp("2026-11-27 13:00:00", tz="UTC"),
        closed_candle_count=300,
        last_closed_candle_utc=now,
        signal=signal,
        proposal=proposal,
        blocked_reason=None,
    )


class FakeEvaluator:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return make_evaluation()


def make_client(*, token=TOKEN, evaluator=None):
    app = create_app(token=token, evaluator=evaluator or FakeEvaluator())
    return TestClient(app)


def auth_headers():
    return {"Authorization": f"Bearer {TOKEN}"}


def test_all_api_routes_require_valid_bearer_authentication():
    client = make_client()

    assert client.get("/state").status_code == 401
    assert client.get("/signal", headers={"Authorization": "Bearer wrong"}).status_code == 401
    response = client.get("/state", headers=auth_headers())

    assert response.status_code == 200
    assert response.json()["account_currency"] == "EUR"


def test_api_fails_closed_when_token_is_not_configured():
    response = make_client(token="").get("/state", headers=auth_headers())

    assert response.status_code == 503
    assert response.json()["detail"] == "Authentification API non configurée"


def test_kill_switch_starts_active_and_blocks_proposals():
    client = make_client()

    response = client.get("/signal", headers=auth_headers())

    assert response.status_code == 200
    assert response.json()["kill_switch_active"] is True
    assert response.json()["signal"]["action"] == "buy"
    assert response.json()["proposal"] is None
    assert response.json()["blocked_reason"] == "Kill switch actif ; proposition bloquée."


def test_proposal_requires_switch_off_and_human_decision_is_only_a_status_change():
    evaluator = FakeEvaluator()
    client = make_client(evaluator=evaluator)

    enabled = client.post("/kill-switch", headers=auth_headers(), json={"active": False})
    assert enabled.status_code == 200
    first = client.get("/signal", headers=auth_headers()).json()["proposal"]
    second = client.get("/signal", headers=auth_headers()).json()["proposal"]
    assert first["status"] == ProposalStatus.PENDING.value
    assert first["proposal_id"] == second["proposal_id"]

    approved = client.post(
        "/validation",
        headers=auth_headers(),
        json={"proposal_id": first["proposal_id"], "decision": "approve"},
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == ProposalStatus.APPROVED.value
    assert evaluator.calls == 2

    repeated = client.get("/signal", headers=auth_headers()).json()["proposal"]
    assert repeated["status"] == ProposalStatus.APPROVED.value


def test_kill_switch_invalidates_pending_proposal_and_blocks_validation():
    client = make_client()
    client.post("/kill-switch", headers=auth_headers(), json={"active": False})
    proposal = client.get("/signal", headers=auth_headers()).json()["proposal"]

    switched = client.post("/kill-switch", headers=auth_headers(), json={"active": True})
    assert switched.json()["kill_switch_active"] is True
    response = client.post(
        "/validation",
        headers=auth_headers(),
        json={"proposal_id": proposal["proposal_id"], "decision": "approve"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Kill switch actif"


def test_validation_rejects_unknown_or_already_decided_proposals():
    client = make_client()
    client.post("/kill-switch", headers=auth_headers(), json={"active": False})
    proposal = client.get("/signal", headers=auth_headers()).json()["proposal"]

    missing = client.post(
        "/validation",
        headers=auth_headers(),
        json={"proposal_id": "wrong-id", "decision": "approve"},
    )
    assert missing.status_code == 409

    rejected = client.post(
        "/validation",
        headers=auth_headers(),
        json={"proposal_id": proposal["proposal_id"], "decision": "reject"},
    )
    assert rejected.status_code == 200
    assert rejected.json()["status"] == ProposalStatus.REJECTED.value

    repeated = client.post(
        "/validation",
        headers=auth_headers(),
        json={"proposal_id": proposal["proposal_id"], "decision": "approve"},
    )
    assert repeated.status_code == 409


def test_mt5_read_failures_do_not_leak_internal_details():
    def failed_evaluator():
        raise RuntimeError("private terminal detail")

    response = make_client(evaluator=failed_evaluator).get(
        "/state", headers=auth_headers()
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Lecture MT5 indisponible"
    assert "private terminal detail" not in response.text
