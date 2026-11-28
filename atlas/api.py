"""API locale authentifiée pour consulter et valider des propositions ATLAS."""

from dataclasses import dataclass
import os
import secrets
from threading import RLock
from typing import Callable, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from atlas.mt5_live import LiveEvaluation, evaluate_mt5
from atlas.proposal import ProposalStatus, TradeProposal, decide_proposal


class ValidationRequest(BaseModel):
    proposal_id: str
    decision: Literal["approve", "reject"]


class KillSwitchRequest(BaseModel):
    active: bool


@dataclass
class _ApiState:
    kill_switch_active: bool = True
    signal_key: Optional[str] = None
    proposal_id: Optional[str] = None
    proposal: Optional[TradeProposal] = None


def _default_evaluator() -> LiveEvaluation:
    costs_verified = os.environ.get("ATLAS_COSTS_VERIFIED", "").lower() in {
        "1",
        "true",
        "yes",
    }
    commission_raw = os.environ.get("ATLAS_COMMISSION_PER_LOT_ACCOUNT_CURRENCY")
    commission = None
    if commission_raw is not None:
        try:
            commission = float(commission_raw)
        except ValueError as exc:
            raise RuntimeError("La commission configurée est invalide") from exc
    return evaluate_mt5(
        costs_verified=costs_verified,
        commission_per_lot_account_currency=commission,
    )


def create_app(
    *,
    token: Optional[str] = None,
    evaluator: Callable[[], LiveEvaluation] = _default_evaluator,
) -> FastAPI:
    """Construit l'API ; le token de production vient de ATLAS_API_TOKEN."""
    app = FastAPI(title="ATLAS API", version="1.0.0")
    app.state.api_state = _ApiState()
    app.state.api_lock = RLock()
    security = HTTPBearer(auto_error=False)

    def require_auth(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    ) -> None:
        expected = token if token is not None else os.environ.get("ATLAS_API_TOKEN")
        if not expected:
            raise HTTPException(status_code=503, detail="Authentification API non configurée")
        if (
            credentials is None
            or credentials.scheme.lower() != "bearer"
            or not secrets.compare_digest(credentials.credentials, expected)
        ):
            raise HTTPException(
                status_code=401,
                detail="Authentification requise",
                headers={"WWW-Authenticate": "Bearer"},
            )

    router = APIRouter(dependencies=[Depends(require_auth)])

    def read_evaluation() -> LiveEvaluation:
        try:
            return evaluator()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Lecture MT5 indisponible") from exc

    @router.get("/state")
    def get_state():
        result = read_evaluation()
        state: _ApiState = app.state.api_state
        with app.state.api_lock:
            return {
                "server": result.server,
                "symbol": result.symbol,
                "account_currency": result.account_currency,
                "equity": result.equity,
                "position_count": result.position_count,
                "spread": result.spread,
                "tick_time_utc": result.tick_time_utc.isoformat(),
                "closed_candle_count": result.closed_candle_count,
                "last_closed_candle_utc": result.last_closed_candle_utc.isoformat(),
                "kill_switch_active": state.kill_switch_active,
            }

    @router.get("/signal")
    def get_signal():
        result = read_evaluation()
        state: _ApiState = app.state.api_state
        signal = result.signal
        signal_key = (
            f"{signal.action}:{signal.time.isoformat()}" if signal is not None else None
        )
        signal_payload = None
        if signal is not None:
            signal_payload = {
                "action": signal.action,
                "time_utc": signal.time.isoformat(),
                "reference_price": signal.reference_price,
                "ema_value": signal.ema_value,
                "atr_value": signal.atr_value,
                "stop_price": signal.stop_price,
            }

        with app.state.api_lock:
            if state.kill_switch_active:
                state.signal_key = signal_key
                state.proposal_id = None
                state.proposal = None
                proposal_payload = None
                blocked_reason = "Kill switch actif ; proposition bloquée."
            elif signal_key is None:
                state.signal_key = None
                state.proposal_id = None
                state.proposal = None
                proposal_payload = None
                blocked_reason = result.blocked_reason
            elif signal is None or signal.action != "buy" or result.proposal is None:
                state.signal_key = signal_key
                state.proposal_id = None
                state.proposal = None
                proposal_payload = None
                blocked_reason = result.blocked_reason
            else:
                if state.signal_key != signal_key:
                    state.signal_key = signal_key
                    state.proposal_id = uuid4().hex
                    state.proposal = result.proposal
                elif state.proposal is None:
                    state.proposal_id = uuid4().hex
                    state.proposal = result.proposal
                proposal_payload = _proposal_payload(state.proposal_id, state.proposal)
                blocked_reason = None

            return {
                "signal": signal_payload,
                "proposal": proposal_payload,
                "blocked_reason": blocked_reason,
                "kill_switch_active": state.kill_switch_active,
            }

    @router.post("/validation")
    def validate_proposal(request: ValidationRequest):
        state: _ApiState = app.state.api_state
        with app.state.api_lock:
            if state.kill_switch_active:
                raise HTTPException(status_code=409, detail="Kill switch actif")
            if (
                state.proposal is None
                or state.proposal_id != request.proposal_id
                or state.proposal.status is not ProposalStatus.PENDING
            ):
                raise HTTPException(status_code=409, detail="Proposition absente ou déjà décidée")
            try:
                state.proposal = decide_proposal(state.proposal, request.decision)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Décision invalide") from exc
            return _proposal_payload(state.proposal_id, state.proposal)

    @router.post("/kill-switch")
    def set_kill_switch(request: KillSwitchRequest):
        state: _ApiState = app.state.api_state
        with app.state.api_lock:
            state.kill_switch_active = request.active
            if request.active:
                state.signal_key = None
                state.proposal_id = None
                state.proposal = None
            return {"kill_switch_active": state.kill_switch_active}

    app.include_router(router)
    return app


def _proposal_payload(
    proposal_id: Optional[str], proposal: Optional[TradeProposal]
) -> Optional[dict]:
    if proposal_id is None or proposal is None:
        return None
    return {
        "proposal_id": proposal_id,
        "status": proposal.status.value,
        "signal_time_utc": proposal.signal_time.isoformat(),
        "reference_price": proposal.reference_price,
        "entry_price": proposal.entry_price,
        "stop_price": proposal.stop_price,
        "lots": proposal.lots,
        "risk_amount": proposal.risk_amount,
        "risk_currency": proposal.risk_currency,
        "equity": proposal.equity,
    }


app = create_app()
