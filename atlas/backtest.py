"""Backtest ATLAS v2 : croisement prix / EMA, stop ATR, exécution réaliste.

Règles (toutes testées dans tests/test_backtest.py) :
- Le signal est calculé à la CLOTURE de la bougie t ; l'ordre est exécuté à
  l'OUVERTURE de la bougie t+1. Jamais au prix de clôture du signal.
- Stop (et TP) testés sur le high/low de chaque bougie. Si SL et TP sont
  touchés dans la même bougie, le SL gagne (hypothèse pessimiste).
- Gap : si la bougie ouvre au-delà du stop, on sort à l'ouverture (pire prix).
- Coûts : spread, slippage et commission. Les prix OHLC sont considérés comme
  des prix "milieu" : achat à +spread/2 +slippage, vente à -spread/2 -slippage.
- Taille de position = atlas.risk.lots_for_risk. Si 0 lot, le trade est ignoré.
- Un seul trade ouvert à la fois. Capital suivi bougie par bougie (vrai
  drawdown, position ouverte valorisée à la clôture, coûts de sortie inclus).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from atlas.config import XAUUSD, InstrumentSpec
from atlas.data import load_candles, validate_candles
from atlas.indicators import atr, ema
from atlas.risk import lots_for_risk


class DataValidationError(Exception):
    """Les données ne passent pas validate_candles : on n'exécute pas le backtest."""


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 10_000.0
    risk_pct: float = 1.0
    ema_span: int = 20
    atr_period: int = 14
    sl_mult: float = 1.5
    tp_mult: Optional[float] = None
    allow_short: bool = False  # True = AJOUTE les ventes (les achats restent)
    spread: float = 0.30  # en USD par once (écart bid/ask total)
    slippage: float = 0.05  # en USD par once, par exécution
    commission_per_lot: float = 7.0  # aller-retour, en USD par lot
    instrument: InstrumentSpec = XAUUSD


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str  # "long" ou "short"
    lots: float
    entry_price: float  # prix exécuté (spread et slippage inclus)
    exit_price: float  # prix exécuté
    exit_reason: str  # stop_loss, stop_loss_gap, take_profit, signal, end_of_data
    gross_pnl: float  # sur les prix "milieu", avant tout coût
    costs: float  # spread + slippage + commission
    net_pnl: float  # gross_pnl - costs


@dataclass
class _Position:
    side: int  # +1 long, -1 short
    lots: float
    raw_entry: float
    exec_entry: float
    entry_time: pd.Timestamp
    stop_raw: float
    tp_raw: Optional[float]


def max_drawdown(equity: pd.Series) -> Tuple[float, float]:
    """Drawdown maximum (en $, en %) d'une courbe de capital."""
    if equity.empty:
        return 0.0, 0.0
    running_peak = equity.cummax()
    dd = running_peak - equity
    dd_pct = dd / running_peak * 100.0
    return float(dd.max()), float(dd_pct.max())


@dataclass
class BacktestResult:
    config: BacktestConfig
    trades: List[Trade]
    equity_curve: pd.Series
    skipped_trades: int
    buy_hold_final_equity: float

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1]) if len(self.equity_curve) else self.config.initial_capital

    @property
    def net_profit(self) -> float:
        return self.final_equity - self.config.initial_capital

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.net_pnl > 0) / len(self.trades) * 100.0

    @property
    def profit_factor(self) -> float:
        wins = sum(t.net_pnl for t in self.trades if t.net_pnl > 0)
        losses = -sum(t.net_pnl for t in self.trades if t.net_pnl < 0)
        if losses == 0:
            return float("inf") if wins > 0 else 0.0
        return wins / losses

    @property
    def expectancy(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.net_pnl for t in self.trades) / len(self.trades)

    @property
    def total_costs(self) -> float:
        return sum(t.costs for t in self.trades)

    @property
    def avg_duration_hours(self) -> float:
        if not self.trades:
            return 0.0
        secs = [(t.exit_time - t.entry_time).total_seconds() for t in self.trades]
        return sum(secs) / len(secs) / 3600.0

    @property
    def drawdown(self) -> Tuple[float, float]:
        return max_drawdown(self.equity_curve)

    def summary(self) -> str:
        dd_usd, dd_pct = self.drawdown
        cap = self.config.initial_capital
        bh_profit = self.buy_hold_final_equity - cap
        lines = [
            "--- Backtest ATLAS v2 (avec coûts) ---",
            f"Capital initial   : ${cap:,.2f}",
            f"Capital final     : ${self.final_equity:,.2f}",
            f"Profit net        : ${self.net_profit:,.2f} ({self.net_profit / cap * 100:.1f} %)",
            f"Trades            : {self.n_trades} (ignorés faute de taille : {self.skipped_trades})",
            f"Win rate          : {self.win_rate:.1f} %",
            f"Profit factor     : {self.profit_factor:.2f}",
            f"Espérance / trade : ${self.expectancy:,.2f}",
            f"Drawdown max      : ${dd_usd:,.2f} ({dd_pct:.1f} %)",
            f"Durée moyenne     : {self.avg_duration_hours:.1f} h",
            f"Coûts payés       : ${self.total_costs:,.2f}",
            "--- Comparaison : achat-conservation (sans levier) ---",
            f"Capital final     : ${self.buy_hold_final_equity:,.2f}",
            f"Profit net        : ${bh_profit:,.2f} ({bh_profit / cap * 100:.1f} %)",
        ]
        return "\n".join(lines)


def run_backtest(df: pd.DataFrame, cfg: Optional[BacktestConfig] = None) -> BacktestResult:
    """Exécute le backtest sur un DataFrame de chandeliers (index = timestamp).

    Ne modifie pas df. Les données doivent déjà avoir passé validate_candles.
    """
    cfg = cfg or BacktestConfig()
    spec = cfg.instrument
    cs = spec.contract_size
    half = cfg.spread / 2.0
    slip = cfg.slippage
    friction_per_lot = (cfg.spread + 2.0 * slip) * cs + cfg.commission_per_lot

    n = len(df)
    index = df.index
    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    ema_arr = ema(df["close"], cfg.ema_span).to_numpy(dtype=float)
    atr_arr = atr(df, cfg.atr_period).to_numpy(dtype=float)
    warmup = max(cfg.ema_span, cfg.atr_period)

    equity = cfg.initial_capital  # capital réalisé (cash)
    trades: List[Trade] = []
    curve = np.full(n, cfg.initial_capital, dtype=float)
    skipped = 0

    pos: Optional[_Position] = None
    pending_exit = False
    pending_entry: Optional[Tuple[int, float]] = None  # (sens, distance du stop)

    def close_position(raw_exit: float, when: pd.Timestamp, reason: str) -> None:
        nonlocal pos, equity
        p = pos
        assert p is not None
        exec_exit = raw_exit - p.side * (half + slip)
        gross = (raw_exit - p.raw_entry) * p.side * p.lots * cs
        costs = friction_per_lot * p.lots
        net = gross - costs
        equity += net
        trades.append(
            Trade(
                entry_time=p.entry_time,
                exit_time=when,
                side="long" if p.side == 1 else "short",
                lots=p.lots,
                entry_price=p.exec_entry,
                exit_price=exec_exit,
                exit_reason=reason,
                gross_pnl=gross,
                costs=costs,
                net_pnl=net,
            )
        )
        pos = None

    for i in range(n):
        # 1) Ordres décidés à la clôture de la bougie précédente, exécutés à l'ouverture
        if pending_exit and pos is not None:
            close_position(open_[i], index[i], "signal")
        pending_exit = False

        if pending_entry is not None and pos is None:
            side, sl_dist = pending_entry
            raw_entry = open_[i]
            # Distance de risque réelle = stop ATR + spread + slippage des deux côtés
            risk_dist = sl_dist + cfg.spread + 2.0 * slip
            lots = lots_for_risk(
                equity, cfg.risk_pct, raw_entry, raw_entry - side * risk_dist, spec
            ) if raw_entry - side * risk_dist > 0 else 0.0
            if lots <= 0:
                skipped += 1
            else:
                stop_raw = raw_entry - side * sl_dist
                tp_raw = (
                    raw_entry + side * cfg.tp_mult * (sl_dist / cfg.sl_mult)
                    if cfg.tp_mult is not None and cfg.sl_mult > 0
                    else None
                )
                pos = _Position(
                    side=side,
                    lots=lots,
                    raw_entry=raw_entry,
                    exec_entry=raw_entry + side * (half + slip),
                    entry_time=index[i],
                    stop_raw=stop_raw,
                    tp_raw=tp_raw,
                )
        pending_entry = None

        # 2) Stop / TP sur la bougie courante (high/low), gap à l'ouverture
        if pos is not None:
            p = pos
            if p.side == 1:
                if open_[i] <= p.stop_raw:
                    close_position(open_[i], index[i], "stop_loss_gap")
                elif low[i] <= p.stop_raw:
                    close_position(p.stop_raw, index[i], "stop_loss")
                elif p.tp_raw is not None and high[i] >= p.tp_raw:
                    close_position(max(open_[i], p.tp_raw), index[i], "take_profit")
            else:
                if open_[i] >= p.stop_raw:
                    close_position(open_[i], index[i], "stop_loss_gap")
                elif high[i] >= p.stop_raw:
                    close_position(p.stop_raw, index[i], "stop_loss")
                elif p.tp_raw is not None and low[i] <= p.tp_raw:
                    close_position(min(open_[i], p.tp_raw), index[i], "take_profit")

        # 3) Fin des données : on ferme à la clôture
        if i == n - 1 and pos is not None:
            close_position(close[i], index[i], "end_of_data")

        # 4) Capital de la bougie (position ouverte valorisée comme si fermée à la clôture)
        if pos is not None:
            unreal = (close[i] - pos.raw_entry) * pos.side * pos.lots * cs - friction_per_lot * pos.lots
            curve[i] = equity + unreal
        else:
            curve[i] = equity

        # 5) Signaux calculés à la clôture de la bougie i (exécutés à l'ouverture de i+1)
        if i >= warmup and i < n - 1 and atr_arr[i] > 0:
            crossed_up = close[i - 1] <= ema_arr[i - 1] and close[i] > ema_arr[i]
            crossed_down = close[i - 1] >= ema_arr[i - 1] and close[i] < ema_arr[i]
            sl_dist = cfg.sl_mult * atr_arr[i]

            if pos is not None:
                if (pos.side == 1 and crossed_down) or (pos.side == -1 and crossed_up):
                    pending_exit = True

            will_be_flat = pos is None or pending_exit
            if will_be_flat:
                if crossed_up:
                    pending_entry = (1, sl_dist)
                elif crossed_down and cfg.allow_short:
                    pending_entry = (-1, sl_dist)

    # Comparaison : achat-conservation sans levier, même capital
    first_open, last_close = open_[0], close[-1]
    bh_final = cfg.initial_capital * (last_close / first_open) if n else cfg.initial_capital

    return BacktestResult(
        config=cfg,
        trades=trades,
        equity_curve=pd.Series(curve, index=index, name="equity"),
        skipped_trades=skipped,
        buy_hold_final_equity=float(bh_final),
    )


def backtest_from_csv(
    path: Union[str, Path],
    cfg: Optional[BacktestConfig] = None,
    timeframe_minutes: int = 60,
) -> BacktestResult:
    """Charge, VALIDE puis backtest. Refuse de tourner si les données sont invalides."""
    df = load_candles(path)
    report = validate_candles(df, timeframe_minutes=timeframe_minutes)
    if not report.ok:
        raise DataValidationError("Données invalides :\n" + report.summary())
    return run_backtest(df, cfg)
