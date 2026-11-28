"""Validation hors échantillon : choisir les réglages sur une période, les
vérifier sur la période suivante, jamais l'inverse.

Chaque période est backtestée séparément (les indicateurs repartent de zéro).
La grille de réglages est volontairement petite : plus on teste de
combinaisons, plus on risque de "trouver" un bon réglage par hasard.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd

from atlas.backtest import BacktestConfig, DataValidationError, run_backtest
from atlas.data import load_candles, validate_candles

GRID_EMAS = (10, 20, 50)
GRID_SL_MULTS = (1.0, 1.5, 2.0)
MIN_TRADES = 30


@dataclass(frozen=True)
class ParamRow:
    ema_span: int
    sl_mult: float
    train_profit: float
    train_trades: int
    test_profit: float
    test_trades: int


def select_best(rows: Sequence[ParamRow], min_trades: int = MIN_TRADES) -> Optional[ParamRow]:
    """Meilleur réglage de la période de CHOIX, parmi ceux qui ont assez de trades."""
    eligible = [r for r in rows if r.train_trades >= min_trades]
    return max(eligible, key=lambda r: r.train_profit) if eligible else None


@dataclass
class TrainTestStudy:
    train_end: str
    train_rows: int
    test_rows: int
    rows: List[ParamRow]
    test_buy_hold_profit: float
    min_trades: int = MIN_TRADES

    @property
    def best(self) -> Optional[ParamRow]:
        return select_best(self.rows, self.min_trades)

    @property
    def profitable_in_test(self) -> int:
        return sum(1 for r in self.rows if r.test_profit > 0)

    def verdict(self) -> str:
        b = self.best
        if b is None:
            return (
                f"Aucun réglage n'a {self.min_trades} trades sur la période de choix : "
                "rien à conclure."
            )
        if b.test_trades < self.min_trades:
            return f"Moins de {self.min_trades} trades en vérification : résultat non significatif."
        if b.test_profit <= 0:
            return "ALERTE : le meilleur réglage perd hors échantillon (signe de sur-optimisation)."
        if b.test_profit <= self.test_buy_hold_profit:
            return "Rentable hors échantillon, mais fait moins bien que l'achat-conservation."
        return (
            "Rentable hors échantillon et meilleur que l'achat-conservation. "
            "Ce n'est qu'une première vérification : à confirmer sur d'autres périodes, puis en démo."
        )

    def summary(self) -> str:
        lines = [
            "--- Validation hors échantillon ---",
            f"Période de choix        : jusqu'au {self.train_end} ({self.train_rows} bougies)",
            f"Période de vérification : après ({self.test_rows} bougies)",
            "",
            "EMA  Stop(xATR)  Choix: profit / trades   Vérif.: profit / trades",
        ]
        for r in self.rows:
            lines.append(
                f"{r.ema_span:<4} {r.sl_mult:<11} {r.train_profit:>10,.0f} $ / {r.train_trades:<6} "
                f"{r.test_profit:>10,.0f} $ / {r.test_trades}"
            )
        b = self.best
        lines.append("")
        if b:
            lines.append(
                f"Meilleur réglage au choix : EMA {b.ema_span}, stop {b.sl_mult} x ATR "
                f"({b.train_profit:,.0f} $)"
            )
            lines.append(
                f"Le même réglage en vérification : {b.test_profit:,.0f} $ ({b.test_trades} trades)"
            )
        lines.append(
            f"Combinaisons rentables en vérification : {self.profitable_in_test}/{len(self.rows)}"
        )
        lines.append(f"Achat-conservation en vérification : {self.test_buy_hold_profit:,.0f} $")
        lines.append("")
        lines.append(self.verdict())
        return "\n".join(lines)


def train_test_study(
    df: pd.DataFrame,
    train_end: str,
    base_cfg: Optional[BacktestConfig] = None,
    emas: Sequence[int] = GRID_EMAS,
    sl_mults: Sequence[float] = GRID_SL_MULTS,
    min_trades: int = MIN_TRADES,
) -> TrainTestStudy:
    """Sépare df à la date train_end (incluse dans la période de choix)."""
    base_cfg = base_cfg or BacktestConfig()
    cut = pd.Timestamp(train_end).normalize() + pd.Timedelta(days=1)
    train = df[df.index < cut]
    test = df[df.index >= cut]
    if train.empty or test.empty:
        raise ValueError(
            f"Séparation impossible à la date {train_end} : une des deux périodes est vide."
        )

    rows: List[ParamRow] = []
    bh_profit = 0.0
    for i, (ema_span, sl_mult) in enumerate((e, s) for e in emas for s in sl_mults):
        cfg = replace(base_cfg, ema_span=ema_span, sl_mult=sl_mult)
        tr = run_backtest(train, cfg)
        te = run_backtest(test, cfg)
        if i == 0:
            bh_profit = te.buy_hold_final_equity - base_cfg.initial_capital
        rows.append(
            ParamRow(ema_span, sl_mult, tr.net_profit, tr.n_trades, te.net_profit, te.n_trades)
        )
    return TrainTestStudy(
        train_end=train_end,
        train_rows=len(train),
        test_rows=len(test),
        rows=rows,
        test_buy_hold_profit=bh_profit,
        min_trades=min_trades,
    )


def train_test_from_csv(
    path: Union[str, Path],
    train_end: str,
    base_cfg: Optional[BacktestConfig] = None,
    timeframe_minutes: int = 60,
    **kwargs,
) -> TrainTestStudy:
    """Charge, VALIDE toutes les données, puis lance l'étude."""
    df = load_candles(path)
    report = validate_candles(df, timeframe_minutes=timeframe_minutes)
    if not report.ok:
        raise DataValidationError("Données invalides :\n" + report.summary())
    return train_test_study(df, train_end, base_cfg, **kwargs)
