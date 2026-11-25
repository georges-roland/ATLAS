"""Contrôle qualité des chandeliers. Aucun backtest ne doit tourner sur des
données qui n'ont pas passé ce contrôle."""

from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd

Gap = Tuple[pd.Timestamp, pd.Timestamp, bool]  # (début, fin, est_un_week_end)


@dataclass
class ValidationReport:
    n_rows: int
    missing_values: int = 0
    non_positive: int = 0
    invalid_ohlc: int = 0
    duplicates: int = 0
    non_monotonic: int = 0
    gaps: List[Gap] = field(default_factory=list)

    @property
    def unexpected_gaps(self) -> List[Gap]:
        return [g for g in self.gaps if not g[2]]

    @property
    def errors(self) -> int:
        return (
            self.missing_values
            + self.non_positive
            + self.invalid_ohlc
            + self.duplicates
            + self.non_monotonic
        )

    @property
    def ok(self) -> bool:
        """True si aucune erreur bloquante (les trous sont des avertissements)."""
        return self.errors == 0

    def summary(self) -> str:
        lines = [
            f"Lignes               : {self.n_rows}",
            f"Valeurs manquantes   : {self.missing_values}",
            f"Prix <= 0            : {self.non_positive}",
            f"OHLC incohérents     : {self.invalid_ohlc}",
            f"Doublons de date     : {self.duplicates}",
            f"Dates non croissantes: {self.non_monotonic}",
            f"Trous hors week-end  : {len(self.unexpected_gaps)} (avertissement)",
            f"RESULTAT             : {'OK' if self.ok else 'ERREURS A CORRIGER'}",
        ]
        return "\n".join(lines)


def _is_weekend_gap(start: pd.Timestamp, end: pd.Timestamp) -> bool:
    # Vendredi/samedi -> dimanche/lundi
    return start.weekday() in (4, 5) and end.weekday() in (6, 0)


def validate_candles(
    df: pd.DataFrame, timeframe_minutes: int, max_gap_multiple: int = 3
) -> ValidationReport:
    """Vérifie un DataFrame de chandeliers (index = timestamp).

    max_gap_multiple : un trou est signalé si l'écart entre deux bougies
    dépasse max_gap_multiple x timeframe.
    """
    if timeframe_minutes <= 0:
        raise ValueError("timeframe_minutes doit être > 0")

    report = ValidationReport(n_rows=len(df))
    if df.empty:
        return report

    cols = ["open", "high", "low", "close"]
    prices = df[cols]

    report.missing_values = int(prices.isna().any(axis=1).sum())
    valid = prices.dropna()
    report.non_positive = int((valid <= 0).any(axis=1).sum())

    bad_ohlc = (
        (valid["high"] < valid["low"])
        | (valid["high"] < valid[["open", "close"]].max(axis=1))
        | (valid["low"] > valid[["open", "close"]].min(axis=1))
    )
    report.invalid_ohlc = int(bad_ohlc.sum())

    idx = df.index
    report.duplicates = int(idx.duplicated().sum())
    report.non_monotonic = int((idx[1:] < idx[:-1]).sum())

    # Trous : seulement sur les dates déjà triées et uniques
    clean_idx = idx.drop_duplicates().sort_values()
    if len(clean_idx) > 1:
        deltas = clean_idx[1:] - clean_idx[:-1]
        limit = pd.Timedelta(minutes=timeframe_minutes * max_gap_multiple)
        for start, end, delta in zip(clean_idx[:-1], clean_idx[1:], deltas):
            if delta > limit:
                report.gaps.append((start, end, _is_weekend_gap(start, end)))

    return report
