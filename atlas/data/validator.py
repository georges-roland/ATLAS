"""Contrôle qualité des chandeliers. Aucun backtest ne doit tourner sur des
données qui n'ont pas passé ce contrôle.

Détection des bougies fabriquées (cas d'un fichier "rempli" par une IA) :
- bougies un samedi, ou un dimanche avant 21:00 : l'or ne se négocie pas ;
- bougies plates (open = high = low = close) et bougies copiées de la
  précédente : bloquant au-delà de 1 % des lignes (si au moins 500 lignes).
Le volume nul est une simple information.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd

Gap = Tuple[pd.Timestamp, pd.Timestamp, bool]  # (début, fin, est_un_week_end)

MIN_ROWS_FOR_RATIO_CHECK = 500
MAX_SUSPECT_RATIO = 0.01  # 1 %
SUNDAY_OPEN_HOUR = 21  # une bougie le dimanche avant cette heure est suspecte


@dataclass
class ValidationReport:
    n_rows: int
    missing_values: int = 0
    non_positive: int = 0
    invalid_ohlc: int = 0
    duplicates: int = 0
    non_monotonic: int = 0
    gaps: List[Gap] = field(default_factory=list)
    weekend_candles: int = 0
    flat_candles: int = 0
    repeated_candles: int = 0
    zero_volume: int = 0

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

    def _ratio_exceeded(self, count: int) -> bool:
        return (
            self.n_rows >= MIN_ROWS_FOR_RATIO_CHECK
            and count / self.n_rows > MAX_SUSPECT_RATIO
        )

    @property
    def fabricated_suspected(self) -> bool:
        """True si le fichier contient des signes de bougies inventées."""
        return (
            self.weekend_candles > 0
            or self._ratio_exceeded(self.flat_candles)
            or self._ratio_exceeded(self.repeated_candles)
        )

    @property
    def ok(self) -> bool:
        """True si aucune erreur bloquante. Les trous, le volume nul et quelques
        bougies plates/copiées isolées sont des avertissements."""
        return self.errors == 0 and not self.fabricated_suspected

    def summary(self) -> str:
        lines = [
            f"Lignes               : {self.n_rows}",
            f"Valeurs manquantes   : {self.missing_values}",
            f"Prix <= 0            : {self.non_positive}",
            f"OHLC incohérents     : {self.invalid_ohlc}",
            f"Doublons de date     : {self.duplicates}",
            f"Dates non croissantes: {self.non_monotonic}",
            f"Bougies de week-end  : {self.weekend_candles} (bloquant si > 0)",
            f"Bougies plates       : {self.flat_candles} (bloquant si > 1 % des lignes, dès 500 lignes)",
            f"Bougies copiées      : {self.repeated_candles} (bloquant si > 1 % des lignes, dès 500 lignes)",
            f"Volume nul           : {self.zero_volume} (information)",
            f"Trous hors week-end  : {len(self.unexpected_gaps)} (avertissement)",
            f"RESULTAT             : {'OK' if self.ok else 'ERREURS A CORRIGER'}",
        ]
        if self.fabricated_suspected:
            lines.append(
                "-> Ce fichier contient des bougies fabriquées (week-end ou copies). "
                "Ne l'utilise pas : réexporte l'historique brut depuis MT5 ou Dukascopy."
            )
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

    # Signes de bougies fabriquées
    dow = idx.day_of_week
    hour = idx.hour
    report.weekend_candles = int(
        ((dow == 5) | ((dow == 6) & (hour < SUNDAY_OPEN_HOUR))).sum()
    )
    report.flat_candles = int(
        ((prices["open"] == prices["high"])
         & (prices["high"] == prices["low"])
         & (prices["low"] == prices["close"])).sum()
    )
    report.repeated_candles = int(prices.eq(prices.shift(1)).all(axis=1).sum())
    if "volume" in df.columns and (df["volume"] > 0).any():
        report.zero_volume = int((df["volume"] == 0).sum())

    # Trous : seulement sur les dates déjà triées et uniques
    clean_idx = idx.drop_duplicates().sort_values()
    if len(clean_idx) > 1:
        deltas = clean_idx[1:] - clean_idx[:-1]
        limit = pd.Timedelta(minutes=timeframe_minutes * max_gap_multiple)
        for start, end, delta in zip(clean_idx[:-1], clean_idx[1:], deltas):
            if delta > limit:
                report.gaps.append((start, end, _is_weekend_gap(start, end)))

    return report
