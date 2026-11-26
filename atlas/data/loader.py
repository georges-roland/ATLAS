"""Chargement de chandeliers depuis un CSV (export MT5 ou CSV simple).

Le loader NE corrige RIEN en silence : pas de tri, pas de suppression de
doublons. C'est le rôle de validate_candles() de signaler les problèmes.
"""

from pathlib import Path
from typing import Union

import pandas as pd

OHLC = ["open", "high", "low", "close"]


def _find_column(df: pd.DataFrame, names: list[str]):
    for name in names:
        if name in df.columns:
            return name
    return None


def load_candles(path: Union[str, Path]) -> pd.DataFrame:
    """Charge un CSV et renvoie un DataFrame indexé par date/heure.

    Formats acceptés :
    - Export MT5 : colonnes <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> ...
      (séparateur tabulation, dates du type 2024.01.02)
    - CSV simple : colonnes time (ou datetime/date), open, high, low, close[, volume]

    Les heures sont celles du serveur du broker (pas converties).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    # Detect encoding (handle UTF‑16 with BOM or UTF‑8 with BOM)
    with open(path, "rb") as f:
        raw_start = f.read(4)
    if raw_start.startswith(b"\xff\xfe") or raw_start.startswith(b"\xfe\xff"):
        enc = "utf-16"
    else:
        enc = "utf-8-sig"
    df = pd.read_csv(path, sep=None, engine="python", encoding=enc)
    df.columns = [str(c).strip().strip("<>").lower() for c in df.columns]

    missing = [c for c in OHLC if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}. Trouvées : {list(df.columns)}")

    date_col = _find_column(df, ["date"])
    time_col = _find_column(df, ["time"])
    dt_col = _find_column(df, ["datetime", "timestamp"])

    if dt_col:
        raw = df[dt_col].astype(str)
    elif date_col and time_col:
        raw = df[date_col].astype(str) + " " + df[time_col].astype(str)
    elif time_col:
        raw = df[time_col].astype(str)
    elif date_col:
        raw = df[date_col].astype(str)
    else:
        raise ValueError("Aucune colonne de date/heure trouvée (date, time, datetime).")

    # Dates MT5 : 2024.01.02 -> 2024-01-02
    raw = raw.str.replace(".", "-", regex=False)
    timestamps = pd.to_datetime(raw)

    vol_col = _find_column(df, ["tickvol", "volume", "vol"])
    out = pd.DataFrame(
        {
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df[vol_col], errors="coerce") if vol_col else 0.0,
        }
    )
    out.index = pd.DatetimeIndex(timestamps, name="timestamp")
    return out
