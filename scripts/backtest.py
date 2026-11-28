"""Lance un backtest sur un CSV de chandeliers (export MT5 ou CSV simple).

Exemple :
    py scripts/backtest.py data/raw/XAUUSD_H4_clean.csv --tf 240 --ema 200 --swap-long 0.126

Les données sont d'abord VALIDEES (atlas.data.validate_candles). Si elles
contiennent des erreurs, le backtest ne tourne pas.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from atlas.backtest import BacktestConfig, DataValidationError, backtest_from_csv  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Backtest ATLAS v2 (prix / EMA, stop ATR)")
    p.add_argument("csv", type=Path)
    p.add_argument(
        "--tf",
        type=int,
        default=60,
        help="timeframe du FICHIER en minutes (sert à la validation, ne rééchantillonne pas)",
    )
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--risk", type=float, default=1.0, help="risque par trade en %% du capital")
    p.add_argument("--ema", type=int, default=20)
    p.add_argument("--atr", type=int, default=14)
    p.add_argument("--sl-mult", type=float, default=1.5)
    p.add_argument("--tp-mult", type=float, default=None)
    p.add_argument("--short", action="store_true", help="AJOUTE les ventes (les achats restent)")
    p.add_argument("--spread", type=float, default=0.30, help="USD par once")
    p.add_argument("--slippage", type=float, default=0.05, help="USD par once, par exécution")
    p.add_argument("--commission", type=float, default=7.0, help="USD par lot, aller-retour")
    p.add_argument("--swap-long", type=float, default=0.0, help="USD par once et par nuit (coût)")
    p.add_argument("--swap-short", type=float, default=0.0, help="USD par once et par nuit (coût)")
    p.add_argument(
        "--start",
        default=None,
        help="début de la période, AAAA-MM-JJ (les indicateurs repartent de zéro à cette date)",
    )
    p.add_argument("--end", default=None, help="fin de la période, AAAA-MM-JJ (incluse)")
    p.add_argument("--trades-csv", type=Path, default=None, help="exporte les trades")
    a = p.parse_args()

    cfg = BacktestConfig(
        initial_capital=a.capital,
        risk_pct=a.risk,
        ema_span=a.ema,
        atr_period=a.atr,
        sl_mult=a.sl_mult,
        tp_mult=a.tp_mult,
        allow_short=a.short,
        spread=a.spread,
        slippage=a.slippage,
        commission_per_lot=a.commission,
        swap_long_per_oz=a.swap_long,
        swap_short_per_oz=a.swap_short,
    )
    try:
        res = backtest_from_csv(
            a.csv, cfg, timeframe_minutes=a.tf, start=a.start, end=a.end
        )
    except ValueError as e:  # période vide, colonnes manquantes...
        print(f"Erreur : {e}")
        return 1
    except DataValidationError as e:
        print(e)
        print("\nBacktest annulé : corrige d'abord les données (py scripts/check_data.py ...).")
        return 1

    print(f"Période testée : {res.equity_curve.index[0]} -> {res.equity_curve.index[-1]}")
    print(res.summary())
    if a.trades_csv:
        pd.DataFrame([t.__dict__ for t in res.trades]).to_csv(a.trades_csv, index=False)
        print(f"\nTrades exportés : {a.trades_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
