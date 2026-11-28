"""Choisit les réglages sur une période, les vérifie sur la suivante.

Exemple :
    py scripts/train_test.py data/raw/XAUUSD_H1.csv --train-end 2024-12-31 --short --spread 0.3 --commission 7
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.backtest import BacktestConfig, DataValidationError  # noqa: E402
from atlas.validation import train_test_from_csv  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Validation hors échantillon ATLAS v2")
    p.add_argument("csv", type=Path)
    p.add_argument("--train-end", required=True, help="dernier jour de la période de choix, AAAA-MM-JJ")
    p.add_argument("--tf", type=int, default=60)
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--risk", type=float, default=1.0)
    p.add_argument("--atr", type=int, default=14)
    p.add_argument("--short", action="store_true")
    p.add_argument("--spread", type=float, default=0.30)
    p.add_argument("--slippage", type=float, default=0.05)
    p.add_argument("--commission", type=float, default=7.0)
    a = p.parse_args()

    cfg = BacktestConfig(
        initial_capital=a.capital,
        risk_pct=a.risk,
        atr_period=a.atr,
        allow_short=a.short,
        spread=a.spread,
        slippage=a.slippage,
        commission_per_lot=a.commission,
    )
    try:
        study = train_test_from_csv(a.csv, a.train_end, cfg, timeframe_minutes=a.tf)
    except DataValidationError as e:
        print(e)
        print("\nAnnulé : corrige d'abord les données (py scripts/check_data.py ...).")
        return 1
    except ValueError as e:
        print(f"Erreur : {e}")
        return 1
    print(study.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
