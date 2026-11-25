"""Contrôle un fichier de données.

Usage :
    py scripts/check_data.py data/raw/XAUUSD_H1.csv --tf 60
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atlas.data import load_candles, validate_candles  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Valide un CSV de chandeliers")
    parser.add_argument("path")
    parser.add_argument("--tf", type=int, required=True, help="timeframe en minutes (H1 = 60)")
    args = parser.parse_args()

    df = load_candles(args.path)
    report = validate_candles(df, timeframe_minutes=args.tf)
    print(f"Période : {df.index.min()} -> {df.index.max()}")
    print(report.summary())
    for start, end, weekend in report.unexpected_gaps[:10]:
        print(f"  trou : {start} -> {end}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
