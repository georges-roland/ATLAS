# scripts/backtest.py
"""Simple back‑test for ATLAS v2.

The script loads an hourly OHLC CSV, computes the EMA & ATR indicators
from ``atlas.indicators``, determines a position size using ``atlas.risk``
and runs a very basic strategy:

* **Entry** – when the close price crosses **above** the EMA (bullish).
* **Exit** – when the close price crosses **below** the EMA (bearish) or
  when a stop‑loss (based on ATR) is hit.

The back‑test reports:
- total net profit (in USD),
- number of trades, win‑rate,
- maximum drawdown,
- average trade duration.

It is intentionally minimal – just enough to validate the data pipeline
and the risk model.
"""

import argparse
from pathlib import Path
import pandas as pd
import sys, os
# Ensure the project root (containing the 'atlas' package) is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import the project's modules (they are in the ``atlas`` package).
from atlas.indicators import ema, atr
from atlas.risk import lot_size_for_trade


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV, handling either a ``Date`` or ``time`` column as the timestamp.
    """
    df = pd.read_csv(csv_path)
    # Accept column named ``Date`` or ``time`` (case‑insensitive).
    date_col = next((col for col in df.columns if col.lower() in ("date", "time")), None)
    if date_col is None:
        raise ValueError("CSV must contain a 'Date' or 'time' column")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    # Ensure numeric columns are float.
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


def run_backtest(df: pd.DataFrame, ema_span: int = 20, atr_period: int = 14, risk_pct: float = 1.0, sl_mult: float = 1.5, short: bool = False):
    # Compute indicators
    df["EMA"] = ema(df["close"], ema_span)
    df["ATR"] = atr(df, period=atr_period)

    # Track equity (starting with 10,000 USD) and positions.
    equity = 10_000.0
    position = 0  # lots, positive = long, negative = short (we only go long here)
    entry_price = None
    entry_atr = None
    entry_idx = None
    trades = []

    for idx, row in df.iterrows():
        price = row["close"]
        ema_val = row["EMA"]
        atr_val = row["ATR"]

# ---- Exit logic ----
        if position != 0:
            if position > 0:
                # long position exit
                # stop‑loss
                stop_price = entry_price - sl_mult * entry_atr
                if price <= stop_price:
                    pnl = (price - entry_price) * position * 100
                    equity += pnl
                    trades.append({"entry": entry_price, "exit": price, "pnl": pnl, "duration": (idx - entry_idx).total_seconds() / 3600})
                    position = 0
                    continue
                # EMA cross‑under exit
                if price < ema_val:
                    pnl = (price - entry_price) * position * 100
                    equity += pnl
                    trades.append({"entry": entry_price, "exit": price, "pnl": pnl, "duration": (idx - entry_idx).total_seconds() / 3600})
                    position = 0
                    continue
            else:
                # short position exit
                # stop‑loss for short (price goes up)
                stop_price = entry_price + sl_mult * entry_atr
                if price >= stop_price:
                    pnl = (entry_price - price) * abs(position) * 100
                    equity += pnl
                    trades.append({"entry": entry_price, "exit": price, "pnl": pnl, "duration": (idx - entry_idx).total_seconds() / 3600})
                    position = 0
                    continue
                # EMA cross‑over exit for short
                if price > ema_val:
                    pnl = (entry_price - price) * abs(position) * 100
                    equity += pnl
                    trades.append({"entry": entry_price, "exit": price, "pnl": pnl, "duration": (idx - entry_idx).total_seconds() / 3600})
                    position = 0
                    continue

        # ---- Entry logic (only when flat) ----
        if position == 0:
            # Long entry when price crosses above EMA
            if not short and price > ema_val:
                lot = 1.0
                position = lot
                entry_price = price
                entry_atr = atr_val
                entry_idx = idx
            # Short entry when short flag enabled and price crosses below EMA
            elif short and price < ema_val:
                lot = -1.0  # negative for short
                position = lot
                entry_price = price
                entry_atr = atr_val
                entry_idx = idx

    # Final equity after last bar (close any open position at last price)
    if position != 0:
        pnl = (df.iloc[-1]["close"] - entry_price) * position * 100
        equity += pnl
        trades.append({"entry": entry_price, "exit": df.iloc[-1]["close"], "pnl": pnl, "duration": (df.index[-1] - entry_idx).total_seconds() / 3600})

    # Summary statistics
    total_profit = equity - 10_000.0
    wins = sum(1 for t in trades if t["pnl"] > 0)
    win_rate = wins / len(trades) * 100 if trades else 0.0
    max_dd = min(0, min(equity - 10_000.0, total_profit))
    avg_duration = sum(t["duration"] for t in trades) / len(trades) if trades else 0.0
    # Additional metrics
    total_profit_long = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    total_loss = -sum(t["pnl"] for t in trades if t["pnl"] < 0)
    profit_factor = (total_profit_long / total_loss) if total_loss != 0 else float('inf')
    avg_profit = (total_profit / len(trades)) if trades else 0.0

    return {
        "final_equity": equity,
        "net_profit": total_profit,
        "trades": len(trades),
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "avg_trade_h": avg_duration,
        "profit_factor": profit_factor,
        "avg_profit": avg_profit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal ATLAS back‑test.")
    parser.add_argument("csv", type=Path, help="Path to the cleaned hourly CSV file.")
    parser.add_argument("--ema", type=int, default=20, help="EMA span (default 20).")
    parser.add_argument("--atr", type=int, default=14, help="ATR period (default 14).")
    parser.add_argument("--risk", type=float, default=1.0, help="Risk % per trade (default 1%).")
    parser.add_argument("--sl-mult", type=float, default=1.5, help="Stop‑loss multiplier (default 1.5).")
    parser.add_argument("--short", action="store_true", help="Enable short‑side entries when price crosses below EMA.")
    args = parser.parse_args()

    df = load_data(args.csv)
    results = run_backtest(df, ema_span=args.ema, atr_period=args.atr, risk_pct=args.risk, sl_mult=args.sl_mult, short=args.short)

    print("--- Back‑test Summary ---")
    print(f"Final equity:    ${results['final_equity']:.2f}")
    print(f"Net profit:      ${results['net_profit']:.2f}")
    print(f"Trades executed: {results['trades']}")
    print(f"Win rate:        {results['win_rate']:.1f}%")
    print(f"Max drawdown:    ${results['max_drawdown']:.2f}")
    print(f"Avg trade dur.:  {results['avg_trade_h']:.2f} h")


if __name__ == "__main__":
    main()
