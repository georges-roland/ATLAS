# atlas/backtest.py
"""
Pure back‑test engine for ATLAS v2.

- No printing – returns a result dictionary.
- Loads / validates data via ``atlas.data`` utilities.
- Uses ``atlas.indicators`` for EMA & ATR.
- Position sizing via ``atlas.risk.lot_size_for_trade``.
- Supports long and short side (``--short`` adds short entries).
- Execution at next candle open, stop‑loss / take‑profit tested on high/low.
- Costs: spread, commission per lot (both entry & exit), slippage.
- Drawdown computed on the equity curve (peak‑to‑trough) in $ and %.
- Provides a buy‑and‑hold benchmark for comparison.
"""

from __future__ import annotations

import pandas as pd
from typing import List, Dict, Any, Optional

from atlas.indicators import ema, atr
from atlas.risk import lot_size_for_trade
from atlas.data.loader import load_candles
from atlas.data.validator import validate_candles


def _equity_curve(trades: List[Dict[str, Any]], initial_capital: float) -> List[float]:
    """Build equity curve (list) from sequential trades.
    Each trade dict must contain ``net_pnl`` (float) and ``exit_index`` (int).
    """
    equity = initial_capital
    curve = []
    for t in sorted(trades, key=lambda x: x["exit_index"]):
        equity += t["net_pnl"]
        curve.append(equity)
    return curve


def _max_drawdown(equity_curve: List[float]) -> Dict[str, float]:
    """Return max drawdown in $ and %.
    ``equity_curve`` is a list of equity values after each trade.
    """
    max_dd = 0.0
    max_dd_pct = 0.0
    peak = -float("inf")
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        draw = peak - equity
        if draw > max_dd:
            max_dd = draw
            max_dd_pct = draw / peak * 100 if peak != 0 else 0.0
    return {"max_dd": max_dd, "max_dd_pct": max_dd_pct}


def run_backtest(
    df: pd.DataFrame,
    *,
    initial_capital: float = 10_000.0,
    risk_pct: float = 1.0,
    ema_span: int = 20,
    atr_period: int = 14,
    sl_mult: float = 1.5,
    tp_mult: Optional[float] = None,
    short: bool = False,
    spread: float = 0.30,
    commission_per_lot: float = 7.0,
    slippage: float = 0.05,
) -> Dict[str, Any]:
    """Run a back‑test on a clean OHLC DataFrame.

    Returns a dict with performance metrics and trade details.
    """
    # 1. Validate data
    report = validate_candles(df, timeframe_minutes=60)
    if not report.ok:
        raise ValueError(f"Data validation failed: {report.summary()}")

    # 2. Compute indicators
    df = df.copy()
    df["EMA"] = ema(df["close"], ema_span)
    df["ATR"] = atr(df, period=atr_period)

    # 3. Initialise tracking variables
    equity = initial_capital
    position = 0.0  # positive = long lots, negative = short lots
    entry_price = None
    entry_atr = None
    entry_idx = None
    trades: List[Dict[str, Any]] = []
    total_commission = 0.0
    total_spread = 0.0
    total_slippage = 0.0

    def _calc_levels(entry: float, atr_val: float) -> tuple[float, Optional[float]]:
        sl = entry - sl_mult * atr_val if position > 0 else entry + sl_mult * atr_val
        tp = None
        if tp_mult is not None:
            tp = entry + tp_mult * atr_val if position > 0 else entry - tp_mult * atr_val
        return sl, tp

    # Iterate over candles – need previous EMA/close for cross detection
    prev_ema = None
    prev_close = None
    indices = list(df.index)

    for i, ts in enumerate(indices):
        row = df.loc[ts]
        price_open, price_high, price_low, price_close = (
            row["open"], row["high"], row["low"], row["close"]
        )
        ema_val = row["EMA"]
        atr_val = row["ATR"]

        # EXIT first
        if position != 0:
            sl_price, tp_price = _calc_levels(entry_price, entry_atr)
            # Gap exit
            if position > 0:
                if price_open <= sl_price:
                    exit_price, reason = price_open, "gap_sl"
                elif tp_price is not None and price_open >= tp_price:
                    exit_price, reason = price_open, "gap_tp"
                else:
                    if price_low <= sl_price:
                        exit_price, reason = sl_price, "sl"
                    elif tp_price is not None and price_high >= tp_price:
                        exit_price, reason = tp_price, "tp"
                    elif price_close < ema_val:
                        exit_price, reason = price_close, "ema_cross"
                    else:
                        exit_price = None
            else:
                if price_open >= sl_price:
                    exit_price, reason = price_open, "gap_sl"
                elif tp_price is not None and price_open <= tp_price:
                    exit_price, reason = price_open, "gap_tp"
                else:
                    if price_high >= sl_price:
                        exit_price, reason = sl_price, "sl"
                    elif tp_price is not None and price_low <= tp_price:
                        exit_price, reason = tp_price, "tp"
                    elif price_close > ema_val:
                        exit_price, reason = price_close, "ema_cross"
                    else:
                        exit_price = None
            if exit_price is not None:
                # Adjust for spread & slippage on exit side
                if position > 0:
                    exit_adj = exit_price - spread / 2 - slippage
                else:
                    exit_adj = exit_price + spread / 2 + slippage
                pnl = (exit_adj - entry_price) * position * 100
                commission = commission_per_lot * abs(position) * 2
                net_pnl = pnl - commission
                equity += net_pnl
                total_commission += commission
                total_spread += spread * abs(position) * 100
                total_slippage += slippage * abs(position) * 100
                trades.append({
                    "entry_index": entry_idx,
                    "exit_index": i,
                    "entry_time": indices[entry_idx] if entry_idx is not None else None,
                    "exit_time": ts,
                    "direction": "long" if position > 0 else "short",
                    "lots": position,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "exit_reason": reason,
                    "gross_pnl": pnl,
                    "commission": commission,
                    "net_pnl": net_pnl,
                    "duration_h": (ts - indices[entry_idx]).total_seconds() / 3600 if entry_idx is not None else None,
                })
                position = 0.0
                entry_price = None
                entry_atr = None
                entry_idx = None
                prev_ema = ema_val
                prev_close = price_close
                continue

        # ENTRY when flat
        if position == 0 and prev_ema is not None and prev_close is not None:
            long_signal = (prev_close < prev_ema) and (price_close > ema_val)
            short_signal = short and (prev_close > prev_ema) and (price_close < ema_val)
            if long_signal or short_signal:
                direction = 1.0 if long_signal else -1.0
                entry_price_candidate = price_open + (spread / 2 + slippage) * direction
                stop_price = entry_price_candidate - direction * sl_mult * atr_val
                lots = lot_size_for_trade(equity, risk_pct, entry_price_candidate, stop_price)
                if lots == 0:
                    prev_ema = ema_val
                    prev_close = price_close
                    continue
                # entry costs (commission only entry now)
                total_commission += commission_per_lot * abs(lots)
                total_spread += spread * abs(lots) * 100
                total_slippage += slippage * abs(lots) * 100
                position = direction * lots
                entry_price = entry_price_candidate
                entry_atr = atr_val
                entry_idx = i
        prev_ema = ema_val
        prev_close = price_close

    # Close any open position at last candle close
    if position != 0:
        exit_price = df.iloc[-1]["close"]
        if position > 0:
            exit_adj = exit_price - spread / 2 - slippage
        else:
            exit_adj = exit_price + spread / 2 + slippage
        pnl = (exit_adj - entry_price) * position * 100
        commission = commission_per_lot * abs(position) * 2
        net_pnl = pnl - commission
        equity += net_pnl
        total_commission += commission
        total_spread += spread * abs(position) * 100
        total_slippage += slippage * abs(position) * 100
        trades.append({
            "entry_index": entry_idx,
            "exit_index": len(df) - 1,
            "entry_time": indices[entry_idx] if entry_idx is not None else None,
            "exit_time": indices[-1],
            "direction": "long" if position > 0 else "short",
            "lots": position,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_reason": "final_close",
            "gross_pnl": pnl,
            "commission": commission,
            "net_pnl": net_pnl,
            "duration_h": (indices[-1] - indices[entry_idx]).total_seconds() / 3600 if entry_idx is not None else None,
        })
        position = 0.0

    # Metrics
    net_profit = equity - initial_capital
    num_trades = len(trades)
    wins = sum(1 for t in trades if t["net_pnl"] > 0)
    win_rate = wins / num_trades * 100 if num_trades else 0.0
    total_profit = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    total_loss = -sum(t["net_pnl"] for t in trades if t["net_pnl"] < 0)
    profit_factor = total_profit / total_loss if total_loss != 0 else float("inf")
    avg_trade_h = (
        sum(t["duration_h"] for t in trades if t["duration_h"] is not None) / num_trades
        if num_trades
        else 0.0
    )
    expectancy = net_profit / num_trades if num_trades else 0.0
    equity_curve_vals = _equity_curve(trades, initial_capital)
    dd = _max_drawdown(equity_curve_vals)

    # Buy‑and‑hold benchmark (1 lot = 100 oz)
    bh_equity = initial_capital + (df.iloc[-1]["close"] - df.iloc[0]["open"]) * 100
    bh_net = bh_equity - initial_capital

    return {
        "final_equity": equity,
        "net_profit": net_profit,
        "trades": num_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_trade_hours": avg_trade_h,
        "max_drawdown": dd["max_dd"],
        "max_drawdown_pct": dd["max_dd_pct"],
        "total_commission": total_commission,
        "total_spread_cost": total_spread,
        "total_slippage_cost": total_slippage,
        "buy_and_hold": {"final_equity": bh_equity, "net_profit": bh_net},
        "trades_detail": trades,
    }

# CLI wrapper (kept for compatibility)
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Run ATLAS pure back‑test.")
    parser.add_argument("csv", type=Path, help="CSV file path")
    parser.add_argument("--ema", type=int, default=20)
    parser.add_argument("--atr", type=int, default=14)
    parser.add_argument("--risk", type=float, default=1.0)
    parser.add_argument("--sl-mult", type=float, default=1.5)
    parser.add_argument("--tp-mult", type=float, default=None)
    parser.add_argument("--short", action="store_true")
    parser.add_argument("--spread", type=float, default=0.30)
    parser.add_argument("--commission", type=float, default=7.0)
    parser.add_argument("--slippage", type=float, default=0.05)
    args = parser.parse_args()
    df_raw = load_candles(args.csv)
    results = run_backtest(
        df_raw,
        initial_capital=10_000.0,
        risk_pct=args.risk,
        ema_span=args.ema,
        atr_period=args.atr,
        sl_mult=args.sl_mult,
        tp_mult=args.tp_mult,
        short=args.short,
        spread=args.spread,
        commission_per_lot=args.commission,
        slippage=args.slippage,
    )
    import json, sys
    json.dump(results, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
