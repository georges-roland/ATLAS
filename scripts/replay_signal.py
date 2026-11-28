"""Vérifie que le moteur de signal reproduit les événements du backtest."""

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

import pandas as pd

# Quand le script est lancé par chemin, Python ajoute scripts/ à sys.path,
# mais pas forcément la racine du projet contenant le paquet atlas.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atlas.backtest import BacktestConfig, run_backtest
from atlas.data import load_candles, validate_candles
from atlas.indicators import ema
from atlas.signal import SignalConfig, generate_signal


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rejoue les signaux XAU/USD et les compare au backtest."
    )
    parser.add_argument("csv_path", help="Fichier de chandeliers OHLC")
    parser.add_argument(
        "--tf",
        type=int,
        default=240,
        help="Timeframe du fichier en minutes (défaut : 240)",
    )
    parser.add_argument("--start", help="Début inclusif (AAAA-MM-JJ)")
    parser.add_argument("--end", help="Fin inclusive (AAAA-MM-JJ)")
    parser.add_argument(
        "--window",
        type=int,
        help="Limite l'historique des indicateurs aux N dernières bougies",
    )
    args = parser.parse_args(argv)

    candles = load_candles(args.csv_path)
    report = validate_candles(candles, timeframe_minutes=args.tf)
    if not report.ok:
        print("Données invalides :")
        print(report.summary())
        return 1

    if args.start is not None:
        candles = candles[candles.index >= pd.Timestamp(args.start).normalize()]
    if args.end is not None:
        end_exclusive = pd.Timestamp(args.end).normalize() + pd.Timedelta(days=1)
        candles = candles[candles.index < end_exclusive]
    if candles.empty:
        print(f"Aucune donnée dans la période demandée (start={args.start}, end={args.end}).")
        return 1

    backtest_config = BacktestConfig(
        initial_capital=10_000.0,
        risk_pct=1.0,
        ema_span=200,
        atr_period=14,
        sl_mult=1.5,
        allow_short=False,
        spread=0.30,
        slippage=0.05,
        commission_per_lot=0.0,
        swap_long_per_oz=0.126,
        swap_short_per_oz=0.0,
    )
    signal_config = SignalConfig(
        ema_span=backtest_config.ema_span,
        atr_period=backtest_config.atr_period,
        sl_mult=backtest_config.sl_mult,
        risk_pct=backtest_config.risk_pct,
        spread=backtest_config.spread,
        slippage=backtest_config.slippage,
        commission_per_lot=backtest_config.commission_per_lot,
        instrument=backtest_config.instrument,
    )
    warmup = max(signal_config.ema_span, signal_config.atr_period)
    if args.window is not None and args.window <= warmup:
        parser.error(f"--window doit être supérieur à {warmup}")
    result = run_backtest(candles, backtest_config)

    index_by_time = {time: i for i, time in enumerate(candles.index)}
    expected_events = []
    trade_indices = []
    for trade in result.trades:
        if trade.side != "long":
            print(f"ERREUR : trade inattendu ({trade.side}) à {trade.entry_time}.")
            return 1
        entry_index = index_by_time[trade.entry_time]
        exit_index = index_by_time[trade.exit_time]
        if entry_index == 0:
            print("ERREUR : entrée sans bougie de signal précédente.")
            return 1
        expected_events.append((entry_index - 1, "buy"))
        trade_indices.append((entry_index, exit_index, trade))
        if trade.exit_reason == "signal":
            if exit_index == 0:
                print("ERREUR : sortie sur signal sans bougie précédente.")
                return 1
            expected_events.append((exit_index - 1, "close_long"))

    expected_events.sort()
    closes = candles["close"].to_numpy(dtype=float)
    full_ema = (
        ema(candles["close"], signal_config.ema_span).to_numpy(dtype=float)
        if args.window is None
        else None
    )
    actual_events = []

    for i in range(warmup, len(candles) - 1):
        if args.window is None:
            history = candles.iloc[: i + 1]
            previous_ema = full_ema[i - 1]
            current_ema = full_ema[i]
        else:
            window_start = max(0, i + 1 - args.window)
            history = candles.iloc[window_start : i + 1]
            if len(history) <= warmup:
                continue
            window_ema = ema(history["close"], signal_config.ema_span)
            previous_ema = float(window_ema.iloc[-2])
            current_ema = float(window_ema.iloc[-1])

        crossed_up = closes[i - 1] <= previous_ema and closes[i] > current_ema
        crossed_down = closes[i - 1] >= previous_ema and closes[i] < current_ema
        if not crossed_up and not crossed_down:
            continue

        position_side = (
            "long"
            if any(entry <= i < exit_ for entry, exit_, _ in trade_indices)
            else None
        )
        equity = backtest_config.initial_capital + sum(
            trade.net_pnl
            for _, exit_, trade in trade_indices
            if exit_ <= i
        )
        signal = generate_signal(
            history,
            equity,
            position_side=position_side,
            config=signal_config,
        )
        if signal is not None:
            actual_events.append((i, signal.action))

    expected_set = set(expected_events)
    actual_set = set(actual_events)
    missing_events = sorted(expected_set - actual_set)
    extra_events = sorted(actual_set - expected_set)
    if missing_events or extra_events:
        if args.window is None:
            label = "ERREUR"
        else:
            label = "ÉCARTS"
        print(
            f"{label} : {len(missing_events)} signal(s) manquant(s), "
            f"{len(extra_events)} signal(s) supplémentaire(s)."
        )
        for event_index, action in missing_events:
            print(f"  Manquant : {candles.index[event_index]} {action}")
        for event_index, action in extra_events:
            print(f"  Supplémentaire : {candles.index[event_index]} {action}")
        return 1 if args.window is None else 0

    suffix = f" (--window {args.window})" if args.window is not None else ""
    print(f"OK, le signal refait exactement le backtest{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
