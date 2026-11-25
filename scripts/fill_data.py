# scripts/fill_data.py
"""Utility to fill missing timestamps in ATLAS raw CSV files.

The original `scripts/check_data.py` validates the data and reports gaps
(e.g., missing hourly candles on weekends). This helper script:

1. Loads a CSV containing OHLC data with a ``Date`` column (ISO format).
2. Generates a complete hourly DateTimeIndex covering the full range.
3. Re‑indexes the DataFrame, inserting ``NaN`` rows for missing timestamps.
4. Fills forward ``close`` values (or uses linear interpolation for ``open``,
   ``high``, ``low`` and ``close``) – you can adjust the ``method`` variable.
5. Writes the cleaned data back to a new file (or overwrites the original).

Usage::
    (.venv) python scripts/fill_data.py <input.csv> [--output <output.csv>] [--method {ffill,interpolate}]

The script prints a short summary of how many rows were added and which
method was used.
"""

import argparse
import pandas as pd
from pathlib import Path


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
    return df


def fill_gaps(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """Create a continuous hourly index and fill missing rows.

    Parameters
    ----------
    df: DataFrame indexed by datetime.
    method: ``"ffill"`` – forward‑fill values (simple, safe for price series).
            ``"interpolate"`` – linear interpolation for numeric columns.
    """
    start, end = df.index.min(), df.index.max()
    # ATLAS works with hourly candles; create an hourly range.
    full_index = pd.date_range(start=start, end=end, freq="h")
    df_full = df.reindex(full_index)
    if method == "ffill":
        df_filled = df_full.ffill()
    elif method == "interpolate":
        df_filled = df_full.interpolate(method="linear")
    else:
        raise ValueError(f"Unsupported method: {method}")
    return df_filled


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill missing hourly candles in ATLAS raw data.")
    parser.add_argument("input", type=Path, help="Path to the raw CSV file.")
    parser.add_argument("--output", type=Path, default=None, help="Path to write the cleaned CSV. If omitted, a ``_filled`` suffix is added.")
    parser.add_argument("--method", choices=["ffill", "interpolate"], default="ffill", help="How to fill missing values.")
    args = parser.parse_args()

    df = load_data(args.input)
    original_rows = len(df)
    df_filled = fill_gaps(df, method=args.method)
    filled_rows = len(df_filled)
    added = filled_rows - original_rows

    output_path = args.output or args.input.with_name(args.input.stem + "_filled" + args.input.suffix)
    df_filled.reset_index().rename(columns={"index": "Date"}).to_csv(output_path, index=False)

    print(f"Original rows: {original_rows}")
    print(f"Rows after filling: {filled_rows} (added {added})")
    print(f"Method used: {args.method}")
    print(f"Cleaned file written to: {output_path}")


if __name__ == "__main__":
    main()
