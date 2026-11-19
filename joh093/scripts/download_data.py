#!/usr/bin/env python
"""Download OHLCV price history from Yahoo Finance and save it to CSV.

Examples
--------
    python scripts/download_data.py --ticker AAPL --period 3y
    python scripts/download_data.py --ticker MSFT --out data/msft.csv

Downloaded files under ``data/`` are git-ignored; only the small bundled
``data/sample_aapl.csv`` is committed to the repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stockforecast import data as data_module  # noqa: E402
from stockforecast.config import DATA_DIR  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Yahoo Finance OHLCV data to CSV.")
    parser.add_argument("--ticker", default="AAPL", help="Yahoo ticker symbol (default: %(default)s)")
    parser.add_argument("--period", default="3y", help="look-back period (default: %(default)s)")
    parser.add_argument("--interval", default="1d", help="bar interval (default: %(default)s)")
    parser.add_argument(
        "--out",
        default=str(DATA_DIR / "sample_aapl.csv"),
        help="output CSV path (default: bundled sample)",
    )
    args = parser.parse_args(argv)

    df = data_module.download_yfinance(args.ticker, period=args.period, interval=args.interval)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index_label="Date")
    print(f"Saved {len(df)} rows ({df.index.min().date()} -> {df.index.max().date()}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
