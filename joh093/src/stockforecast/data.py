"""Market data loading, validation and normalisation.

The data layer supports two sources:

1. **Live download** through :mod:`yfinance` (Yahoo Finance), using a ticker
   symbol such as ``AAPL`` or ``MSFT``.
2. **Local CSV** with columns ``Date, Open, High, Low, Close, Volume`` (column
   names are matched case-insensitively and common aliases are accepted).

A small bundled sample (``data/sample_aapl.csv``) lets the application and the
test suite run fully offline. Large downloaded datasets are never committed to
the repository; see ``data/README.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATA_DIR
from .errors import DataSourceError, InsufficientDataError, InvalidDataError

log = logging.getLogger(__name__)

STANDARD_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Accepted aliases -> standard column (lower-case comparison)
_COLUMN_ALIASES: dict[str, str] = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj close": "Close",
    "adj_close": "Close",
    "adjusted close": "Close",
    "volume": "Volume",
    "vol": "Volume",
}

SAMPLE_CSV = DATA_DIR / "sample_aapl.csv"


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a MultiIndex column frame (field/ticker) to plain field columns."""
    if isinstance(df.columns, pd.MultiIndex):
        known = {"Open", "High", "Low", "Close", "Adj Close", "Volume",
                 "Dividends", "Stock Splits"}

        def _field(col: tuple | list | str) -> str:
            if not isinstance(col, (tuple, list)):
                return str(col)
            for part in col:
                if str(part) in known:
                    return str(part)
            return str(col[0])

        df.columns = [_field(col) for col in df.columns]
    return df


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename alias columns to the standard schema and drop duplicates."""
    mapping: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in _COLUMN_ALIASES:
            mapping[col] = _COLUMN_ALIASES[key]
    df = df.rename(columns=mapping)
    # Keep one copy of each standard column, preferring the first occurrence.
    keep: list[str] = []
    for col in STANDARD_COLUMNS:
        if col in df.columns:
            keep.append(col)
    return df[keep].copy()


def _validate(df: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    """Validate and clean the frame; raise project errors on failure."""
    if df is None or df.empty:
        raise InvalidDataError("The dataset is empty.")

    if "Close" not in df.columns:
        raise InvalidDataError("The dataset must contain a 'Close' price column.")

    # Ensure a sorted, unique datetime index.
    if not isinstance(df.index, pd.DatetimeIndex):
        raise InvalidDataError("The dataset index must be a Date column.")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Coerce numeric columns and drop rows where Close is missing/non-finite.
    for col in STANDARD_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]

    # Forward-fill very short gaps in OHLC; volume gaps become zero.
    df[["Open", "High", "Low", "Close"]] = df[
        ["Open", "High", "Low", "Close"]
    ].ffill()
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)

    if len(df) < min_rows:
        raise InsufficientDataError(
            f"Only {len(df)} bars are available; at least {min_rows} are required."
        )
    return df


def download_yfinance(ticker: str, period: str = "3y", interval: str = "1d") -> pd.DataFrame:
    """Download daily OHLCV bars from Yahoo Finance.

    Raises :class:`DataSourceError` if the ticker cannot be retrieved or the
    returned frame is empty.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - dependency pinned in requirements
        raise DataSourceError("yfinance is not installed.") from exc

    ticker = ticker.strip().upper()
    log.info("Downloading %s (%s, %s)", ticker, period, interval)
    try:
        raw = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    except Exception as exc:  # noqa: BLE001 - network failures vary by backend
        raise DataSourceError(f"Could not download '{ticker}': {exc}") from exc

    if raw is None or raw.empty:
        raise DataSourceError(
            f"No data returned for '{ticker}'. Check the ticker symbol or try again later."
        )

    raw = _flatten_yfinance_columns(raw)
    df = _normalise_columns(raw)
    df.index.name = "Date"
    return df


def load_csv(path: Path | str) -> pd.DataFrame:
    """Load an OHLCV CSV file and normalise its columns."""
    path = Path(path)
    if not path.exists():
        raise DataSourceError(f"CSV file not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        raise InvalidDataError(f"Could not read CSV '{path.name}': {exc}") from exc

    # Locate a date-like column and set it as the index.
    date_col: str | None = None
    for col in df.columns:
        if str(col).strip().lower() in {"date", "datetime", "timestamp", "unnamed: 0"}:
            date_col = col
            break
    if date_col is None:
        raise InvalidDataError("CSV must include a Date column.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col)
    df.index.name = "Date"
    df = _normalise_columns(df)
    return df


def load_dataset(
    source: str | Path,
    period: str = "3y",
    interval: str = "1d",
    min_rows: int = 80,
) -> pd.DataFrame:
    """Load data from a ticker symbol or a local CSV path.

    Parameters
    ----------
    source:
        A Yahoo ticker symbol (``AAPL``) or a path to a CSV file.
    """
    source_str = str(source).strip()

    if Path(source_str).exists():
        df = load_csv(source_str)
    else:
        df = download_yfinance(source_str, period=period, interval=interval)

    return _validate(df, min_rows=min_rows)


def load_sample() -> pd.DataFrame:
    """Load the bundled offline sample dataset."""
    if not SAMPLE_CSV.exists():
        raise DataSourceError(
            f"Bundled sample not found at {SAMPLE_CSV}. Run scripts/download_data.py."
        )
    return _validate(load_csv(SAMPLE_CSV), min_rows=60)


def data_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Produce a small JSON-serialisable summary of a loaded dataset."""
    return {
        "rows": int(len(df)),
        "start": df.index.min().strftime("%Y-%m-%d"),
        "end": df.index.max().strftime("%Y-%m-%d"),
        "columns": list(df.columns),
        "last_close": round(float(df["Close"].iloc[-1]), 4),
        "min_close": round(float(df["Close"].min()), 4),
        "max_close": round(float(df["Close"].max()), 4),
    }
