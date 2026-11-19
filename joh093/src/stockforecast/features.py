"""Feature engineering: technical indicators, lag features and supervised samples.

The forecasting target is the closing price ``horizon`` trading days ahead.
Features combine:

* lagged closing prices (short- and medium-term memory),
* moving averages (SMA/EMA),
* momentum oscillators (RSI, MACD),
* volatility and Bollinger-band position,
* daily return and volume change.

All transformations are computed from the price history only, so the pipeline
is reproducible and does not depend on external sentiment or fundamental
data.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fillna(50.0)


def compute_indicators(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Append technical indicator columns to a validated OHLCV frame."""
    out = df.copy()
    close = out["Close"]

    # Price change and moving averages.
    out["Return"] = close.pct_change()
    for window in cfg.get("sma_windows", [5, 20]):
        out[f"SMA_{window}"] = close.rolling(window=window).mean()
    for window in cfg.get("ema_windows", [12, 26]):
        out[f"EMA_{window}"] = close.ewm(span=window, adjust=False).mean()

    # RSI.
    out["RSI"] = _rsi(close, cfg.get("rsi_window", 14))

    # MACD: line, signal and histogram.
    ema_fast = close.ewm(span=cfg.get("macd_fast", 12), adjust=False).mean()
    ema_slow = close.ewm(span=cfg.get("macd_slow", 26), adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal = macd_line.ewm(span=cfg.get("macd_signal", 9), adjust=False).mean()
    out["MACD"] = macd_line
    out["MACD_Signal"] = signal
    out["MACD_Hist"] = macd_line - signal

    # Bollinger bands and %b position.
    bb_window = cfg.get("bollinger_window", 20)
    bb_mid = close.rolling(window=bb_window).mean()
    bb_std = close.rolling(window=bb_window).std()
    upper = bb_mid + 2.0 * bb_std
    lower = bb_mid - 2.0 * bb_std
    out["BB_Upper"] = upper
    out["BB_Lower"] = lower
    out["BB_PctB"] = (close - lower) / (upper - lower).replace(0.0, np.nan)

    # Rolling realised volatility and volume change.
    vol_window = cfg.get("volatility_window", 10)
    out["Volatility"] = out["Return"].rolling(window=vol_window).std()
    if "Volume" in out.columns:
        out["Volume_Change"] = out["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    else:
        out["Volume_Change"] = 0.0

    return out


def build_supervised(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    horizon: int = 1,
    target_col: str = "Close",
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex, list[str]]:
    """Build a supervised learning frame.

    Returns
    -------
    X: DataFrame
        Feature matrix (indicators + lagged prices).
    y: Series
        Target close price ``horizon`` days ahead.
    dates: DatetimeIndex
        Dates aligned with ``X`` and ``y``.
    feature_names: list[str]
        Ordered feature column names.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    features = compute_indicators(df, cfg)

    # Lagged close prices.
    for lag in cfg.get("lags", [1, 2, 3, 4, 5, 10, 20]):
        features[f"Lag_{lag}"] = features[target_col].shift(lag)

    # Target: close price `horizon` steps ahead.
    features["Target"] = features[target_col].shift(-horizon)

    feature_cols = [col for col in features.columns if col != "Target"]
    features = features.dropna(subset=feature_cols + ["Target"])

    X = features[feature_cols].copy()
    y = features["Target"].copy()
    dates = features.index
    return X, y, dates, feature_cols


def make_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape a tabular matrix into (samples, seq_len, features) sequences.

    Used by the LSTM model. Row ``i`` uses the ``seq_len`` rows ending at
    position ``i`` to predict ``y[i]``.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if len(X) < seq_len:
        raise ValueError("Not enough samples to build sequences.")

    n_samples = len(X) - seq_len + 1
    X_seq = np.empty((n_samples, seq_len, X.shape[1]), dtype=np.float32)
    y_seq = np.empty(n_samples, dtype=np.float32)
    for i in range(n_samples):
        X_seq[i] = X[i : i + seq_len]
        y_seq[i] = y[i + seq_len - 1]
    return X_seq, y_seq


def time_split(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    train_ratio: float = 0.8,
) -> tuple[Any, Any, Any, Any]:
    """Split data in chronological order (no shuffling)."""
    n = len(X)
    split = int(n * train_ratio)
    if isinstance(X, pd.DataFrame):
        X_train, X_test = X.iloc[:split], X.iloc[split:]
    else:
        X_train, X_test = X[:split], X[split:]
    if isinstance(y, pd.Series):
        y_train, y_test = y.iloc[:split], y.iloc[split:]
    else:
        y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test
