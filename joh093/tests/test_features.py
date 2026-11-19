"""Tests for technical indicators and supervised feature construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockforecast import features


def _random_walk(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.012, n)
    close = 100 * np.exp(np.cumsum(returns))
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=dates,
    )


CFG = {
    "lags": [1, 2, 3, 5],
    "sma_windows": [5, 20],
    "ema_windows": [12, 26],
    "rsi_window": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bollinger_window": 20,
    "volatility_window": 10,
}


def test_indicators_rsi_bounded_and_present() -> None:
    df = _random_walk()
    out = features.compute_indicators(df, CFG)

    assert "RSI" in out.columns
    assert "MACD" in out.columns
    assert "BB_PctB" in out.columns

    rsi = out["RSI"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()
    assert len(rsi) > 20


def test_supervised_target_alignment() -> None:
    df = _random_walk()
    horizon = 3
    X, y, dates, feature_names = features.build_supervised(df, CFG, horizon=horizon)

    assert len(X) == len(y) == len(dates)
    assert len(X) > 50
    # Target at row i must equal the close `horizon` rows ahead.
    expected = df["Close"].shift(-horizon).reindex(dates)
    pd.testing.assert_series_equal(
        y.reset_index(drop=True), expected.reset_index(drop=True), check_names=False
    )
    assert "Lag_1" in feature_names
    assert "RSI" in feature_names


def test_make_sequences_shape() -> None:
    X = np.arange(40, dtype=float).reshape(10, 4)
    y = np.arange(10, dtype=float)
    X_seq, y_seq = features.make_sequences(X, y, seq_len=4)

    assert X_seq.shape == (7, 4, 4)
    assert y_seq.shape == (7,)
    # First sequence's last row corresponds to target index 3.
    np.testing.assert_array_equal(X_seq[0, -1], X[3])
    assert y_seq[0] == y[3]


def test_insufficient_sequence_data_raises() -> None:
    X = np.zeros((3, 2))
    y = np.zeros(3)
    with pytest.raises(ValueError):
        features.make_sequences(X, y, seq_len=10)
