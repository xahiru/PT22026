"""Forecast evaluation metrics.

All metrics are computed on the **price scale** (not returns) so they are easy
to interpret alongside the charts. MAPE guards against zero/near-zero prices.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _as_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(_as_array(y_true), _as_array(y_pred))))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(_as_array(y_true), _as_array(y_pred)))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = _as_array(y_true)
    y_pred = _as_array(y_pred)
    mask = np.abs(y_true) > eps
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def r2(y_true, y_pred) -> float:
    return float(r2_score(_as_array(y_true), _as_array(y_pred)))


def direction_accuracy(y_true, y_pred) -> float:
    """Share of steps where the predicted next-day direction matches reality."""
    y_true = _as_array(y_true)
    y_pred = _as_array(y_pred)
    if len(y_true) < 2:
        return float("nan")
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    match = true_dir == pred_dir
    # Treat flat steps as neutral matches only when both are flat.
    return float(np.mean(match) * 100.0)


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Return the full metric bundle for one model."""
    return {
        "RMSE": round(rmse(y_true, y_pred), 4),
        "MAE": round(mae(y_true, y_pred), 4),
        "MAPE": round(mape(y_true, y_pred), 4),
        "R2": round(r2(y_true, y_pred), 4),
        "DirAcc": round(direction_accuracy(y_true, y_pred), 2),
    }


def metrics_table(metrics_by_model: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """Flatten a {model: metrics} dict into rows for display."""
    rows: list[dict[str, Any]] = []
    for model, metrics in metrics_by_model.items():
        row: dict[str, Any] = {"Model": model}
        row.update(metrics)
        rows.append(row)
    return rows
