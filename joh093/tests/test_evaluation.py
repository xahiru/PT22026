"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np

from stockforecast import evaluation


def test_perfect_prediction_metrics() -> None:
    y = np.array([10.0, 12.0, 14.0, 13.0, 15.0])
    metrics = evaluation.compute_metrics(y, y)

    assert metrics["RMSE"] == 0.0
    assert metrics["MAE"] == 0.0
    assert metrics["R2"] == 1.0
    assert metrics["MAPE"] == 0.0


def test_constant_prediction_r2() -> None:
    y = np.array([10.0, 12.0, 14.0, 13.0, 15.0])
    pred = np.full_like(y, y.mean())
    metrics = evaluation.compute_metrics(y, pred)

    assert metrics["R2"] == 0.0
    assert metrics["RMSE"] > 0


def test_mape_percent_scale() -> None:
    y = np.array([100.0, 200.0, 300.0])
    pred = np.array([110.0, 190.0, 330.0])
    # Errors: 10%, 5%, 10% -> mean 8.333...%
    assert abs(evaluation.mape(y, pred) - 8.3333) < 0.01


def test_direction_accuracy_perfect() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert evaluation.direction_accuracy(y, y) == 100.0
