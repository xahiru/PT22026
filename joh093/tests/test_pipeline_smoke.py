"""End-to-end smoke test on the bundled offline sample."""

from __future__ import annotations

import numpy as np

from stockforecast import data, pipeline
from stockforecast.config import load_config


def test_pipeline_runs_on_sample() -> None:
    cfg = load_config()
    df = data.load_sample()

    result = pipeline.train_and_evaluate(cfg, df, model_names=["naive", "ridge"])

    assert {"naive", "ridge"} <= set(result.metrics)
    for metrics in result.metrics.values():
        for key in ("RMSE", "MAE", "MAPE", "R2"):
            assert np.isfinite(metrics[key]), f"{key} is not finite"

    assert not result.predictions.empty
    assert "y_true" in result.predictions.columns

    horizon = int(cfg.forecast.get("horizon", 5))
    for forecast in result.forecasts.values():
        assert len(forecast) == horizon
        assert (forecast["Close"] > 0).all()


def test_dataset_summary_consistent() -> None:
    df = data.load_sample()
    summary = data.data_summary(df)
    assert summary["rows"] == len(df)
    assert summary["last_close"] > 0
    assert summary["start"] <= summary["end"]
