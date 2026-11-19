"""End-to-end forecasting pipeline.

The pipeline ties together data loading, feature engineering, model fitting,
out-of-sample evaluation and recursive multi-step forecasting. The same
result object feeds both the Gradio application and the offline training /
evaluation scripts, so the demo and the reported numbers come from one code
path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import data as data_module
from . import evaluation, features, models
from .config import Config
from .errors import ModelTrainingError

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Container for everything produced by a pipeline run."""

    source: str
    data: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    dates: pd.DatetimeIndex
    feature_names: list[str]
    splits: dict[str, Any]
    fitted: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    forecasts: dict[str, pd.DataFrame] = field(default_factory=dict)
    losses: dict[str, list[float]] = field(default_factory=dict)
    sample: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _latest_feature_window(df: pd.DataFrame, cfg: Config, window: int = 20) -> pd.DataFrame:
    """Return the most recent supervised feature rows (enough for sequences)."""
    X, _, _, _ = features.build_supervised(df, cfg.features, horizon=1)
    return X.tail(window)


def train_and_evaluate(
    cfg: Config,
    df: pd.DataFrame,
    model_names: list[str] | None = None,
) -> PipelineResult:
    """Fit requested models, evaluate on the held-out test split, forecast ahead."""
    feature_cfg = cfg.features
    horizon = int(cfg.forecast.get("horizon", 5))
    train_ratio = float(cfg.data.get("train_split", 0.8))

    X, y, dates, feature_names = features.build_supervised(df, feature_cfg, horizon=1)
    X_train, X_test, y_train, y_test = features.time_split(X, y, train_ratio)

    result = PipelineResult(
        source="",
        data=df,
        X=X,
        y=y,
        dates=dates,
        feature_names=feature_names,
        splits={
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "train_dates": dates[: len(X_train)],
            "test_dates": dates[len(X_train) :],
        },
        sample=data_module.data_summary(df),
    )

    requested = model_names or cfg.enabled_models()
    for name in requested:
        model_cfg = cfg.model(name)
        try:
            model = models.build_model(name, model_cfg, random_state=int(cfg.data.get("random_state", 42)))
        except ModelTrainingError as exc:
            result.warnings.append(str(exc))
            continue

        try:
            model.fit(X_train, y_train)
        except Exception as exc:  # noqa: BLE001 - keep other models alive
            result.warnings.append(f"{name}: training failed - {exc}")
            continue

        # Predict. The LSTM consumes the full matrix so its sequences cross
        # the train/test boundary cleanly; its head returns NaN for early rows.
        if name == "lstm":
            pred_full = model.predict(X)
            pred_test = pred_full[len(X_train) :]
            pred_test = pred_test[~np.isnan(pred_test)]
            y_test_aligned = y_test.to_numpy()[: len(pred_test)]
        else:
            pred_test = np.asarray(model.predict(X_test), dtype=float)
            y_test_aligned = y_test.to_numpy()

        finite = ~np.isnan(pred_test)
        if finite.sum() < 2:
            result.warnings.append(f"{name}: too few valid predictions to evaluate.")
            continue

        metric = evaluation.compute_metrics(y_test_aligned[finite], pred_test[finite])
        result.fitted[name] = model
        result.metrics[name] = metric
        if hasattr(model, "losses"):
            result.losses[name] = list(model.losses)

        pred_series = pd.Series(pred_test, index=result.splits["test_dates"][: len(pred_test)], name=name)
        if result.predictions.empty:
            result.predictions = pd.DataFrame(
                {"y_true": y_test_aligned}, index=result.splits["test_dates"][: len(y_test_aligned)]
            )
        result.predictions[name] = pred_series

        # Recursive multi-step forecast from the latest history.
        result.forecasts[name] = recursive_forecast(model, df, cfg, horizon=horizon)

    if not result.fitted:
        raise ModelTrainingError("No model could be trained. Check the dataset and configuration.")
    return result


def recursive_forecast(
    model: Any,
    df: pd.DataFrame,
    cfg: Config,
    horizon: int = 5,
) -> pd.DataFrame:
    """Produce a recursive multi-step forecast of the close price.

    At each step the predicted close is appended to the history, features are
    recomputed, and the model predicts the next trading day. Forecast dates use
    business days after the last observed date.
    """
    history = df.copy()
    predictions: list[dict[str, Any]] = []
    last_date = history.index.max()

    for step in range(1, horizon + 1):
        latest_X = _latest_feature_window(history, cfg, window=20)
        pred = np.asarray(model.predict(latest_X), dtype=float)
        pred = pred[~np.isnan(pred)]
        if len(pred) == 0:
            break
        next_close = float(pred[-1])

        next_date = last_date + pd.offsets.BDay(step)
        predictions.append({"Date": next_date, "Close": next_close})

        # Append a synthetic bar so the next iteration can update features.
        last_close = float(history["Close"].iloc[-1])
        last_volume = float(history["Volume"].iloc[-1]) if "Volume" in history.columns else 0.0
        history.loc[next_date] = {
            "Open": last_close,
            "High": max(last_close, next_close),
            "Low": min(last_close, next_close),
            "Close": next_close,
            "Volume": last_volume,
        }

    forecast_df = pd.DataFrame(predictions).set_index("Date") if predictions else pd.DataFrame()
    return forecast_df


def run_pipeline(
    cfg: Config,
    source: str | None = None,
    model_names: list[str] | None = None,
) -> PipelineResult:
    """Load data from ``source`` (ticker or CSV) and run the full pipeline."""
    source = source or str(cfg.data.get("default_ticker", "AAPL"))
    df = data_module.load_dataset(
        source,
        period=str(cfg.data.get("period", "3y")),
        interval=str(cfg.data.get("interval", "1d")),
        min_rows=int(cfg.data.get("min_rows", 80)),
    )
    result = train_and_evaluate(cfg, df, model_names=model_names)
    result.source = source
    return result
