"""Matplotlib chart helpers shared by the app, scripts and report.

Charts use a restrained, consistent palette. All functions return a
``matplotlib.figure.Figure`` so callers can either display it in Gradio or save
it to the results / report folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Consistent colour per model across every chart.
MODEL_COLORS: dict[str, str] = {
    "naive": "#7f7f7f",
    "ridge": "#2c7fb8",
    "random_forest": "#31a354",
    "gradient_boosting": "#e6550d",
    "lstm": "#756bb1",
}

MODEL_LABELS: dict[str, str] = {
    "naive": "Naive baseline",
    "ridge": "Ridge regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "lstm": "LSTM",
}

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    }
)


def _style_legend(ax: plt.Axes) -> None:
    ax.legend(loc="best", frameon=True, fontsize=9)


def plot_history(
    data_df: pd.DataFrame,
    forecast_df: pd.DataFrame | None = None,
    title: str = "Close price history",
    tail: int | None = None,
) -> plt.Figure:
    """Plot observed close prices, optionally with a forecast extension."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    history = data_df["Close"]
    if tail is not None:
        history = history.tail(tail)
    ax.plot(history.index, history.values, color="#252525", linewidth=1.4, label="Observed close")

    if forecast_df is not None and not forecast_df.empty:
        # Join the last observed close to the forecast so the line connects.
        bridge = pd.Series([float(history.iloc[-1])], index=[history.index[-1]])
        forecast = pd.concat([bridge, forecast_df["Close"]])
        ax.plot(forecast.index, forecast.values, color="#d95f0e", linewidth=1.6,
                linestyle="--", label="Forecast")
        ax.scatter(forecast_df.index, forecast_df["Close"], color="#d95f0e", s=22, zorder=3)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close price")
    _style_legend(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_test_predictions(predictions: pd.DataFrame, title: str = "Out-of-sample forecast vs actual") -> plt.Figure:
    """Plot actual held-out closes against each model's predictions."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(predictions.index, predictions["y_true"], color="#252525", linewidth=1.6, label="Actual close")

    for model in predictions.columns:
        if model == "y_true":
            continue
        color = MODEL_COLORS.get(model, "#636363")
        label = MODEL_LABELS.get(model, model)
        ax.plot(predictions.index, predictions[model], linewidth=1.2, linestyle="--",
                color=color, label=label, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close price")
    _style_legend(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_metrics_bar(metrics: dict[str, dict[str, float]], metric: str = "RMSE") -> plt.Figure:
    """Bar chart comparing one metric across models (lower is better)."""
    names = list(metrics.keys())
    values = [metrics[m].get(metric, np.nan) for m in names]
    labels = [MODEL_LABELS.get(m, m) for m in names]
    colors = [MODEL_COLORS.get(m, "#636363") for m in names]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(labels, values, color=colors, edgecolor="#252525", linewidth=0.6)
    ax.set_title(f"Model comparison - {metric} (lower is better)")
    ax.set_ylabel(metric)
    for bar, value in zip(bars, values):
        if not np.isnan(value):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}",
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


def plot_losses(losses: dict[str, list[float]], title: str = "LSTM training loss") -> plt.Figure:
    """Plot training MSE curves for neural models."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for model, curve in losses.items():
        if not curve:
            continue
        ax.plot(range(1, len(curve) + 1), curve, color=MODEL_COLORS.get(model, "#756bb1"),
                linewidth=1.5, label=MODEL_LABELS.get(model, model))
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (scaled prices)")
    _style_legend(ax)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: Path | str) -> Path:
    """Save a figure to disk and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path
