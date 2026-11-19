#!/usr/bin/env python
"""Regenerate figures from saved ``results/`` artifacts.

Use this after editing chart styling or rebuilding the report: it reads
``results/metrics.json`` and ``results/predictions.csv`` and re-renders every
chart without retraining the models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stockforecast import charts  # noqa: E402
from stockforecast.config import FIGURES_DIR, REPORT_DIR, RESULTS_DIR  # noqa: E402


def main() -> int:
    metrics_path = RESULTS_DIR / "metrics.json"
    predictions_path = RESULTS_DIR / "predictions.csv"
    if not metrics_path.exists() or not predictions_path.exists():
        print("Run scripts/run_pipeline.py first to produce results artifacts.", file=sys.stderr)
        return 1

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_path, index_col=0, parse_dates=True)

    report_figures = REPORT_DIR / "figures"
    report_figures.mkdir(parents=True, exist_ok=True)

    fig_test = charts.plot_test_predictions(predictions)
    charts.save_figure(fig_test, FIGURES_DIR / "test_predictions.png")
    charts.save_figure(fig_test, report_figures / "test_predictions.png")

    fig_metrics = charts.plot_metrics_bar(payload["metrics"], metric="RMSE")
    charts.save_figure(fig_metrics, FIGURES_DIR / "metrics_rmse.png")
    charts.save_figure(fig_metrics, report_figures / "metrics_rmse.png")

    if payload.get("losses"):
        fig_loss = charts.plot_losses(payload["losses"])
        charts.save_figure(fig_loss, FIGURES_DIR / "training_loss.png")
        charts.save_figure(fig_loss, report_figures / "training_loss.png")

    print(f"Figures regenerated in {FIGURES_DIR} and {report_figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
