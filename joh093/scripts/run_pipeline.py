#!/usr/bin/env python
"""Train, evaluate and save the forecasting pipeline results.

This script reproduces every number and chart used in the technical report.
It loads the configured dataset, fits the enabled models, evaluates them on the
held-out test split, writes metrics and predictions to ``results/`` and renders
figures to ``results/figures/``.

Examples
--------
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --source MSFT --horizon 7
    python scripts/run_pipeline.py --source data/sample_aapl.csv --models ridge,random_forest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stockforecast import charts  # noqa: E402
from stockforecast.config import FIGURES_DIR, RESULTS_DIR, load_config  # noqa: E402
from stockforecast.pipeline import run_pipeline  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and save the forecasting pipeline.")
    parser.add_argument("--source", default=None, help="ticker symbol or CSV path")
    parser.add_argument("--models", default=None, help="comma-separated model names")
    parser.add_argument("--horizon", type=int, default=None, help="forecast horizon (trading days)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_config()

    if args.horizon is not None:
        cfg.raw.setdefault("forecast", {})["horizon"] = args.horizon
    model_names = args.models.split(",") if args.models else None

    result = run_pipeline(cfg, source=args.source, model_names=model_names)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Persist metrics and metadata ---------------------------------------
    payload: dict[str, object] = {
        "source": result.source,
        "sample": result.sample,
        "feature_names": result.feature_names,
        "n_train": len(result.splits["y_train"]),
        "n_test": len(result.splits["y_test"]),
        "metrics": result.metrics,
        "warnings": result.warnings,
        "forecasts": {
            name: df.reset_index().to_dict(orient="records")
            for name, df in result.forecasts.items()
            if not df.empty
        },
        "losses": result.losses,
    }
    metrics_path = RESULTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # --- Persist test predictions -------------------------------------------
    predictions_path = RESULTS_DIR / "predictions.csv"
    result.predictions.to_csv(predictions_path, index_label="Date")

    for name, forecast in result.forecasts.items():
        if not forecast.empty:
            forecast.to_csv(RESULTS_DIR / f"forecast_{name}.csv", index_label="Date")

    # --- Render figures ------------------------------------------------------
    best = min(result.metrics, key=lambda m: result.metrics[m]["RMSE"]) if result.metrics else None
    forecast_df = result.forecasts.get(best) if best else None

    fig_history = charts.plot_history(result.data, forecast_df, title="Close price and forecast")
    charts.save_figure(fig_history, FIGURES_DIR / "history_forecast.png")

    fig_test = charts.plot_test_predictions(result.predictions)
    charts.save_figure(fig_test, FIGURES_DIR / "test_predictions.png")

    fig_metrics = charts.plot_metrics_bar(result.metrics, metric="RMSE")
    charts.save_figure(fig_metrics, FIGURES_DIR / "metrics_rmse.png")

    if result.losses:
        fig_loss = charts.plot_losses(result.losses)
        charts.save_figure(fig_loss, FIGURES_DIR / "training_loss.png")

    # --- Console summary -----------------------------------------------------
    print("\n=== Pipeline summary ===")
    print(f"Source: {result.source} | train: {payload['n_train']} | test: {payload['n_test']}")
    print(f"{'Model':<22}{'RMSE':>10}{'MAE':>10}{'MAPE %':>10}{'R2':>10}")
    for name, metric in result.metrics.items():
        print(f"{name:<22}{metric['RMSE']:>10.2f}{metric['MAE']:>10.2f}"
              f"{metric['MAPE']:>10.2f}{metric['R2']:>10.3f}")
    print(f"\nArtifacts saved to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
