"""Gradio interface for the stock forecasting application.

The interface lets the user choose a ticker (or upload a CSV / use the bundled
sample), pick a model and a forecast horizon, and inspect:

* a plain-language forecast summary,
* out-of-sample accuracy metrics,
* the price history with the forecast extension,
* held-out predictions against actual closes,
* a model comparison chart.

All failures are caught and reported as readable messages; no stack trace
reaches the browser.
"""

from __future__ import annotations

import logging
from typing import Any

import gradio as gr
import pandas as pd

from .. import charts, evaluation
from .. import data as data_module
from ..config import Config, load_config
from ..errors import StockForecastError
from ..models import available_model_names
from ..pipeline import PipelineResult, train_and_evaluate

log = logging.getLogger(__name__)

SAMPLE_OPTION = "Bundled sample (offline)"
LIVE_OPTION = "Live Yahoo Finance data"
CSV_OPTION = "Upload CSV file"

ALL_MODELS = "all"


def _best_model(result: PipelineResult) -> str | None:
    """Return the model with the lowest RMSE, or ``None``."""
    if not result.metrics:
        return None
    return min(result.metrics, key=lambda name: result.metrics[name]["RMSE"])


def _forecast_markdown(result: PipelineResult, horizon: int) -> str:
    sample = result.sample
    best = _best_model(result)
    lines: list[str] = []
    lines.append("### Forecast summary")
    lines.append(
        f"- **Data source:** {result.source} · {sample['rows']} trading days, "
        f"{sample['start']} → {sample['end']}"
    )
    lines.append(f"- **Last observed close:** {sample['last_close']:.2f}")
    lines.append(
        f"- **Models evaluated:** {', '.join(result.metrics.keys())}"
    )
    if best is not None:
        metric = result.metrics[best]
        lines.append(
            f"- **Best model (lowest RMSE):** {best} "
            f"(RMSE {metric['RMSE']:.2f}, MAE {metric['MAE']:.2f}, "
            f"MAPE {metric['MAPE']:.2f}%)"
        )
        forecast = result.forecasts.get(best)
        if forecast is not None and not forecast.empty:
            last_close = sample["last_close"]
            lines.append("")
            lines.append(f"**{horizon}-day ahead forecast ({best}):**")
            lines.append("| Date | Predicted close | Change vs last close |")
            lines.append("|---|---|---|")
            for date, row in forecast.iterrows():
                close = float(row["Close"])
                change = (close - last_close) / last_close * 100.0
                lines.append(
                    f"| {date.strftime('%Y-%m-%d')} | {close:.2f} | {change:+.2f}% |"
                )
    if result.warnings:
        lines.append("")
        lines.append("**Notes:**")
        for warning in result.warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines)


def _metrics_dataframe(result: PipelineResult) -> pd.DataFrame:
    rows = evaluation.metrics_table(result.metrics)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.rename(
        columns={
            "RMSE": "RMSE",
            "MAE": "MAE",
            "MAPE": "MAPE (%)",
            "R2": "R²",
            "DirAcc": "Direction accuracy (%)",
        }
    )
    return df


def _run_pipeline(
    source_type: str,
    ticker: str,
    model_name: str,
    horizon: int,
    csv_file: Any,
    cfg: Config,
) -> PipelineResult:
    cfg.raw.setdefault("forecast", {})["horizon"] = int(horizon)

    if source_type == SAMPLE_OPTION:
        df = data_module.load_sample()
    elif source_type == CSV_OPTION and csv_file is not None:
        df = data_module.load_dataset(
            csv_file.name,
            min_rows=int(cfg.data.get("min_rows", 80)),
        )
    else:
        df = data_module.load_dataset(
            ticker,
            period=str(cfg.data.get("period", "3y")),
            interval=str(cfg.data.get("interval", "1d")),
            min_rows=int(cfg.data.get("min_rows", 80)),
        )

    model_names = None if model_name == ALL_MODELS else [model_name]
    return train_and_evaluate(cfg, df, model_names=model_names)


def build_demo(cfg: Config | None = None) -> gr.Blocks:
    """Build the Gradio Blocks application."""
    cfg = cfg or load_config()
    project = cfg.project

    model_choices = [("All models", ALL_MODELS)] + [
        (charts.MODEL_LABELS.get(name, name), name) for name in available_model_names()
    ]

    with gr.Blocks(title=project.get("name", "Stock Price Forecasting")) as demo:
        gr.Markdown(
            f"# {project.get('name', 'Stock Price Forecasting')}\n\n"
            f"**{project.get('subtitle', '')}**\n\n"
            f"**Student:** {project.get('student_name', '')} · "
            f"**ID:** {project.get('student_id', '')} · "
            f"**Course:** {project.get('course', '')} ({project.get('course_code', '')})\n\n"
            "> **Disclaimer:** This is an experimental research prototype built for a "
            "course assignment. It is **not investment advice**, and past price patterns "
            "do not guarantee future results. Do not base financial decisions on these forecasts."
        )

        with gr.Row():
            source_type = gr.Radio(
                choices=[LIVE_OPTION, SAMPLE_OPTION, CSV_OPTION],
                value=LIVE_OPTION,
                label="Data source",
            )
            ticker = gr.Textbox(
                value=str(cfg.data.get("default_ticker", "AAPL")),
                label="Ticker symbol (Yahoo Finance)",
                placeholder="e.g. AAPL, MSFT, TSLA",
            )
            csv_file = gr.File(label="Upload OHLCV CSV (Date, Open, High, Low, Close, Volume)")

        with gr.Row():
            model_name = gr.Dropdown(
                choices=model_choices,
                value=ALL_MODELS,
                label="Model",
            )
            horizon = gr.Slider(
                minimum=1, maximum=10, value=int(cfg.forecast.get("horizon", 5)),
                step=1, label="Forecast horizon (trading days)",
            )
            run_btn = gr.Button("Run forecast", variant="primary")

        summary = gr.Markdown()
        metrics_table = gr.Dataframe(label="Out-of-sample metrics", wrap=True)

        with gr.Row():
            history_plot = gr.Plot(label="History and forecast")
            test_plot = gr.Plot(label="Held-out predictions vs actual")
        metrics_plot = gr.Plot(label="Model comparison")

        def _handle(source_type, ticker, model_name, horizon, csv_file):
            try:
                result = _run_pipeline(source_type, ticker, model_name, horizon, csv_file, cfg)
            except StockForecastError as exc:
                return f"**Could not produce a forecast:** {exc}", pd.DataFrame(), None, None, None
            except Exception as exc:  # noqa: BLE001 - last-resort UI guard
                log.exception("Unexpected pipeline failure")
                return f"**Unexpected error:** {exc}", pd.DataFrame(), None, None, None

            best = _best_model(result)
            forecast_df = result.forecasts.get(best) if best else None

            fig_history = charts.plot_history(result.data, forecast_df, title="Close price and forecast")
            fig_test = charts.plot_test_predictions(result.predictions)
            fig_metrics = charts.plot_metrics_bar(result.metrics, metric="RMSE")

            return (
                _forecast_markdown(result, int(horizon)),
                _metrics_dataframe(result),
                fig_history,
                fig_test,
                fig_metrics,
            )

        run_btn.click(
            fn=_handle,
            inputs=[source_type, ticker, model_name, horizon, csv_file],
            outputs=[summary, metrics_table, history_plot, test_plot, metrics_plot],
        )

    return demo
