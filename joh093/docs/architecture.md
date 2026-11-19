# System Architecture

## 1. Layered design

The project follows a small layered architecture so that data, features,
models, evaluation and the interface are independently replaceable:

| Layer | Module | Responsibility |
|---|---|---|
| Interface | `src/stockforecast/ui/app_ui.py` | Gradio Blocks, input collection, output rendering, user-facing error handling. |
| Orchestration | `src/stockforecast/pipeline.py` | Data preparation, train/test split, model fitting, evaluation, recursive forecasting. |
| Modelling | `src/stockforecast/models.py` | Model implementations and the model factory. |
| Features | `src/stockforecast/features.py` | Technical indicators, lag features, sequence construction, time split. |
| Data | `src/stockforecast/data.py` | Yahoo Finance download, CSV loading, validation and normalisation. |
| Evaluation | `src/stockforecast/evaluation.py` | RMSE, MAE, MAPE, R², direction accuracy. |
| Presentation | `src/stockforecast/charts.py` | Shared Matplotlib figures for the app and report. |
| Configuration | `src/stockforecast/config.py` | YAML loading, paths, environment overrides. |

## 2. Data flow

1. **Input.** The user supplies a ticker symbol, uploads an OHLCV CSV, or uses
   the bundled sample. `data.py` validates schema, sorts by date, removes
   duplicates and missing closes, and coerces numeric columns.
2. **Features.** `features.py` computes momentum, trend, volatility and
   Bollinger-band indicators, then adds lagged closes. The target is the close
   price `horizon` days ahead.
3. **Split.** Rows are split chronologically (80/20); no shuffling, so the test
   set always represents the future relative to training data.
4. **Models.** Each forecaster implements `fit(X, y)` / `predict(X)`. Scikit-
   learn models use a standardised pipeline; the LSTM builds scaled sequences
   and trains a two-layer recurrent network.
5. **Evaluation.** Metrics are computed on valid out-of-sample predictions.
6. **Forecast.** A recursive loop appends each predicted close to the history,
   recomputes features, and predicts the next trading day for the requested
   horizon.

## 3. Design decisions

* **Single code path for demo and report.** `pipeline.py` is used by both the
  Gradio app and `scripts/run_pipeline.py`, so demonstrated and reported numbers
  come from the same implementation.
* **Optional deep-learning dependency.** PyTorch is isolated to `models.py`;
  without it the four classical models still run end to end.
* **Failures are legible.** Project-specific exceptions in `errors.py` are
  translated into readable messages in the UI instead of stack traces.
* **Reproducibility.** A fixed random seed (configurable via
  `STOCKFORECAST_SEED`) controls tree ensembles and the LSTM initialisation.
* **No data leakage.** Indicators and lags are computed only from past and
  current rows; the target is a future shift, and the split respects time order.

## 4. Trade-offs

* Recursive multi-step forecasting accumulates errors because each predicted
  value feeds the next step; direct multi-step models would avoid this but need
  one head per horizon.
* Tree models cannot extrapolate beyond the observed price range; the LSTM can
  model sequences but is more data-hungry and easier to overfit.
* Price-only features exclude sentiment and fundamentals, deliberately keeping
  the scope reproducible and understandable.
