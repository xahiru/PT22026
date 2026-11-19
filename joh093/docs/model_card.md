# Model Card — Stock Price Forecasting

## Intended use

* **Purpose:** Academic demonstration of time-series regression for short-term
  equity close-price forecasting, as part of the Professional Training course.
* **Intended users:** Course assessors and students running the demo.
* **Not intended for:** Investment decisions, automated trading, advisory
  services, or any real financial decision-making.

## Inputs

* Daily OHLCV bars (`Date, Open, High, Low, Close, Volume`) from Yahoo Finance
  or a local CSV; a bundled AAPL sample is included for offline use.
* Engineered features: lagged closes, returns, SMA/EMA, RSI, MACD, Bollinger
  %b, rolling volatility, volume change.

## Models

| Model | Description |
|---|---|
| Naive baseline | Persistence forecast; predicts the last close. |
| Ridge regression | L2-regularised linear regression on standardised features. |
| Random forest | Bagged regression trees (300 trees, depth-limited). |
| Gradient boosting | HistGradientBoosting regression over sequential trees. |
| LSTM | Two-layer LSTM with a linear output head, trained on scaled sequences. |

## Outputs

* Next-day and multi-day ahead close-price forecasts.
* Out-of-sample metrics: RMSE, MAE, MAPE, R², direction accuracy.
* Charts of history, forecast extension and held-out predictions.

## Training data

* Yahoo Finance daily bars; default look-back 3 years. The exact window and
  ticker are configurable in `config/app.yaml`.
* The chronological 80/20 split keeps the test set strictly later in time.

## Performance

Measured metrics are written to `results/metrics.json` after running
`scripts/run_pipeline.py`. See the technical report (Section 7) for the
comparison table and charts on the selected dataset.

## Capabilities

* Compares a simple baseline against non-linear tree ensembles and a recurrent
  network on the same feature set.
* Works with any OHLCV CSV with the standard schema, not only one ticker.
* Produces interpretable accuracy metrics and direction-of-movement accuracy.

## Limitations

* Price series are non-stationary and noisy; models may fail under market
  regime changes.
* No sentiment, fundamental, macroeconomic or news features are used.
* Tree models cannot extrapolate outside the training price range.
* The LSTM may overfit short noisy series and is sensitive to hyper-parameters.
* Forecasts are not probabilistic; uncertainty bands are not produced.

## Ethical considerations

The system must not be used to advise financial decisions. Forecasts must be
presented with the disclaimer that past patterns do not guarantee future
results, and that market outcomes depend on factors outside historical prices.
