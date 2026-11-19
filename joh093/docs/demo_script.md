# Demonstration Script (3–5 minutes)

## 0. Setup (before the demo)

```bash
python -m pip install -r requirements.txt
# optional: pip install -r requirements-optional.txt   # enables the LSTM
python app.py
```

Confirm the page loads at http://127.0.0.1:7860 and the bundled sample loads
without network access.

## 1. Problem introduction (~45 seconds)

* Topic: stock price forecasting — a time-series regression task.
* Goal: predict the closing price `h` trading days ahead from historical OHLCV
  data and technical indicators.
* State the disclaimer: experimental prototype, not investment advice; past
  patterns do not guarantee future results.

## 2. Model and methodology (~1 minute)

* Data: daily OHLCV bars from Yahoo Finance; chronological 80/20 train/test
  split (no shuffling).
* Features: lagged closes, returns, SMA/EMA, RSI, MACD, Bollinger %b,
  volatility, volume change.
* Models compared: naive baseline, Ridge regression, random forest, gradient
  boosting, optional LSTM.
* Explain the LSTM briefly: sequences of scaled features → two LSTM layers →
  linear output head predicting the next close.

## 3. Live demonstration (~1.5 minutes)

1. Select **Live Yahoo Finance data**, ticker `AAPL`, model **All models**,
   horizon 5 days. Click **Run forecast**.
2. Walk through the summary: data window, last close, best model by RMSE, and
   the 5-day forecast table with percentage change.
3. Show the metrics table: RMSE, MAE, MAPE, R², direction accuracy per model.
4. Inspect the charts:
   * history + forecast extension,
   * held-out predictions vs actual closes,
   * RMSE comparison across models.
5. Switch to the **bundled sample** to show offline operation, or upload a
   local CSV to show schema compatibility.
6. Select a single model (e.g. the LSTM, if installed) and rerun to show model-
   specific forecasts.

## 4. Results and discussion (~45 seconds)

* Identify the best-performing model on the held-out split and its metrics.
* Note where the forecast tracks the actual trend and where it lags or deviates.
* Discuss why tree ensembles or the LSTM may differ, and why the naive baseline
  is a meaningful lower bound.

## 5. Limitations and Q&A (~30 seconds)

* Price-only features; no sentiment, news or fundamentals.
* Non-stationary market regimes; recursive forecast error accumulation.
* No uncertainty bands; models are not investment advice.
* Be ready to explain: feature engineering, train/test leakage prevention,
  LSTM architecture, metric definitions, and how to retrain on another ticker.

## Likely questions

* **Why split by time instead of shuffling?** The test set must represent the
  future; shuffling would leak future information into training.
* **Why MAPE and R² together?** MAPE is scale-relative and easy to interpret;
  R² shows variance explained, while RMSE penalises large errors.
* **How does the multi-day forecast work?** Recursive: each predicted close is
  appended, features are recomputed, and the next day is predicted.
* **How reproducible is this?** Fixed seed, committed configuration, saved
  metrics in `results/metrics.json`.
