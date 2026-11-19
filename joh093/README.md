# Stock Price Forecasting

**Next-day and short-horizon equity price prediction with classical machine
learning and an LSTM, served through an interactive Gradio web application**

> **Author** · Joseph R Johnson — `20243120093`
> **Course** · Professional Training (`YN3012170034`)
> **Project type** · Individual, Python, Gradio web application
> **Repository path** · `joh093/`

---

## 1. What this is

This project forecasts short-term stock price movements from historical OHLCV
(open, high, low, close, volume) data. The user picks a ticker (or uploads a
CSV / uses the bundled sample), selects a model and a forecast horizon, and the
application returns:

* a next-day and multi-day ahead forecast of the closing price,
* out-of-sample accuracy metrics (RMSE, MAE, MAPE, R², direction accuracy),
* charts comparing observed prices, model forecasts and held-out predictions.

Five forecasters are compared: a persistence baseline, Ridge regression, a
random forest, gradient-boosted trees, and an optional two-layer LSTM.

> **Disclaimer.** This is an experimental research prototype built for a course
> assignment. It is **not investment advice**. Stock prices are noisy,
> non-stationary and influenced by news, macroeconomics and sentiment that
> historical price data alone cannot capture. Past patterns do not guarantee
> future results; do not base financial decisions on these forecasts.

### Problem statement

Equity price forecasting is a classic time-series regression problem: given a
window of past prices and derived technical indicators, predict the closing
price `h` trading days ahead. Unlike a static classification task, the target
moves with the market, so models are evaluated on a **chronological hold-out
split** (train on the past, test on the future) rather than shuffled cross-
validation.

---

## 2. Quick start

```bash
# 1. (Recommended) virtual environment
python -m venv .venv
# PowerShell:  . .venv/Scripts/activate
# bash/zsh:    source .venv/bin/activate

# 2. Install core dependencies
python -m pip install -r requirements.txt

# 3. Run the web application
python app.py
# Open http://127.0.0.1:7860
```

**Optional — enable the LSTM model.** The core app runs with Ridge, random
forest, gradient boosting and the naive baseline. To add the LSTM, install the
CPU PyTorch wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
# or:  pip install -r requirements-optional.txt
```

### Other commands

```bash
# Train all enabled models and save metrics + figures to results/
python scripts/run_pipeline.py

# Train a specific ticker / horizon / subset of models
python scripts/run_pipeline.py --source MSFT --horizon 7 --models ridge,random_forest

# Regenerate figures from saved results
python scripts/make_figures.py

# Run the test suite
pytest
```

### Environment options

| Variable | Default | Meaning |
|---|---|---|
| `STOCKFORECAST_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `STOCKFORECAST_DEFAULT_TICKER` | `AAPL` | Default ticker in the interface |
| `STOCKFORECAST_SEED` | `42` | Random seed for reproducibility |

---

## 3. Pipeline

```
                 ┌──────────────────────────────┐
 ticker / CSV →  │ data.py                       │
 bundled sample  │ download (yfinance) → validate│
                 └───────────────┬──────────────┘
                                 │
                 ┌───────────────▼──────────────┐
                 │ features.py                    │
                 │ indicators + lag features     │
                 │ RSI · MACD · Bollinger · SMA/EMA · volatility
                 └───────────────┬──────────────┘
                                 │
                 ┌───────────────▼──────────────┐
                 │ models.py                     │
                 │ naive · ridge · random forest │
                 │ gradient boosting · LSTM      │
                 └───────────────┬──────────────┘
                                 │
                 ┌───────────────▼──────────────┐
                 │ pipeline.py → evaluation.py   │
                 │ chronological train/test split │
                 │ RMSE · MAE · MAPE · R² · DirAcc
                 └───────────────┬──────────────┘
                                 │
                 ┌───────────────▼──────────────┐
                 │ ui/app_ui.py (Gradio)         │
                 │ summary · metrics · charts     │
                 └───────────────────────────────┘
```

---

## 4. Project structure

```
joh093/
├── app.py                     # Gradio application entry point
├── requirements.txt           # Core runtime dependencies
├── requirements-optional.txt  # Optional PyTorch (LSTM) dependency
├── Makefile                   # Convenience commands (setup/run/train/test)
├── pytest.ini
├── config/
│   ├── app.yaml               # Data, feature and forecast settings
│   └── models.yaml            # Per-model hyper-parameters
├── data/
│   ├── README.md
│   └── sample_aapl.csv        # Bundled offline sample (committed)
├── docs/
│   ├── architecture.md         # System design and module responsibilities
│   ├── model_card.md           # Model card: capabilities and limitations
│   ├── ethics.md               # Ethics and responsible-use notes
│   └── demo_script.md          # 3–5 minute demonstration walk-through
├── models/                     # Fitted weights (git-ignored; .gitkeep)
├── results/
│   ├── metrics.json           # Saved metrics, forecasts and losses
│   ├── predictions.csv        # Held-out predictions per model
│   └── figures/                # Rendered charts
├── report/
│   ├── figures/                # Figures used in the technical report
│   └── Stock_Price_Forecasting_Report_20243120093.pdf
├── scripts/
│   ├── download_data.py       # Download Yahoo Finance history to CSV
│   ├── run_pipeline.py         # Train, evaluate and save artifacts
│   └── make_figures.py         # Regenerate charts from saved results
├── src/stockforecast/
│   ├── config.py               # Configuration loading and paths
│   ├── data.py                # Data download, validation, normalisation
│   ├── features.py            # Indicators, lag features, sequences
│   ├── models.py              # Model implementations and factory
│   ├── evaluation.py           # Metric functions
│   ├── pipeline.py             # End-to-end orchestration
│   ├── charts.py              # Matplotlib chart helpers
│   ├── errors.py               # Project-specific exceptions
│   └── ui/
│       └── app_ui.py           # Gradio Blocks interface
└── tests/
    ├── test_features.py
    ├── test_evaluation.py
    └── test_pipeline_smoke.py
```

---

## 5. Data

* **Source:** Yahoo Finance daily bars via `yfinance`; the bundled
  `data/sample_aapl.csv` provides an offline fallback.
* **Schema:** `Date, Open, High, Low, Close, Volume`.
* **Split:** time-ordered, 80% train / 20% test (see `config/app.yaml`).
* **Large datasets are not committed.** Download fresh data with
  `scripts/download_data.py`; downloaded CSVs are git-ignored.

## 6. Models

| Model | Type | Idea |
|---|---|---|
| Naive baseline | Persistence | Predicts the last observed close; lower bound to beat. |
| Ridge regression | Linear model | Standardised linear regression with L2 regularisation on engineered features. |
| Random forest | Bagged trees | Ensemble of regression trees; captures non-linear feature interactions. |
| Gradient boosting | Boosted trees | Sequentially fitted trees that correct prior residuals (HistGradientBoosting). |
| LSTM | Recurrent neural network | Two-layer LSTM over scaled price/feature sequences; optional PyTorch dependency. |

Feature set: lagged closes (1, 2, 3, 4, 5, 10, 20 days), daily return, SMA(5,
20), EMA(12, 26), RSI(14), MACD line/signal/histogram, Bollinger %b, rolling
volatility and volume change.

### Model strengths and limitations

* **Linear models** are fast and interpretable but assume linear relationships
  and struggle with regime changes.
* **Tree ensembles** model non-linearities and feature interactions but can
  extrapolate poorly outside the training price range.
* **The LSTM** models sequential dependence but needs more data and tuning, and
  can overfit noisy price series; it is included as a comparison rather than a
  guaranteed best forecaster.
* No model here uses sentiment, fundamental or macroeconomic inputs, which
  limits real-world accuracy.

## 7. Evaluation metrics

| Metric | Definition |
|---|---|
| RMSE | Root mean squared error; penalises large errors. |
| MAE | Mean absolute error; robust to outliers. |
| MAPE | Mean absolute percentage error; error relative to price level (%). |
| R² | Share of variance explained; 1.0 is perfect, 0.0 matches the mean. |
| Direction accuracy | Share of days where the predicted up/down direction matches reality. |

## 8. Configuration

All tunable settings live in `config/`:

* `app.yaml` — default ticker, period, interval, train split, lag windows,
  indicator parameters, forecast horizon.
* `models.yaml` — model switches and hyper-parameters (trees, LSTM size,
  epochs, learning rate, batch size).

## 9. Tests

```bash
pytest
```

The suite covers indicator sanity checks, metric calculations and an
end-to-end pipeline smoke test on the bundled sample (naive + ridge).

## 10. Repository and submission notes

* The project lives in the assigned folder `joh093/` inside the shared
  `PT22026` repository.
* `.gitignore` excludes virtual environments, model weights, downloaded
  datasets, caches and secrets; only the small sample CSV and generated report
  figures are committed.
* No large model checkpoints or raw datasets are uploaded; data is fetched or
  regenerated via the scripts above.
* The official submission is a Pull Request against the main repository;
  direct pushes to the main branch are not permitted.

## 11. References

1. Yahoo Finance — historical market data, accessed through `yfinance`.
   https://pypi.org/project/yfinance/
2. Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice*,
   3rd ed. OTexts, 2021. https://otexts.com/fpp3/
3. Hochreiter, S., & Schmidhuber, J. "Long Short-Term Memory." *Neural
   Computation*, 9(8), 1997.
4. Breiman, L. "Random Forests." *Machine Learning*, 45, 2001.
5. Pedregosa, F. et al. "Scikit-learn: Machine Learning in Python." *JMLR*, 12,
   2011.
6. Vaswani, A. et al. (reference background on sequence models); see report
   references for the full cited list.
