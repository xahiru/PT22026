# Ethics and Responsible Use

## 1. Not financial advice

This project is a machine-learning course prototype. It does not provide
investment, legal or tax advice, and its forecasts must not be used to buy,
sell or hold any financial instrument. Real trading decisions involve risk of
financial loss and require licensed professionals and up-to-date information.

## 2. Limits of historical price data

Stock prices reflect news, macroeconomic conditions, interest rates, investor
sentiment, company performance and global events. A model trained only on past
prices cannot represent these drivers. Forecasts can be wrong, especially
during crises, sentiment shifts or sudden news events.

## 3. Fairness and misuse

* The system does not make decisions about people, credit, employment or access
  to services.
* It must not be repurposed for automated trading, market manipulation, or
  misleading financial content presented as professional analysis.
* Predictions should always be labelled as experimental and uncertain.

## 4. Data provenance

* Market data comes from Yahoo Finance through `yfinance` and is used for
  study purposes under its terms.
* No personal data is collected or processed by the application.
* Raw downloaded datasets are not committed to the repository; only a small
  historical sample is included.

## 5. Reproducibility and transparency

* Configuration, features and hyper-parameters are documented in `config/` and
  the technical report.
* Metrics are computed on a chronological hold-out split so performance claims
  are checkable.
* External libraries and data sources are listed in the README and report
  references.

## 6. Responsible presentation

When demonstrating the project:

1. State upfront that the prototype is experimental and not investment advice.
2. Show both the forecast and its measured out-of-sample error.
3. Avoid presenting any single model as a reliable predictor of the market.
4. Answer questions about limitations honestly, including what the model does
   not observe.
