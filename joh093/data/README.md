# Data directory

## Contents

| File | Description |
|---|---|
| `sample_aapl.csv` | Small bundled offline sample: daily AAPL OHLCV bars. Committed so the app and tests run without network access. |

Downloaded datasets (e.g. `data/msft.csv`) are **git-ignored** and are not
committed to the repository, in line with the assignment rules on large files.

## Source

Market data is fetched at runtime from **Yahoo Finance** through the
[`yfinance`](https://pypi.org/project/yfinance/) package. The CSV schema is:

| Column | Meaning |
|---|---|
| `Date` | Trading date (index) |
| `Open` | Opening price |
| `High` | Highest price of the bar |
| `Low` | Lowest price of the bar |
| `Close` | Adjusted closing price |
| `Volume` | Shares traded |

## Refreshing the sample

```bash
# Download 3 years of daily AAPL bars and overwrite the bundled sample
python scripts/download_data.py --ticker AAPL --period 3y --out data/sample_aapl.csv

# Download another ticker to a local (git-ignored) CSV
python scripts/download_data.py --ticker MSFT --out data/msft.csv
```

## Notes

* Yahoo Finance provides delayed market data; it is intended for study and
  experimentation only.
* The bundled sample covers a fixed historical window and does not update
  automatically. Use the live data source in the app for recent prices.
* Past price patterns do not guarantee future results; see the disclaimer in
  the application interface and the technical report.
