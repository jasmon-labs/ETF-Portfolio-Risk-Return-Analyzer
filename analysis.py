import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from utils import summary_statistics

def fetch_price_data(tickers, start_date):
    raw = yf.download(tickers, start=start_date, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            return raw.xs("Adj Close", axis=1, level=0).dropna()
        return raw.xs("Close", axis=1, level=0).dropna()
    return raw["Close"].dropna()


def main():
    tickers = ["NIFTYBEES.NS", "BANKBEES.NS", "GOLDBEES.NS"]
    data = fetch_price_data(tickers, "2015-01-01")
    summary = summary_statistics(data)
    print(summary)


if __name__ == "__main__":
    main()
