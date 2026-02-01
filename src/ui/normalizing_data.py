import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from ml_classifier.data import DataInjestor

ticker = ["AAPL", "MSFT"]
start_date = "2018-01-01"  # YYYY-MM-DD
end_date = "2026-01-01"  # YYYY-MM-DD

df = DataInjestor.get(ticker, start_date, end_date)
df.head()
df.info()
df.describe()
df.index
df = df.sort_index()


plt.figure(figsize=(10, 4))
plt.plot(df["Close"])
plt.title(f"{ticker} Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

close_prices = df["Close"]
close_prices.head()

plt.figure(figsize=(10, 4))
plt.plot(close_prices["AAPL"], label="AAPL")
plt.plot(close_prices["MSFT"], label="MSFT")
plt.title("Closing Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
