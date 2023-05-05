import pandas as pd
import ta as ta
# from ta.momentum import RSIIndicator, AwesomeOscillatorIndicator
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt

# Get historical data for SPY
ticker = yf.Ticker("SPY").history(period="5d", interval="1m")
Close = ticker["Close"]
ticker['AwesomeOsc'] = ta.momentum.awesome_oscillator(high=ticker["High"], low=ticker["Low"], window1 = 1, window2 = 5, fillna= False)
ticker['RSI'] = ta.momentum.rsi(close=Close, window= 5, fillna= False)
groups = ticker.groupby(ticker.index.date)
group_dates = list(groups.groups.keys())
second_to_last_date = group_dates[-1]
ticker = groups.get_group(second_to_last_date)

###TODO add rsi to main csv.


# Save the DataFrame with the technical analysis indicators to a CSV file
# ticker = ticker.tail(240)
# ticker.plot()
# plt.show()
ticker.to_csv("ta.csv")
print("3")
# Plot the technical analysis indicators
# fig, ax = plt.subplots()

print("4")
# ticker["MACD_12_26_9"].plot(ax=ax, label="MACD")
# ticker["STOCHRSI_14"].plot(ax=ax, label="Stochastic RSI")
# ax.legend()

print("5")