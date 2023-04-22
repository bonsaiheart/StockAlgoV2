import ta
import yfinance as yf


###TODO add def to add ta to the main df.
ticker = yf.Ticker("SPY").history(
    period="1mo", interval="5m"
)
print(ticker)
ta = ta.add_momentum_ta(df=ticker,high="High",low="Low",close="Close",volume="Volume")
ta.to_csv("ta.csv")