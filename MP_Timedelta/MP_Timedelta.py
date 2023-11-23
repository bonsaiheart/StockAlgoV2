import datetime as dt
import os
from pathlib import Path
import pandas as pd
import yfinance as yf

def convert_date_format(date_str):
    """Converts YYMMDD to YYYY-MM-DD."""
    year = date_str[:2]
    month = date_str[2:4]
    day = date_str[4:6]
    return f"20{year}-{month}-{day}"


def generate_and_save_max_pain_for_ticker(ticker, base_dir="data/ProcessedData"):
    ticker_path = os.path.join(base_dir, ticker)
    ticker_data = {}  # Will contain data for the current ticker

    if not os.path.isdir(ticker_path):  # Check if it's a directory
        return

    for date_dir in os.listdir(ticker_path):
        date_path = os.path.join(ticker_path, date_dir)

        sorted_filenames = sorted(os.listdir(date_path))

        for filename in sorted_filenames:
            if filename.endswith(".csv") and filename >= f"{ticker}_{date_dir}_0930":
                filepath = os.path.join(date_path, filename)

                try:
                    df = pd.read_csv(filepath)
                except pd.errors.EmptyDataError:
                    print(f"Skipping empty or malformed file: {filepath}")
                    continue

                if "ExpDate" in df.columns and "Maximum Pain" in df.columns:
                    for _, row in df.iterrows():
                        exp_date = row["ExpDate"]
                        max_pain = row["Maximum Pain"]
                        ticker_data.setdefault(exp_date, {})[date_dir] = max_pain
                break  # Only process one file per date

    # Convert ticker data to a DataFrame and save it
    df = pd.DataFrame(ticker_data).T  # Transpose for desired format
    # Sort the index and columns in ascending order
    df = df.sort_index(axis=0, ascending=True)
    df = df.sort_index(axis=1, ascending=True)


    # Convert columns to datetime for ensuring accuracy in min and max functions
    converted_dates = [convert_date_format(str(col)) if isinstance(col, str) else col for col in df.columns]
    print(df.columns)
    print(converted_dates)
    df.columns = pd.to_datetime(converted_dates, format='%Y-%m-%d', errors='ignore')
    print(df.dtypes)

    print("DF Columns:", df.columns)
    print("DF Index:", df.index)
    end_date = max(df.columns) + dt.timedelta(days=1)

    stock_data = yf.download(ticker, start=min(df.columns), end=end_date)
    print(stock_data.columns)
    print("Stock Data Index:", stock_data.index)

    # Transpose df and join with stock_data
    combined_frame = df.T.join(stock_data, how='left')

    # Add 'ExpDate' as a column from the index
    combined_frame['ExpDate'] = combined_frame.index

    # Reorder the columns to get OHLC columns first
    cols = ['ExpDate', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] + [col for col in combined_frame.columns
                                                                                 if col not in ['ExpDate','Open', 'High', 'Low',
                                                                                                'Close', 'Adj Close',
                                                                                                'Volume', 'ExpDate']]
    combined_frame = combined_frame[cols]

    # Save the combined DataFrame without an additional index column
    combined_frame.to_csv(f"MP_EXP_{ticker}.csv", index=False)
def process_all_tickers(base_dir="data/ProcessedData"):
    for ticker in os.listdir(base_dir):
        if ticker in ['SPY', 'TSLA']:
            print(ticker)
            generate_and_save_max_pain_for_ticker(ticker, base_dir)

# Run the function
process_all_tickers()
