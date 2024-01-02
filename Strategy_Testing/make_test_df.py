import datetime as dt
from pathlib import Path
import pandas as pd

# Define the base directory relative to the current script
base_dir = Path(__file__).resolve().parent.parent
YYMMDD = dt.datetime.today().strftime("%y%m%d")


def get_1st_frames_to_make_dailyminute_df(ticker):
    ticker = ticker.upper()
    processed_dir = base_dir / "data" / "ProcessedData" / ticker

    for directory in processed_dir.iterdir():
        if directory.name != "Before TA Or Tradier" and directory.is_dir():
            list_of_df = []
            for filepath in directory.glob("*.csv"):
                dataframe_slice = pd.read_csv(filepath)
                dataframe_slice = dataframe_slice.iloc[:1]
                list_of_df.append(dataframe_slice)
            else:
                print("dir is Before TA Or Tradier")

            df = pd.concat(list_of_df, ignore_index=True)
            print(df)
            output_dir2 = base_dir / "dailyminutes_from_processed" / ticker
            output_dir2.mkdir(mode=0o755, parents=True, exist_ok=True)
            df.to_csv(output_dir2 / f"{ticker}_{directory.name}.csv", index=False)


def get_dailyminutes_make_single_multiday_df(ticker):
    ticker = ticker.upper()
    dailyminutes_dir = base_dir / "data" / "DailyMinutes" / ticker
    list_of_df = []

    for filename in sorted(dailyminutes_dir.glob("*.csv")):
        print(filename)
        dataframe_slice = pd.read_csv(filename)
        list_of_df.append(dataframe_slice)

    print(list_of_df[-1])
    df = pd.concat(list_of_df, ignore_index=True)

    try:
        df.drop(
            "Closest Strike Above/Below(below to above,4 each) list",
            axis=1,
            inplace=True,
        )
    except Exception as e:
        print(e)

    output_dir = base_dir / "data" / "historical_multiday_minute_DF"
    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    df.to_csv(output_dir / f"{ticker}_historical_multiday_min.csv", index=False)


def corr_minute_df(ticker, df):
    df["B1% Change"] = (
        (df["Bonsai Ratio"] - df["Bonsai Ratio"].shift(1)) / df["Bonsai Ratio"].shift(1)
    ) * 100
    df["B2% Change"] = (
        (df["Bonsai Ratio 2"] - df["Bonsai Ratio 2"].shift(1))
        / df["Bonsai Ratio 2"].shift(1)
    ) * 100
    # ###     # add later price data to check for corr.
    df["6 hour later change %"] = (
        df["Current Stock Price"].shift(-360) - df["Current Stock Price"]
    ) / df["Current Stock Price"]
    df["5 hour later change %"] = (
        df["Current Stock Price"].shift(-300) - df["Current Stock Price"]
    ) / df["Current Stock Price"]
    df["4 hour later change %"] = (
        df["Current Stock Price"].shift(-240) - df["Current Stock Price"]
    ) / df["Current Stock Price"]
    df["3 hour later change %"] = (
        df["Current Stock Price"].shift(-180) - df["Current Stock Price"]
    ) / df["Current Stock Price"]
    df["2 hour later change %"] = (
        df["Current Stock Price"].shift(-120) - df["Current Stock Price"]
    ) / df["Current Stock Price"]
    df["1 hour later change %"] = (
        df["Current Stock Price"].shift(-60) - df["Current Stock Price"]
    ) / df["Current Stock Price"]
    df["45 min later change %"] = (
        df["Current Stock Price"].shift(-45) - df["Current Stock Price"]
    ) / df["Current Stock Price"]
    df["30 min later change %"] = (
        df["Current Stock Price"].shift(-30) - df["Current Stock Price"]
    ) / df["Current Stock Price"]
    df["20 min later change %"] = (
        df["Current Stock Price"].shift(-20) - df["Current Stock Price"]
    ) / df["Current Stock Price"]
    df["15 min later change %"] = (
        df["Current Stock Price"].shift(-15) - df["Current Stock Price"]
    ) / df["Current Stock Price"]
    df["10 min later change %"] = (
        df["Current Stock Price"].shift(-10) - df["Current Stock Price"]
    ) / df["Current Stock Price"]
    df["5 min later change %"] = (
        df["Current Stock Price"].shift(-5) - df["Current Stock Price"]
    ) / df["Current Stock Price"]
    new_data = {}

    window_sizes_minutes = [15, 30, 45]  # 15, 30, and 45 minutes
    for minutes in window_sizes_minutes:
        window = minutes  # window size in minutes
        shifted = df["Current Stock Price"].shift(-window)
        # Calculate max change and store in new_data dictionary for minute intervals
        new_data[f"{minutes} minute later max change %"] = (
            (df["Current Stock Price"].rolling(window).max() - shifted) / shifted * 100
        )

    for hours in range(1, 8):  # Loop from 1 to 7 hours
        window = hours * 60  # Convert hours to minutes
        shifted = df["Current Stock Price"].shift(-window)
        # Calculate max change and store in new_data dictionary for hour intervals
        new_data[f"{hours} hour later max change %"] = (
            (df["Current Stock Price"].rolling(window).max() - shifted) / shifted * 100
        )

    # Create a new DataFrame from the new_data dictionary
    new_df = pd.DataFrame(new_data, index=df.index)

    # Concatenate this new DataFrame with the original DataFrame
    df = pd.concat([df, new_df], axis=1)

    output_dir = base_dir / "data" / "historical_multiday_minute_corr"
    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    df.corr().to_csv(output_dir / f"{YYMMDD}_{ticker}.csv")


def multiday_minute_series_prep_for_backtest(ticker, df):
    df["B1% Change"] = (
        (df["Bonsai Ratio"] - df["Bonsai Ratio"].shift(1)) / df["Bonsai Ratio"].shift(1)
    ) * 100
    df["B2% Change"] = (
        (df["Bonsai Ratio 2"] - df["Bonsai Ratio 2"].shift(1))
        / df["Bonsai Ratio 2"].shift(1)
    ) * 100
    df["B1/B2"] = df["Bonsai Ratio"] / df["Bonsai Ratio 2"]

    cols = list(df.columns)
    if "time" in df.columns:
        cols.insert(1, cols.pop(cols.index("time")))
    if "date" in df.columns:
        cols.insert(1, cols.pop(cols.index("date")))
    df = df[cols]

    output_dir = base_dir / "historical_multiday_minute_corr" / ticker
    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    output_dir2 = base_dir / "historical_multiday_minute_DF" / ticker
    output_dir2.mkdir(mode=0o755, parents=True, exist_ok=True)

    df.corr().to_csv(output_dir / f"{YYMMDD}_{ticker}.csv")
    df.to_csv(output_dir2 / f"{YYMMDD}_{ticker}.csv")


def daily_series_prep_for_backtest(ticker, df):
    df["Current Price"] = df["Current Stock Price"]
    df["date"] = df["Date"]

    df["B1% Change"] = (
        (df["Bonsai Ratio"] - df["Bonsai Ratio"].shift(1)) / df["Bonsai Ratio"].shift(1)
    ) * 100
    df["B2% Change"] = (
        (df["Bonsai Ratio 2"] - df["Bonsai Ratio 2"].shift(1))
        / df["Bonsai Ratio 2"].shift(1)
    ) * 100
    # ###     # add 1 hour later price data to check for corr.
    df["B1/B2"] = df["Bonsai Ratio"] / df["Bonsai Ratio 2"]
    df["B2/B1"] = df["Bonsai Ratio 2"] / df["Bonsai Ratio"]
    df["12 day later change %"] = (
        (df["Current Price"].shift(12) - df["Current Price"]) / df["Current Price"]
    ) * 100
    df["11 day later change %"] = (
        (df["Current Price"].shift(11) - df["Current Price"]) / df["Current Price"]
    ) * 100
    df["10 day later change %"] = (
        (df["Current Price"].shift(10) - df["Current Price"]) / df["Current Price"]
    ) * 100
    df["9 day later change %"] = (
        (df["Current Price"].shift(9) - df["Current Price"]) / df["Current Price"]
    ) * 100
    df["8 day later change %"] = (
        (df["Current Price"].shift(8) - df["Current Price"]) / df["Current Price"]
    ) * 100
    df["7 day later change %"] = (
        (df["Current Price"].shift(7) - df["Current Price"]) / df["Current Price"]
    ) * 100
    df["6 day later change %"] = (
        (df["Current Price"].shift(6) - df["Current Price"]) / df["Current Price"]
    ) * 100
    df["5 day later change %"] = (
        (df["Current Price"].shift(5) - df["Current Price"]) / df["Current Price"]
    ) * 100
    df["4 day later change %"] = (
        (df["Current Price"].shift(4) - df["Current Price"]) / df["Current Price"]
    ) * 100
    df["3 day later change %"] = (
        (df["Current Price"].shift(3) - df["Current Price"]) / df["Current Price"]
    ) * 100
    df["2 day later change %"] = (
        (df["Current Price"].shift(2) - df["Current Price"]) / df["Current Price"]
    ) * 100
    df["1 day later change %"] = (
        (df["Current Price"].shift(1) - df["Current Price"]) / df["Current Price"]
    ) * 100

    cols = list(df.columns)
    # cols.insert(1, cols.pop(cols.index("time")))
    cols.insert(1, cols.pop(cols.index("date")))
    df = df[cols]
    # df = df.loc[:, cols]

    output_dir = base_dir / "historical_daily_corr" / ticker
    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    output_dir2 = base_dir / "historical_daily_DF" / ticker
    output_dir2.mkdir(mode=0o755, parents=True, exist_ok=True)

    df.corr().to_csv(output_dir / f"historical_daily_{ticker}.csv")
    df.to_csv(output_dir2 / f"historical_daily_{ticker}.csv")


if __name__ == "__main__":
    tickerlist_file = base_dir / "UTILITIES" / "tickerlist.txt"
    with open(tickerlist_file, "r") as f:
        tickerlist = [line.strip().upper() for line in f.readlines()]

    for ticker in tickerlist:
        try:
            get_dailyminutes_make_single_multiday_df(ticker)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
