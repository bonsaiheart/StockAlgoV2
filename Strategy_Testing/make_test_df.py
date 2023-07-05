import datetime as dt
import os
from pathlib import Path

import pandas as pd

YYMMDD = dt.datetime.today().strftime("%y%m%d")

# with open("../Input/tickerlist.txt", "r") as f:
#     tickerlist = [line.strip().upper() for line in f.readlines()]
def get_1st_frames_make_single_multiday_df(ticker):
    expected_format = "XXX_230427_0930.csv"  # Replace "XXX" with the expected prefix

    processed_dir = "..\data\ProcessedData"

    ###TODO manually change tickero
    ticker = ticker.upper()

    ticker_dir = os.path.join(processed_dir, ticker)
    list_of_df = []
    for directory in os.listdir(ticker_dir):
        print(os.listdir(ticker_dir))
        print(directory)

        dir_path = os.path.join(ticker_dir, directory)
        if directory != "Before TA Or Tradier":
            for filename in os.listdir(dir_path):
                if filename.endswith(".csv"):
                    parts = filename.split("_")
                    # print(parts)
                    if len(parts) == 3  and len(parts[1]) == 6 and len(
                            parts[2]) == 8:


                        filepath = os.path.join(dir_path, filename)
                        dataframe_slice = pd.read_csv(filepath)
                        dataframe_slice = dataframe_slice.iloc[:1]
                        dataframe_slice["date"] = filename[-15:-9]
                        dataframe_slice["time"] = filename[-8:-4]
                        # if len(list_of_df) > 2:
                        #     try:
                        #         if dataframe_slice.loc[0]["Current Stock Price"] < list_of_df[-1].loc[0]["Current Stock Price"]:
                        #             dataframe_slice["Up or down"] = "0"
                        #         elif dataframe_slice.loc[0]["Current Stock Price"] > list_of_df[-1].loc[0][
                        #             "Current Stock Price"]:
                        #             dataframe_slice["Up or down"] = "1"

                            # except KeyError:
                            #     pass

                        list_of_df.append(dataframe_slice)
                        # move "time" column to the first position
                        cols = list(dataframe_slice.columns)
                        cols = [cols[-2]] + cols[:-2]
                        dataframe_slice = dataframe_slice[cols]
                    # do something with the modified dataframe_slice
                    else:
                        print(f"{filename}: Format is incorrect.")
            else:print("dir is Before TA Or Tradier")



    df = pd.concat(list_of_df, ignore_index=True)

    print(df)
    output_dir = Path(f"historical_minute_corr/{ticker}/")
    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    output_dir2 = Path(f"historical_minute_DF/{ticker}")
    output_dir2.mkdir(mode=0o755, parents=True, exist_ok=True)
    try:
        df.drop("Closest Strike Above/Below(below to above,4 each) list",axis=1,inplace=True)
    except Exception as e:
        pass
    df.corr().to_csv(f"historical_minute_corr/{ticker}.csv")
    df.to_csv(f"historical_minute_DF/{ticker}.csv")
    return (ticker, df)
def get_1st_frames_to_make_daily_df(ticker):
    expected_format = "XXX_230427_0930.csv"  # Replace "XXX" with the expected prefix

    processed_dir = "..\data\ProcessedData"

    ###TODO manually change tickero
    ticker = ticker.upper()

    ticker_dir = os.path.join(processed_dir, ticker)

    for directory in os.listdir(ticker_dir):
        print(os.listdir(ticker_dir))
        print(directory)
        list_of_df = []
        dir_path = os.path.join(ticker_dir, directory)
        if directory != "Before TA Or Tradier":
            for filename in os.listdir(dir_path):
                if filename.endswith(".csv"):
                    parts = filename.split("_")
                    # print(parts)
                    if len(parts) == 3  and len(parts[1]) == 6 and len(
                            parts[2]) == 8:


                        filepath = os.path.join(dir_path, filename)
                        dataframe_slice = pd.read_csv(filepath)
                        dataframe_slice = dataframe_slice.iloc[:1]
                        dataframe_slice["date"] = filename[-15:-9]
                        dataframe_slice["time"] = filename[-8:-4]
                        if len(list_of_df) > 2:
                            try:
                                if dataframe_slice.loc[0]["Current Stock Price"] < list_of_df[-1].loc[0]["Current Stock Price"]:
                                    dataframe_slice["Up or down"] = "0"
                                elif dataframe_slice.loc[0]["Current Stock Price"] > list_of_df[-1].loc[0][
                                    "Current Stock Price"]:
                                    dataframe_slice["Up or down"] = "1"

                            except KeyError:
                                pass

                        list_of_df.append(dataframe_slice)
                        # move "time" column to the first position
                        cols = list(dataframe_slice.columns)
                        cols = [cols[-2]] + cols[:-2]
                        dataframe_slice = dataframe_slice[cols]
                    # do something with the modified dataframe_slice
                    else:
                        print(f"{filename}: Format is incorrect.")
            else:print("dir is Before TA Or Tradier")



        df = pd.concat(list_of_df, ignore_index=True)

        print(df)
        output_dir = Path(f"corr/{ticker}/")
        output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
        output_dir2 = Path(f"dailyDF/{ticker}")
        output_dir2.mkdir(mode=0o755, parents=True, exist_ok=True)

        df.corr().to_csv(f"corr/{ticker}/{directory}.csv")
        df.to_csv(f"dailyDF/{ticker}/{directory}.csv")
        return (ticker, df)
def multiday_minute_series_prep_for_backtest(ticker,df):
    for line in df:
        df["B1% Change"] = ((df["Bonsai Ratio"] - df["Bonsai Ratio"].shift(1)) / df["Bonsai Ratio"].shift(1)) * 100
        df["B2% Change"] = (
            (df["Bonsai Ratio 2"] - df["Bonsai Ratio 2"].shift(1)) / df["Bonsai Ratio 2"].shift(1)
        ) * 100
        # ###     # add 1 hour later price data to check for corr.
        df["B1/B2"] = df["Bonsai Ratio"] / df["Bonsai Ratio 2"]
        df["6 hour later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-360)
        df["5 hour later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-300)
        df["4 hour later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-240)
        df["3 hour later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-180)
        df["2 hour later change %"] = df["Current SP % Change" "(LAC)"] - df["Current SP % Change(LAC)"].shift(-120)
        df["1 hour later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-60)
        df["45 min later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-45)
        df["30 min later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-30)
        df["20 min later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-20)
        df["15 min later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-15)
        df["10 min later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-10)
        df["5 min later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-5)

    cols = list(df.columns)
    cols.insert(1, cols.pop(cols.index("time")))
    cols.insert(1, cols.pop(cols.index("date")))
    df = df[cols]
    # df = df.loc[:, cols]
    print(df.columns)
    output_dir = Path(f"historical_multiday_minute_corr/{ticker}/")
    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    output_dir2 = Path(f"historical_multiday_minute_DF/{ticker}")
    output_dir2.mkdir(mode=0o755, parents=True, exist_ok=True)

    df.corr().to_csv(f"historical_multiday_minute_corr/{YYMMDD}_{ticker}.csv")
    df.to_csv(f"historical_multiday_minute_DF/{ticker}/{YYMMDD}_{ticker}.csv")
def minute_series_prep_for_backtest(ticker,df):
    for line in df:
        df["B1% Change"] = ((df["Bonsai Ratio"] - df["Bonsai Ratio"].shift(1)) / df["Bonsai Ratio"].shift(1)) * 100
        df["B2% Change"] = (
            (df["Bonsai Ratio 2"] - df["Bonsai Ratio 2"].shift(1)) / df["Bonsai Ratio 2"].shift(1)
        ) * 100
        # ###     # add 1 hour later price data to check for corr.
        df["B1/B2"] = df["Bonsai Ratio"] / df["Bonsai Ratio 2"]
        df["6 hour later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-360)
        df["5 hour later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-300)
        df["4 hour later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-240)
        df["3 hour later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-180)
        df["2 hour later change %"] = df["Current SP % Change" "(LAC)"] - df["Current SP % Change(LAC)"].shift(-120)
        df["1 hour later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-60)
        df["45 min later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-45)
        df["30 min later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-30)
        df["20 min later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-20)
        df["15 min later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-15)
        df["10 min later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-10)
        df["5 min later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-5)

    cols = list(df.columns)
    cols.insert(1, cols.pop(cols.index("time")))
    cols.insert(1, cols.pop(cols.index("date")))
    df = df[cols]
    # df = df.loc[:, cols]
    print(df.columns)
    output_dir = Path(f"historical_minute_corr/{ticker}/")
    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    output_dir2 = Path(f"historical_minute_DF/{ticker}")
    output_dir2.mkdir(mode=0o755, parents=True, exist_ok=True)

    df.corr().to_csv(f"historical_minute_corr/{ticker}.csv")
    df.to_csv(f"historical_minute_DF/{ticker}/{YYMMDD}_{ticker}.csv")
def daily_series_prep_for_backtest(ticker,df):
    df["Current Price"] = df["Current Stock Price"]
    df["date"] = df["Date"]
    for line in df:

        df["B1% Change"] = ((df["Bonsai Ratio"] - df["Bonsai Ratio"].shift(1)) / df["Bonsai Ratio"].shift(1)) * 100
        df["B2% Change"] = (
            (df["Bonsai Ratio 2"] - df["Bonsai Ratio 2"].shift(1)) / df["Bonsai Ratio 2"].shift(1)
        ) * 100
        # ###     # add 1 hour later price data to check for corr.
        df["B1/B2"] = df["Bonsai Ratio"] / df["Bonsai Ratio 2"]
        df["B2/B1"] =df["Bonsai Ratio 2"]/df["Bonsai Ratio"]
        df["12 day later change %"] = ((df["Current Price"].shift(12) - df["Current Price"]) / df['Current Price']) *100
        df["11 day later change %"] = ((df["Current Price"].shift(11) - df["Current Price"]) / df['Current Price']) *100
        df["10 day later change %"] =  ((df["Current Price"].shift(10) - df["Current Price"]) / df['Current Price']) *100
        df["9 day later change %"]  =  ((df["Current Price"].shift(9) - df["Current Price"]) / df['Current Price']) *100
        df["8 day later change %"]  =  ((df["Current Price"].shift(8) - df["Current Price"]) / df['Current Price']) *100
        df["7 day later change %"]  =  ((df["Current Price"].shift(7) - df["Current Price"]) / df['Current Price']) *100
        df["6 day later change %"]  =  ((df["Current Price"].shift(6) - df["Current Price"]) / df['Current Price']) *100
        df["5 day later change %"]  =  ((df["Current Price"].shift(5) - df["Current Price"]) / df['Current Price']) *100
        df["4 day later change %"] =   ((df["Current Price"].shift(4) - df["Current Price"]) / df['Current Price']) *100
        df["3 day later change %"] =  ((df["Current Price"].shift(3) - df["Current Price"]) / df['Current Price']) *100
        df["2 day later change %"] =  ((df["Current Price"].shift(2) - df["Current Price"]) / df['Current Price']) *100
        df["1 day later change %"] =  ((df["Current Price"].shift(1) - df["Current Price"]) / df['Current Price']) *100

    cols = list(df.columns)
    # cols.insert(1, cols.pop(cols.index("time")))
    cols.insert(1, cols.pop(cols.index("date")))
    df = df[cols]
    # df = df.loc[:, cols]

    print(df.columns)
    output_dir = Path(f"historical_daily_corr/{ticker}/")
    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    output_dir2 = Path(f"historical_daily_DF/{ticker}")
    output_dir2.mkdir(mode=0o755, parents=True, exist_ok=True)


    df.corr().to_csv(f"historical_daily_corr/historical_daily_{ticker}.csv")
    df.to_csv(f"historical_daily_DF/historical_daily_{ticker}.csv")
    #
    #
    #
    # #combine bonsai # and itmpcrv,  then bonsai and niv?  hwat else
# df = pd.read_csv(r"C:\Users\natha\PycharmProjects\StockAlgoV2\Historical_Data_Scraper\data\Historical_Processed_ChainData\SPY.csv")
# daily_series_prep_for_backtest("SPY",df)
tickers=['spy']
for x in tickers:
    ticker,df = get_1st_frames_make_single_multiday_df(x)
    print(ticker)
    multiday_minute_series_prep_for_backtest(ticker,df)