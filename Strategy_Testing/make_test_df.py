import datetime as dt
import os
from pathlib import Path

import pandas as pd

YYMMDD = dt.datetime.today().strftime("%y%m%d")
#TODO add function to make dailyminutes from optionsdata
import datetime as dt
import os
from pathlib import Path

import pandas as pd

YYMMDD = dt.datetime.today().strftime("%y%m%d")

# with open("../Input/tickerlist.txt", "r") as f:
#     tickerlist = [line.strip().upper() for line in f.readlines()]
# def get_1st_frames_make_single_multiday_df(ticker):
#     expected_format = "XXX_230427_0930.csv"  # Replace "XXX" with the expected prefix
#
#     processed_dir = "..\data\ProcessedData"
#
#     ###TODO manually change tickero
#     ticker = ticker.upper()
#
#     ticker_dir = os.path.join(processed_dir, ticker)
#     list_of_df = []
#     for directory in os.listdir(ticker_dir):
#         print(os.listdir(ticker_dir))
#         print(directory)
#
#         dir_path = os.path.join(ticker_dir, directory)
#         if directory != "Before TA Or Tradier":
#             for filename in os.listdir(dir_path):
#                 if filename.endswith(".csv"):
#                     parts = filename.split("_")
#                     # print(parts)
#                     if len(parts) == 3  and len(parts[1]) == 6 and len(
#                             parts[2]) == 8:
#
#
#                         filepath = os.path.join(dir_path, filename)
#                         dataframe_slice = pd.read_csv(filepath)
#                         dataframe_slice = dataframe_slice.iloc[:1]
#                         dataframe_slice["date"] = filename[-15:-9]
#                         dataframe_slice["time"] = filename[-8:-4]
#                         if len(list_of_df) > 2:
#                             try:
#                                 if dataframe_slice.loc[0]["Current Stock Price"] < list_of_df[-1].loc[0]["Current Stock Price"]:
#                                     dataframe_slice["Up or down"] = "0"
#                                 elif dataframe_slice.loc[0]["Current Stock Price"] > list_of_df[-1].loc[0][
#                                     "Current Stock Price"]:
#                                     dataframe_slice["Up or down"] = "1"
#
#                             except KeyError:
#                                 pass
#
#                         list_of_df.append(dataframe_slice)
#                         # move "time" column to the first position
#                         cols = list(dataframe_slice.columns)
#                         cols = [cols[-2]] + cols[:-2]
#                         dataframe_slice = dataframe_slice[cols]
#                     # do something with the modified dataframe_slice
#                     else:
#                         print(f"{filename}: Format is incorrect.")
#             else:print("dir is Before TA Or Tradier")
#
#
#
#     df = pd.concat(list_of_df, ignore_index=True)
#
#     print(df)
#     output_dir = Path(f"corr/{ticker}/")
#     output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
#     output_dir2 = Path(f"dailyDF/{ticker}")
#     output_dir2.mkdir(mode=0o755, parents=True, exist_ok=True)
#     df.drop("Closest Strike Above/Below(below to above,4 each) list",axis=1,inplace=True)
#     df.corr().to_csv(f"corr/{ticker}.csv")
#     df.to_csv(f"dailyDF/{ticker}.csv")
#     return (ticker, df)
def get_1st_frames_to_make_dailyminute_df(ticker):
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


                    filepath = os.path.join(dir_path, filename)
                    dataframe_slice = pd.read_csv(filepath)
                    dataframe_slice = dataframe_slice.iloc[:1]
                    # filename_parts = filename.split('_')  # Split the filename by '_'
                    # date = filename_parts[1]  # The date is the second element
                    # time = filename_parts[2][:4]  # The time is the first 4 characters of the third element
                    #
                    # dataframe_slice["date"] = date
                    # dataframe_slice["time"] = time

                    list_of_df.append(dataframe_slice)
                        # move "time" column to the first position
                        # cols = list(dataframe_slice.columns)
                        # cols = [cols[-2]] + cols[:-2]
                        # dataframe_slice = dataframe_slice[cols]
                    # do something with the modified dataframe_slice
                else:
                    print(f"{filename}: Format is incorrect.")
            else:print("dir is Before TA Or Tradier")



        df = pd.concat(list_of_df, ignore_index=True)

        print(df)
        # output_dir = Path(f"corr/{ticker}/")
        # output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
        output_dir2 = Path(f"dailyDF/{ticker}")
        output_dir2.mkdir(mode=0o755, parents=True, exist_ok=True)

        # df.corr().to_csv(f"corr/{ticker}/{directory}.csv")
        df.to_csv(f"dailyminutes_from_processed/{ticker}/{ticker}_{directory}.csv", index=False)

def get_dailyminutes_make_single_multiday_df(ticker):
    # expected_format = "XXX_230427_0930.csv"
    dailyminutes_dir = "../data/DailyMinutes"
    ticker = ticker.upper()
    ticker_dir = os.path.join(dailyminutes_dir, ticker)
    list_of_df = []
    for filename in sorted(os.listdir(ticker_dir)):
        if filename.endswith(".csv"):
            print(filename)
            filepath = os.path.join(ticker_dir, filename)
            dataframe_slice = pd.read_csv(filepath)
            list_of_df.append(dataframe_slice)
            # move "time" column to the first position
        else:
            print(f"{filename}: Format is incorrect.")
    print(list_of_df[-1])
    df = pd.concat(list_of_df, ignore_index=True)
    df.drop_duplicates(subset='LastTradeTime', inplace=True)
    output_dir = Path(rf"../data/historical_multiday_minute_DF/")
    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    try:
        df.drop("Closest Strike Above/Below(below to above,4 each) list",axis=1,inplace=True)
    except Exception as e:
        pass
    df.to_csv(rf"../data/historical_multiday_minute_DF/{ticker}_historical_multiday_min.csv")
    return df

def multiday_minute_series_prep_for_backtest(ticker,df):

    df["B1% Change"] = ((df["Bonsai Ratio"] - df["Bonsai Ratio"].shift(1)) / df["Bonsai Ratio"].shift(1)) * 100
    df["B2% Change"] = (
        (df["Bonsai Ratio 2"] - df["Bonsai Ratio 2"].shift(1)) / df["Bonsai Ratio 2"].shift(1)
    ) * 100
    df["B1/B2"] = df["Bonsai Ratio"] / df["Bonsai Ratio 2"]

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

def daily_series_prep_for_backtest(ticker,df):
    df["Current Price"] = df["Current Stock Price"]
    df["date"] = df["Date"]


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

    # print(df.columns)
    output_dir = Path(f"historical_daily_corr/{ticker}/")
    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    output_dir2 = Path(f"historical_daily_DF/{ticker}")
    output_dir2.mkdir(mode=0o755, parents=True, exist_ok=True)

    df.corr().to_csv(f"historical_daily_corr/historical_daily_{ticker}.csv")
    df.to_csv(f"historical_daily_DF/historical_daily_{ticker}.csv")

#TODO corr_daily_df
def corr_minute_df(ticker,df):

    df["B1% Change"] = ((df["Bonsai Ratio"] - df["Bonsai Ratio"].shift(1)) / df["Bonsai Ratio"].shift(1)) * 100
    df["B2% Change"] = (
        (df["Bonsai Ratio 2"] - df["Bonsai Ratio 2"].shift(1)) / df["Bonsai Ratio 2"].shift(1)
    ) * 100
    # ###     # add later price data to check for corr.
    df["6 hour later change %"] =  (df["Current Stock Price"].shift(-360)-df["Current Stock Price"])/df["Current Stock Price"]
    df["5 hour later change %"] =  (df["Current Stock Price"].shift(-300)-df["Current Stock Price"])/df["Current Stock Price"]
    df["4 hour later change %"] =  (df["Current Stock Price"].shift(-240)-df["Current Stock Price"])/df["Current Stock Price"]
    df["3 hour later change %"] =  (df["Current Stock Price"].shift(-180)-df["Current Stock Price"])/df["Current Stock Price"]
    df["2 hour later change %"] =  (df["Current Stock Price"].shift(-120)-df["Current Stock Price"])/df["Current Stock Price"]
    df["1 hour later change %"] =  (df["Current Stock Price"].shift(-60)-df["Current Stock Price"])/df["Current Stock Price"]
    df["45 min later change %"] =  (df["Current Stock Price"].shift(-45)-df["Current Stock Price"])/df["Current Stock Price"]
    df["30 min later change %"] =  (df["Current Stock Price"].shift(-30)-df["Current Stock Price"])/df["Current Stock Price"]
    df["20 min later change %"] =  (df["Current Stock Price"].shift(-20)-df["Current Stock Price"])/df["Current Stock Price"]
    df["15 min later change %"] =  (df["Current Stock Price"].shift(-15)-df["Current Stock Price"])/df["Current Stock Price"]
    df["10 min later change %"] =  (df["Current Stock Price"].shift(-10)-df["Current Stock Price"])/df["Current Stock Price"]
    df["5 min later change %"] =   (df["Current Stock Price"].shift(-5)-df["Current Stock Price"])/df["Current Stock Price"]
    new_data = {}

    window_sizes_minutes = [15, 30, 45]  # 15, 30, and 45 minutes
    for minutes in window_sizes_minutes:
        window = minutes  # window size in minutes
        shifted = df["Current Stock Price"].shift(-window)
        # Calculate max change and store in new_data dictionary for minute intervals
        new_data[f"{minutes} minute later max change %"] = (df["Current Stock Price"].rolling(
            window).max() - shifted) / shifted * 100

    for hours in range(1, 8):  # Loop from 1 to 7 hours
        window = hours * 60  # Convert hours to minutes
        shifted = df["Current Stock Price"].shift(-window)
        # Calculate max change and store in new_data dictionary for hour intervals
        new_data[f"{hours} hour later max change %"] = (df["Current Stock Price"].rolling(
            window).max() - shifted) / shifted * 100

    # Create a new DataFrame from the new_data dictionary
    new_df = pd.DataFrame(new_data, index=df.index)

    # Concatenate this new DataFrame with the original DataFrame
    df = pd.concat([df, new_df], axis=1)

    output_dir = Path("../data/historical_multiday_minute_corr/")
    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    df.corr().to_csv(f"../data/historical_multiday_minute_corr/{YYMMDD}_{ticker}.csv")
tickers=['spy',
# 'uvxy',
# 'tsla',
# 'roku',
# 'chwy',
# 'ba',
# 'cmps',
# 'mnmd',
# 'goev',
# 'w',
# 'msft',
'goog']
for x in tickers:
    get_1st_frames_to_make_dailyminute_df(x)
    # print(x)
    # df=get_dailyminutes_make_single_multiday_df(x)
    # # print("corr")
    # corr_minute_df(x,df)

