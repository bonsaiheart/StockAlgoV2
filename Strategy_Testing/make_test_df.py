import datetime as dt
import os
from pathlib import Path

import pandas as pd

YYMMDD = dt.datetime.today().strftime("%y%m%d")

# with open("../Input/tickerlist.txt", "r") as f:
#     tickerlist = [line.strip().upper() for line in f.readlines()]
def get_dailyminutes_make_single_multiday_df(ticker):
    expected_format = "XXX_230427_0930.csv"  # Replace "XXX" with the expected prefix

    dailyminutes_dir = "../data/DailyMinutes"

    ###TODO manually change tickero
    ticker = ticker.upper()


    ticker_dir = os.path.join(dailyminutes_dir, ticker)
    list_of_df = []
    for filename in sorted(os.listdir(ticker_dir)):


        if filename.endswith(".csv"):

            # print(parts)
            print(filename)
            filepath = os.path.join(ticker_dir, filename)
            dataframe_slice = pd.read_csv(filepath)


            list_of_df.append(dataframe_slice)
            # move "time" column to the first position
        else:
            print(f"{filename}: Format is incorrect.")
    else:print("dir is Before TA Or Tradier")



    df = pd.concat(list_of_df, ignore_index=True)
    df.drop_duplicates(subset='LastTradeTime', inplace=True)

    output_dir = Path(f"historical_mulitday_minute_corr/")
    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    output_dir2 = Path(rf"../data/historical_multiday_minute_DF/")
    output_dir2.mkdir(mode=0o755, parents=True, exist_ok=True)
    try:
        df.drop("Closest Strike Above/Below(below to above,4 each) list",axis=1,inplace=True)
    except Exception as e:
        pass
    df.to_csv(rf"../data/historical_multiday_minute_DF/{ticker}_historical_multiday_min.csv")
    return df

def multiday_minute_series_prep_for_backtest(ticker,df):
    for line in df:
        df["B1% Change"] = ((df["Bonsai Ratio"] - df["Bonsai Ratio"].shift(1)) / df["Bonsai Ratio"].shift(1)) * 100
        df["B2% Change"] = (
            (df["Bonsai Ratio 2"] - df["Bonsai Ratio 2"].shift(1)) / df["Bonsai Ratio 2"].shift(1)
        ) * 100
        # ###     # add 1 hour later price data to check for corr.
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
def multiday_minutes_corr(ticker,df):

    df["B1% Change"] = ((df["Bonsai Ratio"] - df["Bonsai Ratio"].shift(1)) / df["Bonsai Ratio"].shift(1)) * 100
    df["B2% Change"] = (
        (df["Bonsai Ratio 2"] - df["Bonsai Ratio 2"].shift(1)) / df["Bonsai Ratio 2"].shift(1)
    ) * 100
    # ###     # add later price data to check for corr.
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

    output_dir = Path("../data/historical_multiday_minute_corr/")
    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

    df.corr().to_csv(f"../data/historical_multiday_minute_corr/{YYMMDD}_{ticker}.csv")

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

    print(df.columns)
    output_dir = Path(f"historical_daily_corr/{ticker}/")
    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    output_dir2 = Path(f"historical_daily_DF/{ticker}")
    output_dir2.mkdir(mode=0o755, parents=True, exist_ok=True)

    df.corr().to_csv(f"historical_daily_corr/historical_daily_{ticker}.csv")
    df.to_csv(f"historical_daily_DF/historical_daily_{ticker}.csv")
    # #combine bonsai # and itmpcrv,  then bonsai and niv?  hwat else
# df = pd.read_csv(r"C:\Users\natha\PycharmProjects\StockAlgoV2\Historical_Data_Scraper\data\Historical_Processed_ChainData\SPY.csv")
# daily_series_prep_for_backtest("SPY",df)


# tickers=['spy']
# for x in tickers:
#     print(x)
#     df=get_dailyminutes_make_single_multiday_df(x)
#     print("corr")
#     multiday_minutes_corr(x,df)
def corr_a_df(df):

    df["B1% Change"] = ((df["Bonsai Ratio"] - df["Bonsai Ratio"].shift(1)) / df["Bonsai Ratio"].shift(1)) * 100
    df["B2% Change"] = (
        (df["Bonsai Ratio 2"] - df["Bonsai Ratio 2"].shift(1)) / df["Bonsai Ratio 2"].shift(1)
    ) * 100
    # ###     # add later price data to check for corr.
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

    output_dir = Path("../data/historical_multiday_minute_corr/")
    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    df.corr().to_csv(f"newcorr_checkitsista2.csv")
tickers=['spy']
for x in tickers:
    print(x)
    df=get_dailyminutes_make_single_multiday_df(x)
    print("corr")
    multiday_minutes_corr(x,df)


#
# df = pd.read_csv(
#     r'C:\Users\del_p\PycharmProjects\StockAlgoV2\Strategy_Testing\algooutput_NEW ALL COLUMNS2_SPY_historical_multiday_min.csv')
# corr_a_df(df)