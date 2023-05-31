import datetime as dt
import os
from pathlib import Path

import pandas as pd

YYMMDD = dt.datetime.today().strftime("%y%m%d")

with open("../Input/tickerlist.txt", "r") as f:
    tickerlist = [line.strip().upper() for line in f.readlines()]

expected_format = "XXX_230427_0930.csv"  # Replace "XXX" with the expected prefix

processed_dir = "..\data\ProcessedData"

###TODO manually change tickero
ticker = "OSTK"
list_of_df = []
ticker_dir = os.path.join(processed_dir, ticker)

for directory in os.listdir(ticker_dir):

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


for line in df:
    df["B1% Change"] = ((df["Bonsai Ratio"] - df["Bonsai Ratio"].shift(1)) / df["Bonsai Ratio"].shift(1)) * 100
    df["B2% Change"] = (
        (df["Bonsai Ratio 2"] - df["Bonsai Ratio 2"].shift(1)) / df["Bonsai Ratio 2"].shift(1)
    ) * 100
    # ###     # add 1 hour later price data to check for corr.
    df["b1/b2"] = df["Bonsai Ratio"] / df["Bonsai Ratio 2"]
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
output_dir = Path(f"corr/{ticker}/")
output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
output_dir2 = Path(f"dailyDF/{ticker}")
output_dir2.mkdir(mode=0o755, parents=True, exist_ok=True)

df.corr().to_csv(f"corr/{ticker}.csv")
df.to_csv(f"dailyDF/{ticker}.csv")
    #
    #
    #
    # #combine bonsai # and itmpcrv,  then bonsai and niv?  hwat else


# for ticker in tickerlist:
#     processed_dir = f"../data/ProcessedData/{ticker}/230519"
#     list_of_df = []
#     for filename in os.listdir(processed_dir):
#
#         if filename.endswith(".csv"):
#             file_path = os.path.join(processed_dir, filename)  # Full path to the file
#
#             dataframe_slice = pd.read_csv(file_path)
#             dataframe_slice = dataframe_slice.iloc[:1]
#             dataframe_slice["time"] = filename[-8:-4]
#
#             # move "time" column to the first position
#             cols = list(dataframe_slice.columns)
#             cols = [cols[-1]] + cols[:-1]
#             dataframe_slice = dataframe_slice[cols]
#
#             # do something with the modified dataframe_slice
#             if len(list_of_df) > 2:
#                 if dataframe_slice.loc[0]["Current Stock Price"] < list_of_df[-1].loc[0]["Current Stock Price"]:
#                     dataframe_slice["Up or down"] = "Down"
#                 elif dataframe_slice.loc[0]["Current Stock Price"] > list_of_df[-1].loc[0]["Current Stock Price"]:
#                     dataframe_slice["Up or down"] = "Up"
#
#             list_of_df.append(dataframe_slice)
#     df = pd.concat(list_of_df, ignore_index=True)
#
#     for line in df:
#         df["B1% Change"] = ((df["Bonsai Ratio"] - df["Bonsai Ratio"].shift(1)) / df["Bonsai Ratio"].shift(1)) * 100
#         df["B2% Change"] = (
#             (df["Bonsai Ratio 2"] - df["Bonsai Ratio 2"].shift(1)) / df["Bonsai Ratio 2"].shift(1)
#         ) * 100
#         # ###     # add 1 hour later price data to check for corr.
#
#         df["6 hour later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-360)
#         df["5 hour later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-300)
#         df["4 hour later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-240)
#         df["3 hour later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-180)
#         df["2 hour later change %"] = df["Current SP % Change" "(LAC)"] - df["Current SP % Change(LAC)"].shift(-120)
#         df["1 hour later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-60)
#         df["45 min later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-45)
#         df["30 min later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-30)
#         df["20 min later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-20)
#         df["15 min later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-15)
#         df["10 min later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-10)
#         df["5 min later change %"] = df["Current SP % Change(LAC)"] - df["Current SP % Change(LAC)"].shift(-5)
#         df["b1/b2"] = df["Bonsai Ratio"] / df["Bonsai Ratio 2"]
#         df["b2/b1"] = df["Bonsai Ratio 2"] / df["Bonsai Ratio"]
#
#     cols.insert(1, cols.pop(cols.index("time")))
#     # df = df.loc[:, cols]
#     output_dir = Path(f"corr/{ticker}/")
#     output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
#     output_dir2 = Path(f"dailyDF/{ticker}")
#     output_dir2.mkdir(mode=0o755, parents=True, exist_ok=True)
#
#     df.corr().to_csv(f"corr/{ticker}/{filename}.csv")
#     df.to_csv(f"dailyDF/{ticker}/{filename}.csv")
#     #
#     #
#     #
#     # #combine bonsai # and itmpcrv,  then bonsai and niv?  hwat else
