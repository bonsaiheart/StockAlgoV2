import datetime as dt
import os
from pathlib import Path

import pandas as pd

YYMMDD = dt.datetime.today().strftime("%y%m%d")

with open("Input/tickerlist.txt", "r") as f:
    tickerlist = [line.strip().upper() for line in f.readlines()]


processed_dir = "data\ProcessedData"
for ticker in tickerlist:
    ticker_dir = os.path.join(processed_dir, ticker)

    list_of_df = []
    for directory in os.listdir(ticker_dir):

        dir_path = os.path.join(ticker_dir, directory)
        for filename in os.listdir(dir_path):
            if filename.endswith(".csv"):
                filepath = os.path.join(dir_path, filename)
                dataframe_slice = pd.read_csv(filepath)
                print(dataframe_slice.columns)

                dataframe_slice = dataframe_slice["Maximum Pain"]
                list_of_df.append(dataframe_slice)
                break
    # print(list_of_df)