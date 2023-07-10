import datetime as dt
import os
from pathlib import Path

import pandas as pd

YYMMDD = dt.datetime.today().strftime("%y%m%d")

with open("UTILITIES/tickerlist.txt", "r") as f:
    tickerlist = [line.strip().upper() for line in f.readlines()]


processed_dir = "data\ProcessedData"



for ticker in tickerlist:
    ticker_dir = os.path.join(processed_dir, ticker)
    list_of_df = []
    exp_date_list = []
    for directory in os.listdir(ticker_dir):
        dir_path = os.path.join(ticker_dir, directory)

        for filename in os.listdir(dir_path):
            if filename.endswith(".csv"):
                filepath = os.path.join(dir_path, filename)
                dataframe_slice = pd.read_csv(filepath)

                # Extract date from the filename
                date = filename.split("_")[1]
                column_name = f"MP {date}"

                if 'Maximum Pain' in dataframe_slice.columns:
                    dataframe_slice[column_name] = dataframe_slice['Maximum Pain']
                    list_of_df.append(dataframe_slice[[column_name]])

                    # Check if 'ExpDate' column exists and add its values to the list of ExpDates
                    if 'ExpDate' in dataframe_slice.columns:
                        exp_date_list.extend(dataframe_slice['ExpDate'])

                    break  # Stop processing the current file

    # Create a DataFrame with the combined ExpDate values
    exp_date_df = pd.DataFrame({'ExpDate': exp_date_list})

    # Concatenate the DataFrames and include the 'ExpDate' column on the far left
    combined_frame = pd.concat([exp_date_df.reset_index(drop=True)] + [df.reset_index(drop=True) for df in list_of_df],
                               axis=1)
    print(combined_frame)
    combined_frame.to_csv("MP_EXP.csv", index=False)
