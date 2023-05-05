from pathlib import Path

import os
import pandas as pd
import datetime as dt

YYMMDD = dt.datetime.today().strftime('%y%m%d')

with open("../Input/tickerlist.txt", 'r') as f:
    tickerlist = [line.strip().upper() for line in f.readlines()]


processed_dir = "..\data\ProcessedData"
for ticker in tickerlist:
    ticker_dir = os.path.join(processed_dir, ticker)


    for directory in os.listdir(ticker_dir):
        list_of_df = []
        dir_path = os.path.join(ticker_dir, directory)
        for filename in os.listdir(dir_path):
            if filename.endswith(".csv"):
                filepath = os.path.join(dir_path, filename)
                dataframe_slice = pd.read_csv(filepath)
                dataframe_slice = dataframe_slice.iloc[:1]
                dataframe_slice['time'] = filename[-8:-4]

                # move "time" column to the first position
                cols = list(dataframe_slice.columns)
                cols = [cols[-1]] + cols[:-1]
                dataframe_slice = dataframe_slice[cols]

                # do something with the modified dataframe_slice
                if len(list_of_df) > 2:
                    if dataframe_slice.loc[0]['Current Stock Price'] < list_of_df[-1].loc[0]['Current Stock Price']:
                         dataframe_slice['Up or down'] = "Down"
                    elif dataframe_slice.loc[0]['Current Stock Price'] > list_of_df[-1].loc[0]['Current Stock Price']:
                         dataframe_slice['Up or down'] = "Up"

                list_of_df.append(dataframe_slice)
        df = pd.concat(list_of_df, ignore_index=True)

        print(df)
        for line in df:

            df['B1% Change'] = ((df['Bonsai Ratio'] - df['Bonsai Ratio'].shift(1)) / df['Bonsai Ratio'].shift(1)) * 100
            df['B2% Change'] = ((df['Bonsai Ratio 2'] - df['Bonsai Ratio 2'].shift(1)) / df['Bonsai Ratio 2'].shift(1)) * 100
            # ###     # add 1 hour later price data to check for corr.

            df['6 hour later change %'] = df['Current SP % Change(LAC)'] - df['Current SP % Change(LAC)'].shift(-360)
            df['5 hour later change %'] = df['Current SP % Change(LAC)'] - df['Current SP % Change(LAC)'].shift(-300)
            df['4 hour later change %'] = df['Current SP % Change(LAC)'] - df['Current SP % Change(LAC)'].shift(-240)
            df['3 hour later change %'] = df['Current SP % Change(LAC)'] - df['Current SP % Change(LAC)'].shift(-180)
            df['2 hour later change %'] = df['Current SP % Change(LAC)'] - df['Current SP % Change(LAC)'].shift(-120)
            df['1 hour later change %'] = df['Current SP % Change(LAC)'] - df['Current SP % Change(LAC)'].shift(-60)
            df['45 min later change %'] = df['Current SP % Change(LAC)'] - df['Current SP % Change(LAC)'].shift(-45)
            df['30 min later change %'] = df['Current SP % Change(LAC)'] - df['Current SP % Change(LAC)'].shift(-30)
            df['20 min later change %'] =  df['Current SP % Change(LAC)'] - df['Current SP % Change(LAC)'].shift(-20)
            df['15 min later change %'] =  df['Current SP % Change(LAC)'] - df['Current SP % Change(LAC)'].shift(-15)
            df['10 min later change %'] = df['Current SP % Change(LAC)'] - df['Current SP % Change(LAC)'].shift(-10)
            df['5 min later change %'] =  df['Current SP % Change(LAC)'] - df['Current SP % Change(LAC)'].shift(-5)

        cols.insert(1, cols.pop(cols.index('time')))
        # df = df.loc[:, cols]
        print(df.columns)
        output_dir = Path(f"corr/{ticker}/")
        output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
        output_dir2 = Path(f"dailyDF/{ticker}")
        output_dir2.mkdir(mode=0o755, parents=True, exist_ok=True)

        df.corr().to_csv(f"corr/{ticker}/{directory}.csv")
        df.to_csv(f"dailyDF/{ticker}/{directory}.csv")
    #
    #
    #
    # #combine bonsai # and itmpcrv,  then bonsai and niv?  hwat else
