import os
import re
from datetime import datetime
import pandas as pd

root_directory_path = r'H:\stockalgo_data\data\processeddata'

for ticker_dir in os.listdir(root_directory_path):
    if not os.path.isdir(os.path.join(root_directory_path, ticker_dir)):
        continue
    print(ticker_dir)
    ticker_path = os.path.join(root_directory_path, ticker_dir)
    dfs = []

    for date_dir in os.listdir(ticker_path):
        date_path = os.path.join(ticker_path, date_dir)
        if not os.path.isdir(date_path):
            continue
        print(date_dir)
        for file in os.listdir(date_path):
            if file.endswith(".csv"):
                try:
                    filename = os.path.join(date_path, file)
                    match = re.search(r"(\d{6})_(\d{4})\.csv$", file)
                    if match:
                        datetime_str = f"20{match.group(1)}{match.group(2)}"
                        dt_object = datetime.strptime(datetime_str, "%Y%m%d%H%M")

                        df = pd.read_csv(filename, nrows=1)
                        df['fetch_timestamp'] = dt_object
                        dfs.append(df)
                except Exception as e:
                    print(f"Error reading file {ticker_dir}/{date_dir}/{file}: {e}")

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(f'processeddata_combined_{ticker_dir}.csv', index=False)

# Now you have a dictionary 'combined_dfs' where keys are tickers
# and values are the combined DataFrames for each ticker
# e.g.: combined_dfs['spy'] will give you the combined DataFrame for SPY option chain data
# ... Access other ticker DataFrames similarly.


# # You could now combine all the dfs here
# all_dfs = pd.concat(combined_dfs, ignore_index=True)
# print(all_dfs.head().to_markdown(index=False, numalign="left", stralign="left"))
