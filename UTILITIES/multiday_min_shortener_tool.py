import pandas as pd
import os

input_directory = r"C:\Users\del_p\PycharmProjects\StockAlgoV2\data\historical_multiday_minute_DF"
output_directory = r"C:\Users\del_p\PycharmProjects\StockAlgoV2\data\historical_multiday_minute_DF\older"

for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_directory, filename)
        df = pd.read_csv(file_path)

        # Convert 'LastTradeTime' to a datetime object with the correct format
        df['LastTradeTime'] = pd.to_datetime(df['LastTradeTime'], format='%y%m%d_%H%M')

        # Create a timestamp for '231002' and extract the date component
        timestamp_cutoff = pd.to_datetime('231002', format='%y%m%d').date()

        # Filter rows where 'LastTradeTime' is prior to the cutoff date
        filtered_df = df[df['LastTradeTime'].dt.date < timestamp_cutoff].copy()

        # Format 'LastTradeTime' to the desired format
        filtered_df['LastTradeTime'] = filtered_df['LastTradeTime'].dt.strftime('%y%m%d_%H%M')

        new_filename = filename.replace(".csv", "prior_231002.csv")
        new_file_path = os.path.join(output_directory, new_filename)

        # Use .loc to set the 'LastTradeTime' values in the filtered DataFrame
        filtered_df.loc[:, 'LastTradeTime'] = filtered_df['LastTradeTime']

        filtered_df.to_csv(new_file_path, index=False)