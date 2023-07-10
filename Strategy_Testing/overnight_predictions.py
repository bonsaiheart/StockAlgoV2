import os
import pandas as pd
from Strategy_Testing.Trained_Models import trained_daily_models

directory = '../data/ProcessedData/SPY'  # Directory containing the subdirectories

df_list = []

for subdir in os.listdir(directory):
    subdir_path = os.path.join(directory, subdir)

    if os.path.isdir(subdir_path):
        csv_files = [file for file in os.listdir(subdir_path) if file.endswith('.csv')]

        if len(csv_files) > 0:
            first_file_path = os.path.join(subdir_path, csv_files[0])
            first_df = pd.read_csv(first_file_path)

            if 'Current Stock Price' in first_df.columns:
                open_value = first_df['Current Stock Price'].iloc[0]

                last_file_path = os.path.join(subdir_path, csv_files[-1])
                last_df = pd.read_csv(last_file_path)
                last_df.at[0, 'Open'] = open_value
                df_list.append(last_df.head(1))

# Concatenate all the DataFrames in the list
result_df = pd.concat(df_list)
BuyHistA1 = trained_daily_models.A1_Buy_historical_prediction(result_df)
result_df['BuyHistA1'] = BuyHistA1
SellHistA1 = trained_daily_models.A1_Sell_historical_prediction(result_df)
result_df['SellHistA1'] = SellHistA1

# Save the concatenated DataFrame to a CSV file
result_df.to_csv('overnight_Prediction.csv', index=False)
