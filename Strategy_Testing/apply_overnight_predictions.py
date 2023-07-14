import os
import pandas as pd
from Strategy_Testing.Trained_Models import trained_daily_models

directory = "../data/DailyMinutes/SPY"  # Directory containing the subdirectories
df_list = []
dailyminutes_list = sorted([file for file in os.listdir(directory) if file.endswith(".csv")])


print()

for dailyminutes in dailyminutes_list:
    dailyminutes = pd.read_csv(os.path.join(directory, dailyminutes))
    print(dailyminutes)
    print(dailyminutes['Current Stock Price'])
    open_value = dailyminutes["Current Stock Price"].iloc[0]  # Corrected column name
    dailyminutes.loc[dailyminutes.index[-1], "Open"] = open_value  # Use .loc to assign values
    print(dailyminutes)
    df_list.append(dailyminutes.tail(1))

# Concatenate all the DataFrames in the list
result_df = pd.concat(df_list)
result_df.reset_index(drop=True, inplace=True)

BuyHistA1 = trained_daily_models.A1_Buy_historical_prediction(result_df)
SellHistA1 = trained_daily_models.A1_Sell_historical_prediction(result_df)
# Create a DataFrame for BuyHistA1 and SellHistA1 predictions
predictions_df = pd.DataFrame({'BuyHistA1': BuyHistA1, 'SellHistA1': SellHistA1})

# Concatenate result_df and predictions_df
# using pd.concat(axis=1)
result_df = pd.concat([result_df, predictions_df], axis=1)
result_df.to_csv("overnight_Prediction.csv", index=True)


# Save the concatenated DataFrame to a CSV file
# result_df.to_csv("overnight_Prediction.csv", index=False)
