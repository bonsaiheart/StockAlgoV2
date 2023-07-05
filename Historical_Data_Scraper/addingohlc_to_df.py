import pandas as pd

# Load the CSV files into DataFrames
existingwithoutopenhighlowclose = pd.read_csv(r'C:\Users\natha\PycharmProjects\StockAlgoV2\Historical_Data_Scraper\data\Historical_Processed_ChainData\SPY.csv')
hasopenhighlowclose = pd.read_csv(r'C:\Users\natha\PycharmProjects\StockAlgoV2\Historical_Data_Scraper\data\historical_optionchain\SPY.csv')

print(hasopenhighlowclose.columns)
# Select desired columns from one of the DataFrames (df1)
selected_columns = ['Open', 'High', 'Low','Close','Date']
hasopenhighlowclose = hasopenhighlowclose[selected_columns]

# Remove duplicates from the DataFrame that contains duplicates (df2)
hasopenhighlowclose = hasopenhighlowclose.drop_duplicates(subset='Date')

# Merge or concatenate based on the "Date" column
merged_df = pd.merge(existingwithoutopenhighlowclose, hasopenhighlowclose, on='Date')

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('MERGEDNEWWOW.csv', index=False)