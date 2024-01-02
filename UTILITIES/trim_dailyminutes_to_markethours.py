import pandas as pd
import os


def filter_row(row):
    # Check for null or missing values
    if pd.isnull(row["LastTradeTime"]):
        return False  # Exclude rows with null 'LastTradeTime'

    try:
        # Extracts 'HHMM' and converts to integer
        last_time_str = row["LastTradeTime"].split("_")[1]
        last_time = int(last_time_str)
        print(f"Processing time: {last_time_str} -> {last_time}")  # Debug print

        # Check if time is within the range 09:30 to 16:00
        within_range = 930 <= last_time <= 1600
        print(f"Time within range: {within_range}")  # Debug print
        return within_range
    except (ValueError, IndexError) as e:
        print(
            f"Error processing row: {row['LastTradeTime']}, Error: {e}"
        )  # Debug print
        return False  # Exclude row in case of error


def filter_csv_files(directory):
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                print(file)
                file_path = os.path.join(subdir, file)
                df = pd.read_csv(file_path)

                # Apply the filter condition
                df_filtered = df[df.apply(filter_row, axis=1)]

                # Save the filtered data back to CSV
                df_filtered.to_csv(file_path, index=False)


filter_csv_files(r"C:\Users\del_p\PycharmProjects\StockAlgoV2\data\DailyMinutes")
