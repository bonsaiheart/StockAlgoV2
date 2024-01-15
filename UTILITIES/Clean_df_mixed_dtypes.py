import os
import pandas as pd
from collections import Counter
"""
info:  This should find dataframes that have mixed dtypes in columns; and attempts to convert all data in each column to the most common dtype in the respective column.
    - Shoul also remove all columns not in a defined list in the dir. (will be generated if missing.)
    -Caution: This will be easy to forget, if you've added new feature columns they may be removed here if you don't add.-
"""

def most_common_type(col):
    # types = col.dropna().apply(lambda x: type(x).__name__)
    # type_counts = Counter(types)
    type_counts = Counter(col.apply(lambda x: type(x).__name__))

    # Print all unique types and their counts in the column
    if len(type_counts) > 1:
        print(f"Column '{col.name}' has the following types: {type_counts}")
    else:
        print(f"Column '{col.name}' has only one type: {type_counts}")

    # Find and return the most common type if more than one type exists
    if len(type_counts) > 1:
        most_common = type_counts.most_common(1)[0][0]
        return most_common
    else:
        return None

def convert_column(col, target_type):
    try:
        if target_type == 'str':
            return col.astype(str)
        elif target_type == 'int':
            return pd.to_numeric(col, downcast='integer', errors='coerce')
        elif target_type == 'float':
            return pd.to_numeric(col, errors='coerce')
    except Exception as e:
        print(f"Error converting column '{col.name}': {e}")
        return col  # Return original column if conversion fails

prefixes_to_match = ["SPY", "GOOG", "TSLA"]
dir_path = "../data/historical_multiday_minute_DF/"
unique_columns = set()
# Read columns to keep from file
columns_to_keep_file = 'columns_to_keep.txt'
with open(columns_to_keep_file, 'r') as file:
    columns_to_keep = {line.strip() for line in file}

for filename in os.listdir(dir_path):
    output_file_path = f'{filename}_column_names.txt'

    filepath = os.path.join(dir_path, filename)

    if filename.endswith(".csv") and any(filename.startswith(prefix) for prefix in prefixes_to_match):
        df = pd.read_csv(filepath)

        # Keep only the columns that are in columns_to_keep
        df = df[df.columns[df.columns.isin(columns_to_keep)]]

        for col in df.columns:
            unique_columns.add(col)
        # Open the file and write column names
        for col_name in df.columns:
            col = df[col_name]
            target_type = most_common_type(col)
            if target_type==None:
                continue

            if target_type and target_type != col.dtype.name:
                print(f"Converting column '{col_name}' in file '{filename}' to {target_type}.")
                df[col_name] = convert_column(col, target_type)

        # Update new_dirpath to the directory where you want to save the files
        new_dirpath = dir_path
        if not os.path.exists(new_dirpath):
            os.makedirs(new_dirpath)
        new_filename = "new"+filename
        new_filepath = os.path.join(new_dirpath, new_filename)

        # Write unique column names to the file
        with open(new_filepath, 'w') as file:
            for col in unique_columns:
                file.write(col + '\n')

        print("Unique column names written to:", new_filepath)
        # df.to_csv(os.path.join(new_dirpath, f"cleaned_{filename}"), index=False)
      # Save the cleaned DataFrame
        cleaned_file_path = os.path.join(dir_path, f"{filename}")
        df.to_csv(cleaned_file_path, index=False)

        print(f"Cleaned file saved: {cleaned_file_path}")