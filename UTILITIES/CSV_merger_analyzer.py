import pandas as pd
import os
import glob
from collections import defaultdict

def analyze_csv_structure(base_path):
    option_chain_path = os.path.join(base_path, "optionchain")
    processed_data_path = os.path.join(base_path, "ProcessedData")

    option_data_summary = analyze_directory(option_chain_path, "Option Chain")
    stock_data_summary = analyze_directory(processed_data_path, "Stock Data")

    return option_data_summary, stock_data_summary

def analyze_directory(directory_path, data_type):
    all_columns = set()
    column_counts = defaultdict(int)
    file_count = 0
    tickers = set()

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                ticker = file.split('_')[0]
                tickers.add(ticker)

                try:
                    df = pd.read_csv(file_path, nrows=1)
                    all_columns.update(df.columns)
                    for col in df.columns:
                        column_counts[col] += 1
                    file_count += 1
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    summary_df = pd.DataFrame({
        'column_name': list(all_columns),
        'files_present': [column_counts[col] for col in all_columns],
        'total_files': file_count
    })
    summary_df['presence_percentage'] = (summary_df['files_present'] / file_count) * 100
    summary_df = summary_df.sort_values('presence_percentage', ascending=False)

    print(f"\n{data_type} Summary:")
    print(f"Total number of files processed: {file_count}")
    print(f"Number of unique tickers: {len(tickers)}")
    print(f"Tickers: {', '.join(sorted(tickers))}")
    print("\nColumn Summary:")
    print(summary_df)

    return summary_df

# Example usage
base_path = r"H:\stockalgo_data\data"
option_summary, stock_summary = analyze_csv_structure(base_path)

# Optionally, save the results
option_summary.to_csv('option_data_summary.csv', index=False)
stock_summary.to_csv('stock_data_summary.csv', index=False)