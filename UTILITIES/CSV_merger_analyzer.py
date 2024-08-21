import pandas as pd
import os
import csv
from collections import defaultdict

def analyze_spy_data(base_path):
    option_chain_path = os.path.join(base_path, "optionchain", "SPY")
    processed_data_path = os.path.join(base_path, "ProcessedData", "SPY")

    option_data_summary, option_problems = analyze_directory(option_chain_path, "SPY Option Chain")
    stock_data_summary, stock_problems = analyze_directory(processed_data_path, "SPY Stock Data")

    return option_data_summary, stock_data_summary, option_problems, stock_problems

def analyze_directory(directory_path, data_type):
    all_columns = set()
    column_counts = defaultdict(int)
    file_count = 0
    problematic_files = []

    for root, dirs, files in os.walk(directory_path):
        print(dirs)
        for file in files:
            if file.endswith('.csv') and file.startswith('SPY'):
                file_path = os.path.join(root, file)

                try:
                    # Try reading with pandas first
                    df = pd.read_csv(file_path, nrows=5)
                    all_columns.update(df.columns)
                    for col in df.columns:
                        column_counts[col] += 1
                    file_count += 1
                except pd.errors.EmptyDataError:
                    print(f"Empty file: {file_path}")
                    problematic_files.append((file_path, "Empty file"))
                except Exception as e:
                    print(f"Error reading file with pandas {file_path}: {e}")
                    # If pandas fails, try with csv module
                    try:
                        with open(file_path, 'r', newline='') as csvfile:
                            dialect = csv.Sniffer().sniff(csvfile.read(1024))
                            csvfile.seek(0)
                            reader = csv.reader(csvfile, dialect)
                            headers = next(reader)
                            if headers:
                                all_columns.update(headers)
                                for col in headers:
                                    column_counts[col] += 1
                                file_count += 1
                            else:
                                print(f"No headers found in file: {file_path}")
                                problematic_files.append((file_path, "No headers found"))
                    except Exception as csv_e:
                        print(f"Error reading file with csv module {file_path}: {csv_e}")
                        problematic_files.append((file_path, str(csv_e)))

    summary_df = pd.DataFrame({
        'column_name': list(all_columns),
        'files_present': [column_counts[col] for col in all_columns],
        'total_files': file_count
    })
    summary_df['presence_percentage'] = (summary_df['files_present'] / file_count) * 100
    summary_df = summary_df.sort_values('presence_percentage', ascending=False)

    print(f"\n{data_type} Summary:")
    print(f"Total number of files processed: {file_count}")
    print("\nColumn Summary:")
    print(summary_df)

    if problematic_files:
        print("\nProblematic Files:")
        for file, error in problematic_files:
            print(f"{file}: {error}")

    return summary_df, problematic_files

# Example usage
base_path = r"H:\stockalgo_data\data"
option_summary, stock_summary, option_problems, stock_problems = analyze_spy_data(base_path)

# Save the results
option_summary.to_csv('SPY_option_data_summary.csv', index=False)
stock_summary.to_csv('SPY_stock_data_summary.csv', index=False)

# Save problematic files list
with open('SPY_problematic_files.txt', 'w') as f:
    f.write("SPY Option Chain Problematic Files:\n")
    for file, error in option_problems:
        f.write(f"{file}: {error}\n")
    f.write("\nSPY Stock Data Problematic Files:\n")
    for file, error in stock_problems:
        f.write(f"{file}: {error}\n")

print("\nAnalysis complete. Results saved to CSV files and problematic files listed in SPY_problematic_files.txt")