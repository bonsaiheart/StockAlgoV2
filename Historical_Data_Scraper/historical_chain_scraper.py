import os
import time
import requests
import csv
from bs4 import BeautifulSoup
import re
from requests.exceptions import RequestException

column_titles = ["Contract", "Ticker", "ExpDate", "Put_Call", "Strike", "Date", "Open", "High", "Low", "Close", "Volume",
                 "Open Interest"]

##await asyncio.sleep() in process_subdirectory

error_log_file = open('error_log.txt', 'a')  # Open the error log file for appending

missed_subdirs = []

filename = 'test.csv'

# Check if the file exists and its size is zero
if not os.path.isfile(filename) or os.path.getsize(filename) == 0:
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_titles)

def get_and_write_data(firstpage,lastpage):
    try:
        with open('last_processed.txt', 'r') as f:
            firstpage = int(f.readline())
            subdir_index = int(f.readline())
    except FileNotFoundError:
        last_processed = firstpage
        subdir_index = 0

    for page in range(firstpage, lastpage):
        url = f'https://chartexchange.com/list/optionequity/?page={page}'
        print(url)
        try:
            response = requests.get(url)
            response.raise_for_status()
        except RequestException as e:
            error_log_file.write(f"Error occurred while retrieving page {page}: {e}\n")

            # Save the current progress
            with open('last_processed.txt', 'w') as f:
                f.write(str(page) + '\n')
                f.write(str(subdir_index) + '\n')

            continue

        soup = BeautifulSoup(response.content, 'html.parser')


        anchor_tags = soup.find_all('a')

        subdirectories = []

        for anchor in anchor_tags:
            href = anchor.get('href')
            if href and href.startswith('/symbol/opra-'):
                subdirectory = href.replace("/symbol/opra-", "").strip('/')
                subdirectories.append(subdirectory)
                print(subdirectory)
        # Process each subdirectory
        max_retries = 3  # Maximum number of retries

        for i in range(subdir_index, len(subdirectories)):
            subdir = subdirectories[i]

            retry_count = 0  # Counter for retry attempts

            while retry_count < max_retries:
                try:
                    response = requests.get(f'https://chartexchange.com/symbol/opra-{subdir}')
                    response.raise_for_status()

                    break  # Break out of the retry loop if the request is successful
                except RequestException as e:
                    error_log_file.write(f"Error occurred while retrieving subdirectory {subdir}: {e}\n")
                    error_log_file.write(f"max retries {subdir}: {e}\n")
                    retry_count += 1
                    time.sleep(1)  # Wait for 1 second before retrying

            if retry_count == max_retries:
                # Save the current progress
                with open('last_processed.txt', 'w') as f:
                    f.write(str(page) + '\n')
                    f.write(str(i) + '\n')

                missed_subdirs.append(subdir)  # Add the missed subdir to the list
                continue

            content = response.content.decode('utf-8')

            pattern = r'\["(\d+)","([\d.]+)","([\d.]+)","([\d.]+)","([\d.]+)","(\d+)"(?:,"(\d+)")?\]'
            matches = re.findall(pattern, content)



            symbol_pattern = r"([a-zA-Z]+)(\d{8})([a-zA-Z])(\d+\.?\d*)"
            match_symbol = re.match(symbol_pattern, subdir)

            if match_symbol:
                symbol = match_symbol.group(1).upper()
                expiration_date = match_symbol.group(2)
                option_type = match_symbol.group(3).upper()
                price = float(match_symbol.group(4))
                formatted_strike_price = f"{price:.2f}".replace(".", "").zfill(8)
                option_contract = f"{symbol}{expiration_date}{option_type}{formatted_strike_price}"

                for match in matches:
                    print(option_contract)
                    data_row = [option_contract, symbol, expiration_date, option_type, formatted_strike_price, *match]
                    writer.writerow(data_row)
        # Remove successfully processed subdirectories from the missed_subdirs list

        subdir_index = 0  # Reset the subdir_index for the next page

        # Save the last processed values of x and subdir_index
        with open('last_processed.txt', 'w') as f:
            f.write(str(page + 1) + '\n')  # Add 1 to account for the current iteration
            f.write(str(subdir_index) + '\n')


        # Save the missed subdirectories to a file for later processing
        with open('missed_subdirs.txt', 'a') as f:
            for subdir in missed_subdirs:
                f.write(page + '\n')
                f.write(subdir + '\n')

    error_log_file.close()

def process_missed_subdirs(writer):
    with open('missed_subdirs.txt', 'r') as read_missed_subdirs:
        lines = read_missed_subdirs.readlines()

    with open('missed_subdirs.txt', 'w') as write_missed_subdir:
        for i in range(0, len(lines), 2):
            page = int(lines[i].strip())
            subdir = lines[i + 1].strip()

            max_retries = 3  # Maximum number of retries



            retry_count = 0  # Counter for retry attempts

            while retry_count < max_retries:
                try:
                    response = requests.get(f'https://chartexchange.com/symbol/opra-{subdir}')
                    response.raise_for_status()
                    print(response)
                    break  # Break out of the retry loop if the request is successful
                except RequestException as e:
                    error_log_file.write(f"Error occurred while retrieving subdirectory {subdir}: {e}\n")
                    retry_count += 1
                    time.sleep(1)  # Wait for 1 second before retrying

            if retry_count == max_retries:
                # Save the current progress
                with open('last_processed.txt', 'w') as f:
                    f.write(str(page) + '\n')
                    f.write(str(i) + '\n')


            content = response.content.decode('utf-8')
            print("content", content)
            pattern = r'\["(\d+)","([\d.]+)","([\d.]+)","([\d.]+)","([\d.]+)","(\d+)"(?:,"(\d+)")?\]'
            matches = re.findall(pattern, content)
            print(matches)

            print(subdir, "SUBBBBBB")
            symbol_pattern = r"([a-zA-Z]+)(\d{8})([a-zA-Z])(\d+\.?\d*)"
            match_symbol = re.match(symbol_pattern, subdir)
            print(match_symbol, "MATACHHHH")
            if match_symbol:
                symbol = match_symbol.group(1).upper()
                expiration_date = match_symbol.group(2)
                option_type = match_symbol.group(3).upper()
                price = float(match_symbol.group(4))
                formatted_strike_price = f"{price:.2f}".replace(".", "").zfill(8)
                option_contract = f"{symbol}{expiration_date}{option_type}{formatted_strike_price}"

                for match in matches:
                    print(option_contract)
                    data_row = [option_contract, symbol, expiration_date, option_type, formatted_strike_price, *match]
                    writer.writerow(data_row)

            print(f"Processing missed subdir: {subdir}, missed page: {page}")
            time.sleep(1)

        # Rewrite the remaining lines to the file
            write_missed_subdir.writelines(lines[i+2:])
            # Set the file position to the beginning and truncate the file
            write_missed_subdir.seek(0)
            write_missed_subdir.truncate()

        # Rewrite the remaining lines to the file


# Call the function within the with block
with open(filename, 'a', newline='') as file:
    writer = csv.writer(file)
    get_and_write_data(firstpage,lastpage)
    process_missed_subdirs(writer)