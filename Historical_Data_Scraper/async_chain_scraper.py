import os
import time
import asyncio
import aiohttp
import csv

import requests
from bs4 import BeautifulSoup
import re
from aiohttp import ClientError
from requests import RequestException


firstpage = 0
lastpage = 2000
filename = f"Pages_{firstpage}_{lastpage}.csv"


column_titles = [
    "Contract",
    "Ticker",
    "ExpDate",
    "Put_Call",
    "Strike",
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Open Interest",
]

error_log_file = open("error_log.txt", "a")  # Open the error log file for appending

missed_subdirs = []


# Check if the file exists and its size is zero
if not os.path.isfile(filename) or os.path.getsize(filename) == 0:
    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(column_titles)


async def process_subdirectory(session, subdir, writer):
    url = f"https://chartexchange.com/symbol/opra-{subdir}"
    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)

        try:
            async with session.get(url) as response:
                response.raise_for_status()
                content = await response.text()

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
                        data_row = [
                            option_contract,
                            symbol,
                            expiration_date,
                            option_type,
                            formatted_strike_price,
                            *match,
                        ]
                        writer.writerow(data_row)

        except ClientError as e:
            error_log_file.write(f"Error occurred while retrieving subdirectory {subdir}: {e}\n")
            error_log_file.write(f"max retries {subdir}: {e}\n")


async def process_subdirectories(queue, session, writer, sem):
    while not queue.empty():
        subdir = await queue.get()

        async with sem:
            await process_subdirectory(session, subdir, writer)

        queue.task_done()


async def get_and_write_data():
    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)

        try:
            with open("last_processed.txt", "r") as f:
                firstpage = int(f.readline())
                subdir_index = int(f.readline())
        except FileNotFoundError:
            firstpage = 0
            subdir_index = 0

        for page in range(firstpage, lastpage):
            url = f"https://chartexchange.com/list/optionequity/?page={page}"
            print(page)
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        response.raise_for_status()
                        soup = BeautifulSoup(await response.text(), "html.parser")

                        anchor_tags = soup.find_all("a")

                        subdirectories = []

                        for anchor in anchor_tags:
                            href = anchor.get("href")
                            if href and href.startswith("/symbol/opra-"):
                                subdirectory = href.replace("/symbol/opra-", "").strip("/")
                                subdirectories.append(subdirectory)

                        # Create a queue and populate it with subdirectories
                        queue = asyncio.Queue()
                        for subdir in subdirectories[subdir_index:]:
                            queue.put_nowait(subdir)

                        # Configure the maximum concurrency level
                        max_concurrency = 10
                        print(max_concurrency)
                        sem = asyncio.Semaphore(max_concurrency)

                        # Process the subdirectories concurrently
                        tasks = []
                        for _ in range(max_concurrency):
                            task = asyncio.create_task(process_subdirectories(queue, session, writer, sem))
                            tasks.append(task)

                        # Wait for all tasks to complete
                        await asyncio.gather(*tasks)

            except ClientError as e:
                error_log_file.write(f"Error occurred while retrieving page {page}: {e}\n")

                # Save the current progress
                with open("last_processed.txt", "w") as f:
                    f.write(str(page) + "\n")
                    f.write(str(subdir_index) + "\n")

                continue

            subdir_index = 0  # Reset the subdir_index for the next page

            # Save the last processed values of x and subdir_index
            with open("last_processed.txt", "w") as f:
                f.write(str(page + 1) + "\n")  # Add 1 to account for the current iteration
                f.write(str(subdir_index) + "\n")

            # Save the missed subdirectories to a file for later processing
            with open("missed_subdirs.txt", "a") as f:
                for subdir in missed_subdirs:
                    f.write(page + "\n")
                    f.write(subdir + "\n")


def process_missed_subdirs():
    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)
        with open("missed_subdirs.txt", "r") as read_missed_subdirs:
            lines = read_missed_subdirs.readlines()

        with open("missed_subdirs.txt", "w") as write_missed_subdir:
            for i in range(0, len(lines), 2):
                page = int(lines[i].strip())
                subdir = lines[i + 1].strip()

                max_retries = 3  # Maximum number of retries

                retry_count = 0  # Counter for retry attempts

                while retry_count < max_retries:
                    try:
                        response = requests.get(f"https://chartexchange.com/symbol/opra-{subdir}")
                        response.raise_for_status()
                        break  # Break out of the retry loop if the request is successful
                    except RequestException as e:
                        error_log_file.write(f"Error occurred while retrieving subdirectory {subdir}: {e}\n")
                        error_log_file.write(f"max retries while doing the missed subdir: {subdir}: {e}\n")
                        retry_count += 1
                        time.sleep(1)  # Wait for 1 second before retrying

                if retry_count == max_retries:
                    # Save the current progress
                    with open("last_processed.txt", "w") as f:
                        f.write(str(page) + "\n")
                        f.write(str(i) + "\n")

                content = response.content.decode("utf-8")
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
                        data_row = [
                            option_contract,
                            symbol,
                            expiration_date,
                            option_type,
                            formatted_strike_price,
                            *match,
                        ]
                        writer.writerow(data_row)

                print(f"Processing missed subdir: {subdir}, missed page: {page}")
                time.sleep(1)

                # Rewrite the remaining lines to the file
                write_missed_subdir.writelines(lines[i + 2 :])
                # Set the file position to the beginning and truncate the file
                write_missed_subdir.seek(0)
                write_missed_subdir.truncate()


while lastpage <= 2261775:
    try:
        asyncio.run(get_and_write_data())
        print(firstpage, time)
        process_missed_subdirs()
        firstpage += 2000
        lastpage += 2000
    except Exception as e:
        # Handle any exceptions that occur during execution
        error_log_file.write(f"Error occurred: {e}\n")

    finally:
        error_log_file.close()
