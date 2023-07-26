import glob
import os
import pandas as pd
from datetime import datetime
import pandas_market_calendars as mcal

today = datetime.utcnow().date()
today_timestamp = pd.to_datetime(today)


def save_market_schedule_to_file():
    nyse = mcal.get_calendar("NYSE")
    today = datetime.utcnow().date()
    market_schedule = nyse.schedule(start_date=today, end_date=today)
    market_schedule["market_open_utc"] = pd.to_datetime(market_schedule["market_open"])
    market_schedule["market_close_utc"] = pd.to_datetime(market_schedule["market_close"])

    directory = "UTILITIES"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Delete other .csv files in the directory
    file_pattern = os.path.join(directory, "*.csv")
    existing_files = glob.glob(file_pattern)
    for file in existing_files:
        os.remove(file)

    file_path = os.path.join(directory, f"market_schedule_{today}.csv")
    market_schedule.to_csv(file_path, index=True)


def is_market_open_now():
    today = datetime.utcnow().date()
    today_timestamp = pd.to_datetime(today)
    directory = "UTILITIES"
    file_path = os.path.join(directory, f"market_schedule_{today}.csv")

    if not os.path.exists(file_path):
        save_market_schedule_to_file()

    market_schedule = pd.read_csv(file_path, parse_dates=True, index_col=0)
    market_schedule["market_open_utc"] = pd.to_datetime(market_schedule["market_open"])
    market_schedule["market_close_utc"] = pd.to_datetime(market_schedule["market_close"])

    # Load the saved market schedule from file
    market_schedule.index = pd.to_datetime(market_schedule.index)  # Convert index to datetime

    if today_timestamp in market_schedule.index:
        market_open_utc = market_schedule.loc[today_timestamp, "market_open_utc"].time()
        market_close_utc = market_schedule.loc[today_timestamp, "market_close_utc"].time()

        # Get the current time in UTC
        now_utc = datetime.utcnow().time()
        if market_open_utc <= now_utc <= market_close_utc:
            print("The stock market is currently open.")
            is_market_open = True
        else:
            print("The stock market is currently closed.")
            is_market_open = False
    else:
        print("Today is not a trading day.")
        is_market_open = False

    return is_market_open


# Call this function to check if the market is open
