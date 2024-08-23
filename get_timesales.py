import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import PrivateData.tradier_info
import aiohttp
import numpy as np
import pandas as pd
import ta
from ta import momentum,trend,volume,volatility

from UTILITIES.logger_config import logger
from tradierAPI_marketdata import fetch

import pytz

def unix_to_eastern_hhmm(timestamp):
  # Convert Unix timestamp to datetime object (assuming timestamp is in seconds)
  dt = datetime.fromtimestamp(timestamp, tz=pytz.timezone('US/Eastern'))

  # Format the datetime object to HH:MM
  return dt.strftime('%Y%m%d %H:%M')

# Example usage
timestamp = 1724049300
eastern_time_hhmm = unix_to_eastern_hhmm(timestamp)
print(eastern_time_hhmm)
paper_auth = PrivateData.tradier_info.paper_auth
real_acc = PrivateData.tradier_info.real_acc
real_auth = PrivateData.tradier_info.real_auth

client_session = None
async def create_client_session():
    global client_session
    if client_session is not None:
        await client_session.close()  # Close the existing session if it's not None
    client_session = aiohttp.ClientSession()


async def get_ta(session, ticker):
    start = (datetime.today() - timedelta(days=5)).strftime("%Y-%m-%d %H:%M")
    end = datetime.today().strftime("%Y-%m-%d %H:%M")
    headers = {f"Authorization": f"Bearer {real_auth}", "Accept": "application/json"}
    # TODO move this into getoptions data?  then I can run it thru calc ANYTIME b/c it will have all data.  This can be processpooled.
    time_sale_response = await fetch(
        session,
        "https://api.tradier.com/v1/markets/timesales",
        params={
            "symbol": ticker,
            "interval": "1min",
            "start": start,
            "end": end,
            "session_filter": "all",
        },
        headers=headers,
    )

    if (
        time_sale_response
        and "series" in time_sale_response
        and "data" in time_sale_response["series"]
    ):
        df = pd.DataFrame(time_sale_response["series"]["data"]).set_index("time")
    else:
        print(
            f"Failed to retrieve TA data for ticker {ticker}: time_sale_response or required keys are missing or None"
        )
        return None

    df.index = pd.to_datetime(df.index)
    return df
async def main():
    await create_client_session()
    session = client_session
    df = await get_ta(session, "EFRA")
    # df['timestamp'] = df['timestamp'] /1000
    # Convert each timestamp in the Series (directly pass datetime objects)

    df['timestamp'] = df['timestamp'].apply(unix_to_eastern_hhmm)
    df.to_csv('aaa_df.csv')

asyncio.run(main())