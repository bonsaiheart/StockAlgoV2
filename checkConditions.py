import datetime
from datetime import date, timedelta

import pandas_market_calendars as mcal


def is_market_open_today():
    today_str = date.today().strftime("%Y-%m-%d")
    start_date = date.today() - timedelta(days=3)
    end_date = date.today() + timedelta(days=5)
    nyse = mcal.get_calendar("NYSE")
    nyse_schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    # display(nyse_schedule)

    if today_str in nyse_schedule.index:
        print(f"Today {today_str} stock market is open.")
        is_market_open = True

    else:
        print(f"Today {today_str} stock market is NOT open.")
        is_market_open = "Closed"

    return is_market_open
