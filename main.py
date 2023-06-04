from datetime import datetime
import os
import time
import traceback

import check_Market_Conditions
import tradierAPI_marketdata

# import webullapi

datetime = datetime.today()
print(datetime)
###TODO took 1:32 to run on dell touchscreen. CAN run about 5-8 tickers in a minute.
# SPY
# UVXY
# TSLA
# ROKU
# CHWY
# BA
# CMPS
# MNMD
# GOEV
# W
# OSTK
# MSFT
# GOOG


####TODO make sure it runs at d-9am on days when market is open.
with open("Input/tickerlist.txt", "r") as f:
    tickerlist = [line.strip().upper() for line in f.readlines()]

log_dir = "errorlog"
log_file = "error.log"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_path = os.path.join(log_dir, log_file)

max_retries = 4
retry_delay = 5  # seconds
success = False

for i in range(max_retries):
    try:
        if checkConditions.is_market_open_today() == True:
            for ticker in tickerlist:
                print(ticker)
                (
                    LAC,
                    current_price,
                    price_change_percent,
                    StockLastTradeTime,
                    this_minute_ta_frame,
                    closest_exp_date,
                ) = tradierAPI_marketdata.get_options_data(ticker)
                (
                    optionchain,
                    processeddata,
                    x,
                    closest_exp_date,
                    ticker,
                ) = tradierAPI_marketdata.perform_operations(
                    ticker,
                    LAC,
                    current_price,
                    price_change_percent,
                    StockLastTradeTime,
                    this_minute_ta_frame,
                    closest_exp_date,
                )

                tradierAPI_marketdata.actions(optionchain, processeddata, x, closest_exp_date, ticker)

                # email_me.email_me(processeddata)
            break
        else:
            with open(log_path, "a") as f:
                f.write(f"Ran at {datetime}. Market was closed today.\n")
            break

    except Exception as e:
        with open(log_path, "a") as f:
            print(traceback.format_exc())
            print(f"Error occurred: {traceback.format_exc()}.  Retrying in {retry_delay} seconds...")
            f.write(
                f"Ran at {datetime}. Occurred on attempt #{i +1}: {traceback.format_exc()}. Retrying in {retry_delay} seconds... \n"
            )

    time.sleep(retry_delay)

if i == max_retries - 1:
    print(f"Maximum retries exceeded. Exiting...")
    with open(log_path, "a") as f:
        f.write(f"Ran at {datetime}. Max Retries exceeded.  Exiting... \n")
else:
    if success:
        print(f"Success!")
