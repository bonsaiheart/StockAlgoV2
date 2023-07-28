
from datetime import datetime, timedelta
import asyncio
import os
import traceback
import trade_algos
from UTILITIES import check_Market_Conditions
import tradierAPI_marketdata
from IB import ibAPI

log_dir = "errorlog"
log_file = "error.log"
###7/27 taking 51.3 seconds currently.
async def ib_connect_and_main():
    while True:
        await ibAPI.ib_connect()  # Connect to IB here
        await asyncio.sleep(15 * 60)
        print('running ib_connect_and_main again.')

async def run_program():
    await asyncio.gather(ib_connect_and_main(), main())

async def main():
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, log_file)
    max_retries = 4
    retry_delay = 5  # seconds

    while True:
        start_time = datetime.now()
        print(start_time)
        try:
            if check_Market_Conditions.is_market_open_now():
                with open("UTILITIES/tickerlist.txt", "r") as f:
                    tickerlist = [line.strip().upper() for line in f.readlines()]

                for ticker in tickerlist:
                    ticker = ticker.upper()
                    print(ticker)
                    LAC, current_price, price_change_percent, StockLastTradeTime, this_minute_ta_frame, closest_exp_date = await tradierAPI_marketdata.get_options_data(ticker)

                    print("getoptindata complete")
                    (
                        optionchain,
                        dailyminutes,
                        processeddata,
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
                    print(current_price)
                    if ticker == "SPY":
                        await trade_algos.actions(optionchain, dailyminutes, processeddata, ticker, current_price)

            else:
                with open(log_path, "a") as f:
                    f.write(f"Ran at {datetime.now()}. Market was closed today.\n")

        except Exception as e:
            with open(log_path, "a") as f:
                print(traceback.format_exc())
                print(f"Error occurred: {traceback.format_exc()}. Retrying in {retry_delay} seconds...")
                f.write(
                    f"Ran at {datetime.now()}. Occurred on attempt: {traceback.format_exc()}. Retrying in {retry_delay} seconds... \n"
                )
        current_time = datetime.now()
        next_iteration_time = start_time + timedelta(seconds=60)
        _60sec_countdown = (next_iteration_time - current_time).total_seconds()
        print(start_time, next_iteration_time, current_time, _60sec_countdown)
        await asyncio.sleep(_60sec_countdown)  # Delay for 60 seconds before the next iteration

if __name__ == "__main__":
    try:
        asyncio.run(run_program())
    finally:
        ibAPI.ib_disconnect()  # Disconnect at the end of the script
