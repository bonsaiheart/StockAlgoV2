import asyncio
from datetime import datetime
import os
import traceback
import trade_algos
from UTILITIES import check_Market_Conditions
import tradierAPI_marketdata
from IB import ibAPI
from ib_insync import util


# import webullapi

datetime = datetime.today()
print(datetime)
###TODO took 1:32 to run on dell touchscreen. CAN run about 5-8 tickers in a minute.
import asyncio
from datetime import datetime
import os
import traceback
import trade_algos
from UTILITIES import check_Market_Conditions
import tradierAPI_marketdata
from IB import ibAPI

datetime = datetime.today()
print(datetime)

log_dir = "errorlog"
log_file = "error.log"

async def main():
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, log_file)
    max_retries = 4
    retry_delay = 5  # seconds
    success = False

    for i in range(max_retries):
        try:
            if check_Market_Conditions.is_market_open_now() == True:
                with open("UTILITIES/tickerlist.txt", "r") as f:
                    tickerlist = [line.strip().upper() for line in f.readlines()]
                await ibAPI.ib_connect()
                print(ibAPI.ib.isConnected())

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

                ibAPI.ib.disconnect()
                break
            else:
                with open(log_path, "a") as f:
                    f.write(f"Ran at {datetime}. Market was closed today.\n")

        except Exception as e:
            with open(log_path, "a") as f:
                print(traceback.format_exc())
                print(f"Error occurred: {traceback.format_exc()}.  Retrying in {retry_delay} seconds...")
                f.write(
                    f"Ran at {datetime}. Occurred on attempt #{i +1}: {traceback.format_exc()}. Retrying in {retry_delay} seconds... \n"
                )

if __name__ == "__main__":
    util.run(main())
