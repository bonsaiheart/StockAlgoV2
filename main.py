import cProfile
import logging
from datetime import datetime, timedelta
import asyncio
import os
import traceback
import trade_algos
from UTILITIES import check_Market_Conditions
import tradierAPI_marketdata
from IB import ibAPI
import aiohttp
from UTILITIES.logger_config import logger




async def profiled_actions(optionchain, dailyminutes, processeddata, ticker, current_price):
    pr = cProfile.Profile()
    pr.enable()

    try:
        await trade_algos.actions(optionchain, dailyminutes, processeddata, ticker, current_price)
    except Exception as e:
        print(f"Error occurred: {traceback.format_exc()}")

    pr.disable()
    pr.print_stats()


#TODO actions is taking 16 of the 35 seconds.
async def ib_connect_and_main():
    while True:
        await ibAPI.ib_connect()  # Connect to IB here
        await asyncio.sleep(5 * 60)
        print('running ib_connect_and_main again.')

async def run_program():
    await asyncio.gather(ib_connect_and_main(), main())
async def handle_ticker(session, ticker):
    ticker = ticker.upper()
    try:
        try:
            LAC, current_price, price_change_percent, StockLastTradeTime, this_minute_ta_frame, closest_exp_date = await tradierAPI_marketdata.get_options_data(session,ticker)
            print(f"{ticker} OptionData complete at {datetime.now()}.")
        except Exception as e:
            logger.exception(f"Error in get_options_data for {ticker}: {e}")
            raise

        try:
            (optionchain, dailyminutes, processeddata, ticker) = tradierAPI_marketdata.perform_operations(
                ticker, LAC, current_price, price_change_percent, StockLastTradeTime, this_minute_ta_frame, closest_exp_date
            )
            print(f"{ticker} PerformOptions complete at {datetime.now()}.")
        except Exception as e:
            logger.exception(f"Error in perform_operations for {ticker}: {e}")
            raise

        if ticker in ["SPY", "TSLA", "GOOG"]:
            try:
                asyncio.create_task(trade_algos.actions(optionchain, dailyminutes, processeddata, ticker, current_price))
                print(f"{ticker} Actions complete at {datetime.now()}.")
            except Exception as e:
                logger.exception(f"Error in actions for {ticker}: {e}")
                raise

    except Exception as e:
        print(f"Error occurred: {traceback.format_exc()}")
        logger.exception(f"An error occurred while handling ticker {ticker}: {e}")


async def main():
    max_retries = 4
    retry_delay = 5  # seconds

    async with aiohttp.ClientSession() as session:
        while True:
            start_time = datetime.now()
            print(start_time)
            try:
                # if check_Market_Conditions.is_market_open_now() == True or False:
                with open("UTILITIES/tickerlist.txt", "r") as f:
                    tickerlist = [line.strip().upper() for line in f.readlines()]


                #JUST CHSANGED THIS 9.1.23
                tasks = [asyncio.create_task(handle_ticker(session, ticker)) for ticker in tickerlist]
                await asyncio.gather(*tasks)
                # Use session in loop
                # await asyncio.gather(*(handle_ticker(session, ticker) for ticker in tickerlist))
            # else:
            #     with open(log_path, "a") as f:
            #         f.write(f"Ran at {datetime.now()}. Market was closed today.\n")

            except Exception as e:
                print(f"Error occurred in aio session: {traceback.format_exc()}")
                logger.exception(f"Error occurred in aio session: {e}")

            current_time = datetime.now()
            next_iteration_time = start_time + timedelta(seconds=60)
            _60sec_countdown = (next_iteration_time - current_time).total_seconds()
            print(start_time, next_iteration_time, current_time,"Time Remaining in Loop: ", _60sec_countdown,"~~~~~~~~")
            await asyncio.sleep(_60sec_countdown)  # Delay for 60 seconds before the next iteration

if __name__ == "__main__":
    try:
        asyncio.run(run_program())
    finally:
        ibAPI.ib_disconnect()  # Disconnect at the end of the script
