import cProfile
import logging
from datetime import datetime, timedelta
import asyncio
import os
import traceback

import calculations
# import new_marketdata
import trade_algos
import trade_algos2
from UTILITIES import check_Market_Conditions
import tradierAPI_marketdata
from IB import ibAPI
import aiohttp
from UTILITIES.logger_config import logger


async def handle_ticker_cycle(session, ticker):
    while True:
        start_time = datetime.now()
        try:
            data = await get_options_data_for_ticker(session, ticker)
            await handle_ticker( *data)

            end_time = datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()
            sleep_time = max(0, 60 - elapsed_time)
            await asyncio.sleep(sleep_time)
        except Exception as e:
            print(f"Error occurred for ticker {ticker}: {traceback.format_exc()}")
            logger.exception(f"Error occurred for ticker {ticker}: {e}")
            await asyncio.sleep(60)  # If there's an error, we can still wait for 60 seconds or handle it differently


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


semaphore = asyncio.Semaphore(500)
async def get_options_data_for_ticker(session, ticker):
    ticker = ticker.upper()
    try:
        LAC, current_price, price_change_percent, StockLastTradeTime, this_minute_ta_frame, closest_exp_date,YYMMDD = await tradierAPI_marketdata.get_options_data(session, ticker)
        print(f"{ticker} OptionData complete at {datetime.now()}.")
        return ticker, LAC, current_price, price_change_percent, StockLastTradeTime, this_minute_ta_frame, closest_exp_date,YYMMDD
    except Exception as e:
        logger.exception(f"Error in get_options_data for {ticker}: {e}")
        raise
async def handle_ticker( ticker, LAC, current_price, price_change_percent, StockLastTradeTime, this_minute_ta_frame, closest_exp_date,YYMMDD):
    try:
        (optionchain, dailyminutes, processeddata, ticker) = calculations.perform_operations(
            ticker, LAC, current_price, price_change_percent, StockLastTradeTime, this_minute_ta_frame, closest_exp_date,YYMMDD
        )
        print(f"{ticker} PerformOptions complete at {datetime.now()}.")
    except Exception as e:
        logger.exception(f"Error in perform_operations for {ticker}: {e}")
        raise

    if ticker in ["SPY", "TSLA", "GOOG"]:
        try:
            asyncio.create_task(trade_algos2.actions(optionchain, dailyminutes, processeddata, ticker, current_price))
            print(f"{ticker} Actions complete at {datetime.now()}.")
        except Exception as e:
            logger.exception(f"Error in actions for {ticker}: {e}")
            raise

async def main():
    async with aiohttp.ClientSession() as session:
        while True:
            start_time = datetime.now()
            print(start_time)

            try:
                with open("UTILITIES/tickerlist.txt", "r") as f:
                    tickerlist = [line.strip().upper() for line in f.readlines()]

                # Compute the delay interval
                delay_interval = 60.0 / len(tickerlist)

                tasks = []
                for i, ticker in enumerate(tickerlist):
                    await asyncio.sleep(i * delay_interval)  # This will stagger the start times
                    task = asyncio.create_task(handle_ticker_cycle(session, ticker))
                    tasks.append(task)

                # Wait for all ticker tasks (this will effectively never end unless there's an exception)
                await asyncio.gather(*tasks)

            except Exception as e:
                print(f"Error occurred in aio session: {traceback.format_exc()}")
                logger.exception(f"Error occurred in aio session: {e}")

            current_time = datetime.now()
            next_iteration_time = start_time + timedelta(seconds=60)
            _60sec_countdown = (next_iteration_time - current_time).total_seconds()
            print(start_time, next_iteration_time, current_time, "Time Remaining in Loop:", _60sec_countdown,
                  "~~~~~~~~")

if __name__ == "__main__":
    try:
        asyncio.run(run_program())
    finally:
        ibAPI.ib_disconnect()  # Disconnect at the end of the script
