import cProfile
import logging
from datetime import datetime, timedelta
import asyncio
import os
import traceback

import ib_insync

import ib_insync.util
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
            await asyncio.sleep(0)
            await handle_ticker( *data)

            end_time = datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()

            sleep_time = max(0, 60 - elapsed_time)
            await asyncio.sleep(sleep_time)
        except Exception as e:
            print(f"Error occurred for ticker {ticker}: {traceback.format_exc()}")
            logger.exception(f"Error occurred for ticker {ticker}: {e}")
            end_time = datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()
            sleep_time = max(0, 60 - elapsed_time)
            await asyncio.sleep(sleep_time)

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
async def check_and_reconnect_ib():
    while True:
        ib_insync.util.getLoop()
        if not ibAPI.ib.isConnected():
            try:
                await ibAPI.ib_connect()
                logger.info("Reconnected to IB.")
            except Exception as e:
                logger.error(f"Failed to reconnect to IB: {e}")
        await asyncio.sleep(300)  # Check every 5 minutes

async def run_program():
    reconnect_task = asyncio.create_task(check_and_reconnect_ib())
    main_task = asyncio.create_task(main())

    # Wait for both tasks to complete
    # If one task is intended to run indefinitely, this will also run indefinitely
    await asyncio.gather(reconnect_task, main_task)


semaphore = asyncio.Semaphore(500)
async def get_options_data_for_ticker(session, ticker):
    ticker = ticker.upper()
    await asyncio.sleep(0)
    try:

        LAC, current_price, price_change_percent, StockLastTradeTime, this_minute_ta_frame, combined_optionchain_df,YYMMDD = await tradierAPI_marketdata.get_options_data(session, ticker)
        print(f"{ticker} OptionData complete at {datetime.now()}.")
        return ticker, LAC, current_price, price_change_percent, StockLastTradeTime, this_minute_ta_frame, combined_optionchain_df,YYMMDD
    except Exception as e:
        logger.exception(f"Error in get_options_data for {ticker}: {e}")
        raise
async def handle_ticker( ticker, LAC, current_price, price_change_percent, StockLastTradeTime, this_minute_ta_frame, combined_optionchain_df,YYMMDD):
    try:
        (combined_optionchain_df_withlac_diff, dailyminutes, processeddata, ticker) = await calculations.perform_operations(
            ticker, LAC, current_price, price_change_percent, StockLastTradeTime, this_minute_ta_frame, combined_optionchain_df,YYMMDD
        )
        print(f"{ticker} PerformOptions complete at {datetime.now()}.")
    except Exception as e:
        logger.exception(f"Error in perform_operations for {ticker}: {e}")
        raise

    if ticker in ["SPY", "TSLA", "GOOG"]:
        try:
            asyncio.create_task(trade_algos2.actions(combined_optionchain_df_withlac_diff, dailyminutes, processeddata, ticker, current_price))
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
                    await asyncio.sleep(0)
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
        #use this to exit all and plose open orders.
        # ibAPI.ib_reset_and_close_pos()
        # ibAPI.util.patchAsyncio()
        asyncio.run(run_program())

    finally:
        ibAPI.ib_disconnect()  # Disconnect at the end of the script
