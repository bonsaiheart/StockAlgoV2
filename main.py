
import cProfile
import logging
from datetime import datetime, timedelta
import asyncio
import os
import time
import traceback

import pytz

import IB.ibAPI
import UTILITIES.check_Market_Conditions
import calculations
# import new_marketdata
import trade_algos2
from UTILITIES import check_Market_Conditions
import tradierAPI_marketdata
from IB import ibAPI
import aiohttp
from UTILITIES.logger_config import logger

client_session = None
market_open_time_utc = None
market_close_time_utc = None
# Flag to track if handle_ticker_cycle tasks are running

ticker_cycle_running = asyncio.Event()

# Global task list to keep track of all created tasks
# all_tasks = []
#
# # Modified function to add created tasks to the global list
# async def calculate_operations(session, ticker, LAC, current_price, StockLastTradeTime, YYMMDD, current_time):
#     # existing code...
#     new_task = asyncio.create_task(trade_algos(optionchain, dailyminutes, processeddata, ticker, current_price))
#     all_tasks.append(new_task)
#     # rest of the function...
#
#
# # Similarly, modify other functions like trade_algos to add created tasks to all_tasks
#
# # Modified main function
# async def main():
#     global all_tasks
#     # existing code...
#     try:
#         # existing code for initializing tasks
#         for i, ticker in enumerate(tickerlist):
#             await asyncio.sleep(.1)
#             new_task = asyncio.create_task(handle_ticker_cycle(session, ticker))
#             all_tasks.append(new_task)
#
#         # Wait for all initialized tasks to complete
#         await asyncio.gather(*all_tasks)
#         print("All tasks completed. Exiting at:", datetime.now())
#     except Exception as e:
#         print(f"Error occurred: {traceback.format_exc()}")
#
# # Main program entry point
# if __name__ == "__main__":
#     try:
#         asyncio.run(run_program())
#     except KeyboardInterrupt:
#         pass
#     finally:
#         if client_session is not None:
#             asyncio.run(client_session.close())
#         if ibAPI.ib.isConnected():
#             ibAPI.ib_disconnect()
#
# # Ensure to modify run_program to also wait for all tasks
# async def run_program():
#     global all_tasks
#     try:
#         await asyncio.gather(ib_connect(), main())
#         # After main tasks are done, wait for any remaining tasks
#         await asyncio.gather(*all_tasks)
#     except Exception as e:
#         logger.exception(f"Error in run_program: {e}")
# ``````````````````````````````````



async def wait_until_time(target_time_utc):
    utc_now = datetime.utcnow()
    now_unix_time = int(utc_now.timestamp())
    target_time_unix_time = int(target_time_utc.timestamp())
    time_difference = target_time_unix_time - now_unix_time

    # Uncomment for debugging
    # print(f"Current Unix Time: {now_unix_time}, Target Unix Time: {target_time_unix_time}")
    print(f"Time Difference (seconds): {time_difference}")

    # Sleep until 20 seconds before target time
    time.sleep(max(0, time_difference-1))


async def create_client_session():
    global client_session
    if client_session is not None:
        await client_session.close()  # Close the existing session if it's not None
    client_session = aiohttp.ClientSession()
# async def profiled_actions(optionchain, dailyminutes, processeddata, ticker, current_price):
#     pr = cProfile.Profile()
#     pr.enable()
#
#     try:
#         await trade_algos.actions22(optionchain, dailyminutes, processeddata, ticker, current_price)
#     except Exception as e:
#         print(f"Error occurred: {traceback.format_exc()}")
#
#     pr.disable()
#     pr.print_stats()
# #TODO actions is taking 16 of the 35 seconds.

async def ib_connect():
    while True:
        while ticker_cycle_running.is_set():

            try:
                order_manager = IB.ibAPI.IBOrderManager()

                await order_manager.ib_connect()  # Connect to IB here
                await asyncio.sleep(5 * 60)
                print('running ib_connect_and_main again.')
            except Exception as e:
                # Log the error and continue running
                logger.exception(f"Error in ib_connect: {e}")
        else:
            break
async def run_program():
    while True:  # This loop ensures the program continues running even if the session closes
        try:
            await asyncio.gather(ib_connect(), main())
            break
        except Exception as e:
            # Log the error and continue running
            logger.exception(f"Error in run_program: {e}")
            break
semaphore = asyncio.Semaphore(500)
async def get_options_data_for_ticker(session, ticker,loop_start_time):

    ticker = ticker.upper()
    try:
        LAC, current_price,  StockLastTradeTime,  YYMMDD = await tradierAPI_marketdata.get_options_data(session, ticker,loop_start_time)
        # print(f"{ticker} OptionData complete at {datetime.now()}.")
        return LAC, current_price,  StockLastTradeTime,  YYMMDD
    except Exception as e:
        logger.exception(f"Error in get_options_data for {ticker}: {e}")
        raise

async def calculate_operations( session,ticker, LAC, current_price, StockLastTradeTime, YYMMDD,current_time):
    try:
        (optionchain, dailyminutes, processeddata, ticker) = await calculations.perform_operations(session,
            ticker, LAC, current_price, StockLastTradeTime, YYMMDD,current_time
        )

        # print(f"{ticker} PerformOptions complete at {datetime.now()}.")
    except Exception as e:
        logger.exception(f"Error in perform_operations for {ticker}: {e}")
        raise
    if ticker in ["SPY", "TSLA", "GOOGL","CHWY","ROKU","V"]:
    # asyncio.create_task(trade_algos(optionchain, dailyminutes, processeddata, ticker, current_price))
    # new_task=asyncio.create_task(trade_algos(optionchain, dailyminutes, processeddata, ticker, current_price))
    # all_tasks.append(new_task)
        asyncio.create_task(trade_algos(optionchain, dailyminutes, processeddata, ticker, current_price))
    return optionchain, dailyminutes, processeddata, ticker

async def trade_algos( optionchain, dailyminutes, processeddata, ticker, current_price):

    try:
        # asyncio.create_task(trade_algos2.actions(optionchain, dailyminutes, processeddata, ticker, current_price))
        await trade_algos2.actions(optionchain, dailyminutes, processeddata, ticker, current_price)
        # print(f"{ticker} Actions complete at {datetime.now()}.")
    except Exception as e:
        logger.exception(f"Error in actions for {ticker}: {e}")
        raise
###Currently taking 39,40 seconds.
async def handle_ticker_cycle(session, ticker):
    global market_close_time_utc
    start_time = datetime.now(pytz.utc)

    while start_time < market_close_time_utc+ timedelta(seconds=0):
    # while True:
        try:
            now = datetime.now()
            loop_start_time_est = now.strftime("%y%m%d_%H%M")
            LAC, CurrentPrice,  StockLastTradeTime, YYMMDD = await get_options_data_for_ticker(session, ticker,loop_start_time_est)
            if LAC ==None or CurrentPrice == None or StockLastTradeTime ==None or YYMMDD == None:
                # end_time = datetime.now(pytz.utc)
                # elapsed_time = (end_time - start_time).total_seconds()
                # sleep_time = max(0, 60 - elapsed_time)
                logger.exception(
                    f"time{now}.  lac: {LAC}, current price: {CurrentPrice}, stock last trade time: {StockLastTradeTime}, yymmd: {YYMMDD}")
                # break
                # await asyncio.sleep(sleep_time)
                # start_time = datetime.now(pytz.utc)
            else:
                # asyncio.create_task(calculate_operations(session,ticker, LAC, CurrentPrice, StockLastTradeTime, YYMMDD,loop_start_time_est))
                await calculate_operations(session,ticker, LAC, CurrentPrice, StockLastTradeTime, YYMMDD,loop_start_time_est)


        except Exception as e:
            print(f"Error occurred for ticker {ticker}: {traceback.format_exc()}")
            logger.exception(f"Error occurred for ticker {ticker}: {e}")

        end_time = datetime.now(pytz.utc)
        elapsed_time = (end_time - start_time).total_seconds()
        record_elapsed_time(ticker, elapsed_time)

        sleep_time = max(0, 60 - elapsed_time)

        await asyncio.sleep(sleep_time)
        start_time = datetime.now(pytz.utc)


    # Clear the flag to indicate that this task has finished
    ticker_cycle_running.clear()


def record_elapsed_time( ticker, elapsed_time):
    with open("elapsed_times.txt", "a") as file:
        file.write(f"{ticker} ,{datetime.now().isoformat()},{elapsed_time}\n")

#TODO work out the while loops between main and handle ticker cycle...  i think its redundant ins ome ways..
async def main():



    try:
        await create_client_session()  # Create a new client session
        # Your main program logic here using the new client session
        session = client_session
    except Exception as e:
        logging.exception(f"Error in main: {e}")
    try:
        with open("UTILITIES/tickerlist.txt", "r") as f:
            tickerlist = [line.strip().upper() for line in f.readlines()]
         # Compute the delay interval
        tasks=[]
        # delay_interval = .1
        for i, ticker in enumerate(tickerlist):

            await asyncio.sleep(.1)  # This will stagger the start times
            tasks.append(asyncio.create_task(handle_ticker_cycle(session, ticker)))
        await asyncio.gather(*tasks)
        print("OVER AT:",datetime.now())
        # exit()



    except Exception as e:
        print(f"Error occurred in aio session: {traceback.format_exc()}")
        logger.exception(f"Error occurred in aio session: {e}")

        # current_time = datetime.now()
        # next_iteration_time = start_time + timedelta(seconds=60)
        # _60sec_countdown = (next_iteration_time - current_time).total_seconds()
        # print(start_time, next_iteration_time, current_time, "Time Remaining in Loop:", _60sec_countdown,
        #       "~~~~~~~~")
        ###TODO I wanted this to tell me how long each iteration is taking (all tickers)
if __name__ == "__main__":
    try:

        market_open_time_utc, market_close_time_utc= asyncio.run(check_Market_Conditions.get_market_open_close_times())
        asyncio.run(wait_until_time(market_open_time_utc))
        logger.info(f"Main.py began at utc time: {datetime.utcnow()}")
        # Set the flag to indicate that this task is running
        ticker_cycle_running.set()

        asyncio.run(run_program())
    except KeyboardInterrupt:
        pass
    finally:
        if client_session is not None:
            asyncio.run(client_session.close())  # Await the session closure
        if ibAPI.ib.isConnected():
            ibAPI.ib_disconnect()  # Disconnect at the end of the script
        logger.info(f"Main.py ended at utc time: {datetime.utcnow()}")

