
import cProfile
import logging
from datetime import datetime, timedelta
import asyncio
import os

import traceback
import UTILITIES.check_Market_Conditions
import calculations
# import new_marketdata
import trade_algos
import trade_algos2
from UTILITIES import check_Market_Conditions
import tradierAPI_marketdata
from IB import ibAPI
import aiohttp

from UTILITIES.logger_config import logger
is_market_open = check_Market_Conditions.is_market_open_now()
# is_market_open=True
client_session = None


async def create_client_session():
    global client_session
    if client_session is not None:
        await client_session.close()  # Close the existing session if it's not None
    client_session = aiohttp.ClientSession()
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

async def ib_connect():
    while True:
        try:
            await ibAPI.ib_connect()  # Connect to IB here
            await asyncio.sleep(5 * 60)
            print('running ib_connect_and_main again.')
        except Exception as e:
            # Log the error and continue running
            logger.exception(f"Error in ib_connect: {e}")
async def run_program():
    while True:  # This loop ensures the program continues running even if the session closes
        try:
            await asyncio.gather(ib_connect(), main())
        except Exception as e:
            # Log the error and continue running
            logger.exception(f"Error in run_program: {e}")

semaphore = asyncio.Semaphore(500)
async def get_options_data_for_ticker(session, ticker):

    ticker = ticker.upper()
    try:
        LAC, current_price, price_change_percent, StockLastTradeTime, this_minute_ta_frame, closest_exp_date,YYMMDD = await tradierAPI_marketdata.get_options_data(session, ticker)
        # print(f"{ticker} OptionData complete at {datetime.now()}.")
        return ticker, LAC, current_price, price_change_percent, StockLastTradeTime, this_minute_ta_frame, closest_exp_date,YYMMDD
    except Exception as e:
        logger.exception(f"Error in get_options_data for {ticker}: {e}")
        raise

async def handle_ticker( ticker, LAC, current_price, price_change_percent, StockLastTradeTime, this_minute_ta_frame, closest_exp_date,YYMMDD):
    try:
        (optionchain, dailyminutes, processeddata, ticker) = calculations.perform_operations(
            ticker, LAC, current_price, price_change_percent, StockLastTradeTime, this_minute_ta_frame, closest_exp_date,YYMMDD
        )
        # print(f"{ticker} PerformOptions complete at {datetime.now()}.")
    except Exception as e:
        logger.exception(f"Error in perform_operations for {ticker}: {e}")
        raise


    try:
        asyncio.create_task(trade_algos2.actions(optionchain, dailyminutes, processeddata, ticker, current_price))
        # print(f"{ticker} Actions complete at {datetime.now()}.")
    except Exception as e:
        logger.exception(f"Error in actions for {ticker}: {e}")
        raise

async def handle_ticker_cycle(session, ticker):

    while True:
        start_time = datetime.now()

        try:

            data = await get_options_data_for_ticker(session, ticker)

            if ticker in ["SPY", "TSLA", "GOOGL"]:
                asyncio.create_task(handle_ticker(*data))



        except Exception as e:
            print(f"Error occurred for ticker {ticker}: {traceback.format_exc()}")
            logger.exception(f"Error occurred for ticker {ticker}: {e}")
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        sleep_time = max(0, 60 - elapsed_time)
        await asyncio.sleep(sleep_time)

#TODO work out the while loops between main and handle ticker cycle...  i think its redundant ins ome ways..
async def main():
    while True:
        try:
            await create_client_session()  # Create a new client session
            # Your main program logic here using the new client session
            session = client_session
        except Exception as e:
            logging.exception(f"Error in main: {e}")
        start_time = datetime.now()
        print(start_time)

        try:
            with open("UTILITIES/tickerlist.txt", "r") as f:
                tickerlist = [line.strip().upper() for line in f.readlines()]

            # Compute the delay interval
            tasks=[]
            delay_interval = .1
            for i, ticker in enumerate(tickerlist):

                await asyncio.sleep(1)  # This will stagger the start times
                tasks.append(asyncio.create_task(handle_ticker_cycle(session, ticker)))
            await asyncio.gather(*tasks)




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
        asyncio.run(run_program())
    except KeyboardInterrupt:
        pass
    finally:
        if client_session is not None:
            asyncio.run(client_session.close())
        if ibAPI.ib.isConnected():
            ibAPI.ib_disconnect()  # Disconnect at the end of the script
