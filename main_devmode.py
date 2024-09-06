import asyncio
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
import aiohttp
import pytz  # Make sure to install pytz if you haven't already
import IB.ibAPI
import calculations
import database_operations
import trade_algos2
import tradierAPI_marketdata
from UTILITIES.logger_config import logger
from PrivateData import sql_db
from sqlalchemy.orm import Session  # Import Session
from sqlalchemy import create_engine
import analysis_functions
import UTILITIES.check_Market_Conditions
client_session = None
market_open_time_utc = None
market_close_time_utc = None
order_manager = IB.ibAPI.IBOrderManager()
semaphore = asyncio.Semaphore(500)

DATABASE_URI = sql_db.LOCAL_DATABASE_URI
# Create a synchronous engine using create_engine
engine = create_engine(DATABASE_URI, echo=False, pool_size=50, max_overflow=100)
ticker_cycle_running = asyncio.Event()
trade_algos_queues = {}  # Dictionary to hold a queue for each ticker


async def wait_until_time(target_time_utc):
    utc_now = datetime.now(pytz.utc)  # Make utc_now timezone-aware
    time_difference = target_time_utc - utc_now  # Now both are timezone-aware
    print(f"Time Till Open: {time_difference}")
    # Sleep until 0 seconds before target time
    sleep_duration_seconds = max(0, time_difference.total_seconds() - 0)
    await asyncio.sleep(sleep_duration_seconds)


async def create_client_session():
    global client_session
    if client_session is not None:
        await client_session.close()  # Close the existing session if it's not None
    client_session = aiohttp.ClientSession()


async def ib_connect():
    while ticker_cycle_running.is_set():
        try:
            await order_manager.ib_connect()  # Connect to IB here
            await asyncio.sleep(5 * 60)
            print("running ib_connect_and_main again.")
        except Exception as e:
            # Log the error and continue running
            logger.exception(f"Error in ib_connect: {e}")


async def run_program():
    ticker_cycle_running.set()
    ib_task = asyncio.create_task(ib_connect())  # Start ib_connect as a separate task
    # time.sleep(3)
    await main()  # Run main

    if not ib_task.done():
        ib_task.cancel()  # Attempt to cancel the ib_task
        try:
            await ib_task  # Wait for ib_task to exit (it should handle cancellation)
        except asyncio.CancelledError:
            pass  # Handle CancelledError if necessary

async def calculate_operations(
    ticker,
    LAC,
    current_price,
    current_time,
    optionchain_df,
  symbol_name
):
    try:
        loop = asyncio.get_running_loop()
        args = (
            ticker,
            LAC,
            current_price,
            current_time,
            optionchain_df,
            symbol_name
        )

        with ProcessPoolExecutor() as pool:
            func = partial(calculations.perform_operations, *args)
            optionchain, dailyminutes, processeddata, ticker = (
                await loop.run_in_executor(pool, func)
            )

        return optionchain, dailyminutes, processeddata, ticker

    except Exception as e:
        logger.exception(f"Error in calculate_operations for {ticker}: {e}")

        raise


async def trade_algos(
    optionchain, processeddata, ticker, current_price, current_time
):
    if ticker not in trade_algos_queues:
        trade_algos_queues[ticker] = asyncio.Queue()
    queue_length = trade_algos_queues[ticker].qsize()

    if queue_length == 0:
        await trade_algos_queues[ticker].put(
            (
                optionchain,
                processeddata,
                ticker,
                current_price,
                current_time,
            )
        )
        queue_length = trade_algos_queues[ticker].qsize()
        print(f"Added task to {ticker} queue, current len: {queue_length}")


async def process_ticker_queue(ticker):
    print(f"Worker for {ticker} started")
    while True:
        # trade_algos_queues[ticker].task_done()

        # print(f"Worker for {ticker} waiting for task")
        (
            optionchain,

            processeddata,
            ticker,
            current_price,
            current_time,
        ) = await trade_algos_queues[ticker].get()
        # print(f"Processing task for {ticker}")
        try:
#TODO need to change the algo stuff to use new formatted dfs.
            await trade_algos2.actions(
                optionchain,

                processeddata,
                ticker,
                current_price,
                current_time,
            )
            trade_algos_queues[ticker].task_done()
            queue_length = trade_algos_queues[ticker].qsize()
            print(f"Finished task for {ticker} queue, current len: {queue_length}")
        except Exception as e:
            logger.warning(f"Error in tradealgo queue for {ticker}: {e}", exc_info=True)
            trade_algos_queues[ticker].task_done()
            queue_length = trade_algos_queues[ticker].qsize()
            print(
                f"Could not complete task for {ticker} queue. {e}.  Task removed. current len: {queue_length}"
            )



async def handle_ticker_cycle(client_session, ticker):

    start_time = datetime.now(pytz.utc)
    global market_close_time_utc, db_session
    max_retries = 1
    while True:
    # while start_time <= market_close_time_utc:
        current_time = datetime.now()
        # loop_start_time_est = current_time.strftime("%y%m%d_%H%M")
        loop_start_time_w_seconds_est = current_time.strftime("%y%m%d_%H:%M:%S")

        try:
            # Create a new database session for each ticker
            with Session(engine) as session:  # Use the Session object directly to create a session

                option_data_success = await tradierAPI_marketdata.get_options_data(
                    session, client_session, ticker, current_time
                )
                # After data insertion, calculate the ratio
                # in_the_money_ratio = await analysis_functions.calculate_in_the_money_pcr(session, ticker, current_time)
                # net_iv = await analysis_functions.calculate_net_iv(session, ticker, current_time)
                # max_pain = await analysis_functions.calculate_maximum_pain(session, ticker, current_time)
                # # If ratio is above 1, return the ticker
                # print(ticker,net_iv, max_pain)
                # if in_the_money_ratio > 1:
                #     print(ticker,in_the_money_ratio)
                #     # return ticker  # Or you can return more data as needed

                # if ticker in TICKERS_FOR_CALCULATIONS:
            #     if option_data_success:
            #         LAC, CurrentPrice, optionchaindf, symbol_name = option_data_success
            #
            #         calculate_operations_success = await calculate_operations(
            #             ticker, LAC, CurrentPrice, current_time, optionchaindf, symbol_name
            #         )
            # # #     # print(calculate_operations_success)
            #         if calculate_operations_success:
            #             optionchain_df, processed_data_df, ticker, data_to_insert =  calculate_operations_success
            #                         # # Insert data into the database
            #
            #             calculated_data_insert_success =  tradierAPI_marketdata.insert_calculated_data(ticker,engine,data_to_insert)
            #
            #
            #                         # print(calculated_data_insert_success)
            #                 #
            #             if (
            #                 ticker in TICKERS_FOR_TRADE_ALGOS
            #                 and optionchain_df is not None
            #                 and not optionchain_df.empty
            #                 and order_manager.ib.isConnected()
            #             ):
            #                 print("ordermanager connected. ubt not doing trade algos")
                                    # asyncio.create_task(
                                    #     trade_algos(
                                    #         optionchain_df,
                                    #         processed_data_df,
                                    #         ticker,
                                    #         CurrentPrice,
                                    #         current_time,
                                    #     )
                                    # )
                            # await trade_algos(
                            #         optionchain,
                            #         dailyminutes,
                            #         processeddata,
                            #         ticker,
                            #         CurrentPrice,
                            #         current_time,
                            #     )


        except Exception as e:
            # if db_session.is_active:
            #     await db_session.rollback()
            logger.exception(f"Error fetching options data for {ticker}: {e}")  # Log ticker and error


        elapsed_time = (datetime.now(pytz.utc) - start_time).total_seconds()
        print(
            f"Ticker: {ticker}| Elapsed_time: {elapsed_time}| Loop Start: {loop_start_time_w_seconds_est}"
        )
        record_elapsed_time(ticker, elapsed_time)
        if elapsed_time > 60:
            logger.warning(f"{ticker} took {elapsed_time} to complete cycle.")
            exit()
        await asyncio.sleep(max(0, 60 - elapsed_time))
        start_time = datetime.now(pytz.utc)


# Separate functions for attempt_get_options_data, attempt_calculate_operations, and attempt_trade_algos
# These functions will contain the respective logic and error handling for each operation


def record_elapsed_time(ticker, elapsed_time):
    with open("elapsed_times.txt", "a") as file:
        file.write(f"{ticker} ,{datetime.now().isoformat()},{elapsed_time}\n")


# TODO work out the while loops between main and handle ticker cycle...  i think its redundant ins ome ways..


async def main():
    try:
        await create_client_session()
        session = client_session
        database_operations.create_schema_and_tables(engine)
        with open("UTILITIES/tickerlist.txt", "r") as f:
            tickerlist = [line.strip().upper() for line in f.readlines()]

        # Create worker tasks for each ticker
        worker_tasks = []
        # for ticker in TICKERS_FOR_TRADE_ALGOS:
        for ticker in tickerlist:
            if ticker not in trade_algos_queues:
                trade_algos_queues[ticker] = asyncio.Queue()
            worker_task = asyncio.create_task(process_ticker_queue(ticker))
            worker_tasks.append(worker_task)

        ticker_tasks = []

        delay = 60/len(tickerlist)   # Set your desired delay in seconds
        for ticker in tickerlist:
            task = asyncio.create_task(handle_ticker_cycle(session, ticker))
            ticker_tasks.append(task)
            #TODO delay or nay?
            await asyncio.sleep(
                delay
            )  # Wait for the specified delay before starting next task

        # Wait for all ticker tasks to complete
        await asyncio.gather(*ticker_tasks)

        # Wait for all tasks in trade_algos_queues to be processed
        await asyncio.gather(*(queue.join() for queue in trade_algos_queues.values()))

        # Cancel worker tasks
        for worker_task in worker_tasks:
            worker_task.cancel()

        print("OVER AT:", datetime.now())
    except Exception as e:
        print(f"Error occurred in aio session: {traceback.format_exc()}")
        logger.exception(f"Error occurred in aio session: {e}")


if __name__ == "__main__":
    try:
        # logger.info(f"Main.py began at utc time: {datetime.utcnow()}")
        # market_open_time_utc, market_close_time_utc = asyncio.run(
        #     UTILITIES.check_Market_Conditions.get_market_open_close_times()
        # )
        # if market_open_time_utc == None and market_close_time_utc == None:
        #     logger.info(f"Market is not open today.")
        #     exit()
        # asyncio.run(wait_until_time(market_open_time_utc))
        logger.info(
            f"Main_devmode.py started data collection with market open, at utc time: {datetime.utcnow()}"
        )

        asyncio.run(run_program())
    except KeyboardInterrupt:
        pass
    finally:
        if market_open_time_utc == None and market_close_time_utc == None:
            logger.info(f"Market is not open today.")
            exit()
        ticker_cycle_running.clear()
        logger.info("Shutting down, waiting for ib_)connect. <5min")
        if client_session is not None:
            asyncio.run(client_session.close())
        if order_manager.ib.isConnected():
            order_manager.ib_disconnect()


