import asyncio
import logging
import traceback
from datetime import timedelta
import aiohttp
from datetime import datetime
import asyncio
import pytz  # Make sure to install pytz if you haven't already
import IB.ibAPI
import calculations

# import new_marketdata
import trade_algos2
import tradierAPI_marketdata
# from UTILITIES import check_Market_Conditions, eod_scp_dailyminutes_to_studiopc
from UTILITIES.logger_config import logger
from Strategy_Testing.make_test_df import get_dailyminutes_make_single_multiday_df

client_session = None
market_open_time_utc = None
market_close_time_utc = None
order_manager = IB.ibAPI.IBOrderManager()
semaphore = asyncio.Semaphore(500)
trade_algos_queues = {}  # Dictionary to hold a queue for each ticker
ticker_cycle_running = asyncio.Event()


# def get_next_minute_start():
#     now = datetime.now(pytz.utc)
#     next_minute_start = now + timedelta(seconds=60 - now.second)
#     return next_minute_start
###Currently taking 39,40 seconds.


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
    session,
    ticker,
    LAC,
    current_price,
    StockLastTradeTime,
    YYMMDD,
    loop_start_time,
    optionchaindf,
):
    try:
        (optionchain, dailyminutes, processeddata, ticker) = (
            await calculations.perform_operations(
                session,
                ticker,
                LAC,
                current_price,
                StockLastTradeTime,
                YYMMDD,
                loop_start_time,
                optionchaindf,
            )
        )
        return optionchain, dailyminutes, processeddata, ticker

    except Exception as e:
        print("in calculate ops")

        logger.exception(f"Error in calculate_operations for {ticker}: {e}")
        raise


async def trade_algos(optionchain, dailyminutes, processeddata, ticker, current_price):
    if ticker not in trade_algos_queues:
        trade_algos_queues[ticker] = asyncio.Queue()
    queue_length = trade_algos_queues[ticker].qsize()
    if queue_length == 0:
        await trade_algos_queues[ticker].put(
            (optionchain, dailyminutes, processeddata, ticker, current_price)
        )
        queue_length = trade_algos_queues[ticker].qsize()
        print(f"Added task to {ticker} queue, current len: {queue_length}")


async def process_ticker_queue(ticker):
    print(f"Worker for {ticker} started")
    while True:
        # trade_algos_queues[ticker].task_done()
        queue_length = trade_algos_queues[ticker].qsize()
        print(
            f"Working on task for {ticker} queue, current # of tasks for ticker: {queue_length}"
        )
        # print(f"Worker for {ticker} waiting for task")
        optionchain, dailyminutes, processeddata, ticker, current_price = (
            await trade_algos_queues[ticker].get()
        )
        # print(f"Processing task for {ticker}")
        try:

            trade_algos_queues[ticker].task_done()
            await trade_algos2.actions(
                optionchain, dailyminutes, processeddata, ticker, current_price
            )
            queue_length = trade_algos_queues[ticker].qsize()
            print(f"Finished task for {ticker} queue, current len: {queue_length}")
        except Exception as e:
            logger.warning(f"Error in tradealgo queue for {ticker}: {e}", exc_info=True)
            trade_algos_queues[ticker].task_done()
            queue_length = trade_algos_queues[ticker].qsize()
            print(
                f"Could not complete task for {ticker} queue. {e}.  Task removed. current len: {queue_length}"
            )


async def get_options_data_for_ticker(session, ticker, loop_start_time):
    ticker = ticker.upper()
    try:
        (LAC, current_price, StockLastTradeTime, YYMMDD, optionchaindf) = (
            await tradierAPI_marketdata.get_options_data(
                session, ticker, loop_start_time
            )
        )
        # print(f"{ticker} OptionData complete at {datetime.now()}.")t_opti
        return LAC, current_price, StockLastTradeTime, YYMMDD, optionchaindf
    except Exception as e:
        print("in getoptionsdata")
        raise


tasks = []
TICKERS_FOR_TRADE_ALGOS = ["SPY", "TSLA", "ROKU", "MSFT"]
TICKERS_FOR_CALCULATIONS = [
    "SPY",
    "TSLA",
    "GOOGL",
    "UVXY",
    "ROKU",
    "QQQ",
    "SQQQ",
    "SPXS",
    "MSFT",
]


async def handle_ticker_cycle(session, ticker):
    start_time = datetime.now(pytz.utc)
    global market_close_time_utc
    max_retries = 1

    while True:
        loop_start_time_est = datetime.now().strftime("%y%m%d_%H%M")
        LAC, optionchain, optionchaindf = None, None, None

        try:
            option_data_success = await get_options_data_for_ticker(
                session, ticker, loop_start_time_est
            )
            if ticker in TICKERS_FOR_CALCULATIONS:
                if option_data_success:
                    LAC, CurrentPrice, StockLastTradeTime, YYMMDD, optionchaindf = (
                        option_data_success
                    )
                    calculate_operations_success = await calculate_operations(
                        session,
                        ticker,
                        LAC,
                        CurrentPrice,
                        StockLastTradeTime,
                        YYMMDD,
                        loop_start_time_est,
                        optionchaindf,
                    )

                    if calculate_operations_success:
                        optionchain, dailyminutes, processeddata, ticker = (
                            calculate_operations_success
                        )

                        if (
                            ticker in TICKERS_FOR_TRADE_ALGOS
                            and optionchain is not None
                            and not optionchain.empty
                            and order_manager.ib.isConnected()
                        ):
                            # asyncio.create_task(trade_algos(optionchain, dailyminutes, processeddata, ticker, CurrentPrice))
                            await trade_algos(
                                optionchain,
                                dailyminutes,
                                processeddata,
                                ticker,
                                CurrentPrice,
                            )
        except Exception as e:
            logger.exception(e)
        elapsed_time = (datetime.now(pytz.utc) - start_time).total_seconds()
        print(
            f"Ticker: {ticker}| Elapsed_time: {elapsed_time}| Loop Start: {loop_start_time_est}"
        )
        record_elapsed_time(ticker, elapsed_time)
        if elapsed_time > 60:
            logger.warning(f"{ticker} took {elapsed_time} to complete cycle.")

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
        with open("UTILITIES/tickerlist.txt", "r") as f:
            tickerlist = [line.strip().upper() for line in f.readlines()]

        # Create worker tasks for each ticker
        worker_tasks = []
        for ticker in TICKERS_FOR_TRADE_ALGOS:

            if ticker not in trade_algos_queues:
                trade_algos_queues[ticker] = asyncio.Queue()
            worker_task = asyncio.create_task(process_ticker_queue(ticker))
            worker_tasks.append(worker_task)

        ticker_tasks = []

        delay = len(tickerlist) / 20  # Set your desired delay in seconds
        for ticker in tickerlist:
            task = asyncio.create_task(handle_ticker_cycle(session, ticker))
            ticker_tasks.append(task)
            await asyncio.sleep(
                delay
            )  # Wait for the specified delay before starting next task

        # Wait for all ticker tasks to complete
        await asyncio.gather(*ticker_tasks)

        # Wait for all tasks in trade_algos_queues to be processed
        for queue in trade_algos_queues.values():
            await queue.join()

        # Cancel remaining worker tasks
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
        #     check_Market_Conditions.get_market_open_close_times()
        # )
        # asyncio.run(wait_until_time(market_open_time_utc))
        # logger.info(
        #     f"Main.py started data collection with market open, at utc time: {datetime.utcnow()}"
        # )

        asyncio.run(run_program())
    except KeyboardInterrupt:
        pass
    # finally:
    #     ticker_cycle_running.clear()
    #     logger.info("Shutting down, waiting for ib_)connect. <5min")
    #     if client_session is not None:
    #         asyncio.run(client_session.close())
    #     if order_manager.ib.isConnected():
    #         order_manager.ib_disconnect()
    #
    #     with open("UTILITIES/tickerlist.txt", "r") as f:
    #         tickerlist = [line.strip().upper() for line in f.readlines()]
    #         for ticker in tickerlist:
    #             if ticker == "SPY":
    #                 try:
    #                     get_dailyminutes_make_single_multiday_df(ticker)
    #
    #                 except Exception as e:
    #                     print(ticker, e)
    #     ssh_client = eod_scp_dailyminutes_to_studiopc.create_ssh_client(
    #         "192.168.1.109", 22, "bonsaiheart", "/home/bonsai/.ssh/id_rsa"
    #     )
    #     eod_scp_dailyminutes_to_studiopc.scp_transfer_files(
    #         ssh_client,
    #         "/home/bonsai/Python_Projects/StockAlgoV2/data/historical_multiday_minute_DF/SPY_historical_multiday_min.csv",
    #         r"PycharmProjects/StockAlgoV2/data/historical_multiday_minute_DF/SPY_historical_multiday_min.csv",
    #     )
    #     ssh_client.close()
    #     logger.info(f"Main.py ended at utc time: {datetime.utcnow()}")
