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
from UTILITIES import check_Market_Conditions, eod_scp_dailyminutes_to_studiopc
from UTILITIES.logger_config import logger
from Strategy_Testing.make_test_df import get_dailyminutes_make_single_multiday_df

client_session = None
market_open_time_utc = None
market_close_time_utc = None
ticker_cycle_running = asyncio.Event()
# flag to track if handle_ticker_cycle() tasks are still runing


# #TODO actions is taking 16 of the 35 seconds.
order_manager = IB.ibAPI.IBOrderManager()

semaphore = asyncio.Semaphore(500)

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
    await main()  # Run main

    if not ib_task.done():
        ib_task.cancel()  # Attempt to cancel the ib_task
        try:
            await ib_task  # Wait for ib_task to exit (it should handle cancellation)
        except asyncio.CancelledError:
            pass  # Handle CancelledError if necessary


async def get_options_data_for_ticker(session, ticker, loop_start_time):
    ticker = ticker.upper()
    try:
        (
            LAC,
            current_price,
            StockLastTradeTime,
            YYMMDD,
        ) = await tradierAPI_marketdata.get_options_data(
            session, ticker, loop_start_time
        )
        # print(f"{ticker} OptionData complete at {datetime.now()}.")t_opti
        return LAC, current_price, StockLastTradeTime, YYMMDD
    except Exception as e:
        print("in getoptionsdata")
        raise


async def calculate_operations(
    session, ticker, LAC, current_price, StockLastTradeTime, YYMMDD, loop_start_time
):
    try:
        (
            optionchain,
            dailyminutes,
            processeddata,
            ticker,
        ) = await calculations.perform_operations(
            session,
            ticker,
            LAC,
            current_price,
            StockLastTradeTime,
            YYMMDD,
            loop_start_time,
        )
        return optionchain, dailyminutes, processeddata, ticker

    except Exception as e:
        print("in calculate ops")

        logger.exception(f"Error in calculate_operations for {ticker}: {e}")
        raise


async def trade_algos(optionchain, dailyminutes, processeddata, ticker, current_price):
    try:
        # asyncio.create_task(trade_algos2.actions(optionchain, dailyminutes, processeddata, ticker, current_price))
        await trade_algos2.actions(
            optionchain, dailyminutes, processeddata, ticker, current_price
        )
        # print(f"{ticker} Actions complete at {datetime.now()}.")
    except Exception as e:
        logger.exception(f"Error in trade_algos for {ticker}: {e}")
        raise


# TODO make it so that pairs are handled on their own?  not usre
async def handle_ticker_cycle(session, ticker):
    global market_close_time_utc
    start_time = datetime.now(pytz.utc)
    max_retries = 1  # Maximum number of retries

    while start_time < market_close_time_utc + timedelta(seconds=0):
        # for i in range (17):
        #     print(i)
        now = datetime.now()
        loop_start_time_est = now.strftime("%y%m%d_%H%M")
        retries = 0
        LAC = None  # Initialize with a default value
        while retries < max_retries:
            try:
                (
                    LAC,
                    CurrentPrice,
                    StockLastTradeTime,
                    YYMMDD,
                ) = await get_options_data_for_ticker(
                    session, ticker, loop_start_time_est
                )
                # print("Fetched options data successfully")#TODO if fetch derror or file exists, start at top of next min.
                break
            except Exception as e:
                logger.info(f"Error in get_options_data_for_ticker for {ticker}: {e}")
                retries += 1
                await asyncio.sleep(20)  # Wait before retrying
                continue  # Skip the rest of the loop and retry
        if LAC is None:
            logger.info("LAC is None for ticker %s after %d retries", ticker, retries)
        if LAC is not None:
            if ticker in [
                "SPY",
                "TSLA",
                "GOOGL",
                "UVXY",
                "ROKU",
                "QQQ",
                "SQQQ",
                "SPXS",
            ]:
                try:
                    (
                        optionchain,
                        dailyminutes,
                        processeddata,
                        ticker,
                    ) = await calculate_operations(
                        session,
                        ticker,
                        LAC,
                        CurrentPrice,
                        StockLastTradeTime,
                        YYMMDD,
                        loop_start_time_est,
                    )
                    # print("Calculated operations successfully")

                except Exception as e:
                    logger.info(f"Error in calculate_operations for {ticker}: {e}")
            if ticker in ["SPY", "TSLA", "GOOGL"]:
                asyncio.create_task(
                    trade_algos(
                        optionchain, dailyminutes, processeddata, ticker, CurrentPrice
                    )
                )
                # await trade_algos(optionchain, dailyminutes, processeddata, ticker, CurrentPrice)

        end_time = datetime.now(pytz.utc)
        elapsed_time = (end_time - start_time).total_seconds()
        record_elapsed_time(ticker, elapsed_time)

        sleep_time = max(0, 60 - elapsed_time)

        await asyncio.sleep(sleep_time)
        start_time = datetime.now(pytz.utc)


def record_elapsed_time(ticker, elapsed_time):
    with open("elapsed_times.txt", "a") as file:
        file.write(f"{ticker} ,{datetime.now().isoformat()},{elapsed_time}\n")


# TODO work out the while loops between main and handle ticker cycle...  i think its redundant ins ome ways..
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
        tasks = []
        # delay_interval = .1
        for i, ticker in enumerate(tickerlist):
            await asyncio.sleep(0.5)  # This will stagger the start times
            tasks.append(asyncio.create_task(handle_ticker_cycle(session, ticker)))
        await asyncio.gather(*tasks)
        print("OVER AT:", datetime.now())

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
        logger.info(f"Main.py began at utc time: {datetime.utcnow()}")
        market_open_time_utc, market_close_time_utc = asyncio.run(
            check_Market_Conditions.get_market_open_close_times()
        )
        asyncio.run(wait_until_time(market_open_time_utc))
        asyncio.run(run_program())
    except KeyboardInterrupt:
        pass
    finally:
        ticker_cycle_running.clear()
        logger.info("Shutting down, waiting for ib_)connect. <5min")
        if client_session is not None:
            asyncio.run(client_session.close())
        if order_manager.ib.isConnected():
            order_manager.ib_disconnect()

        with open("UTILITIES/tickerlist.txt", "r") as f:
            tickerlist = [line.strip().upper() for line in f.readlines()]
            for ticker in tickerlist:
                if ticker == "SPY":
                    try:
                        get_dailyminutes_make_single_multiday_df(ticker)

                    except Exception as e:
                        print(ticker, e)
        ssh_client = eod_scp_dailyminutes_to_studiopc.create_ssh_client(
            "192.168.1.109", 22, "bonsaiheart", "/home/bonsai/.ssh/id_rsa"
        )
        eod_scp_dailyminutes_to_studiopc.scp_transfer_files(
            ssh_client,
            "/home/bonsai/Python_Projects/StockAlgoV2/data/historical_multiday_minute_DF/SPY_historical_multiday_min.csv",
            r"PycharmProjects/StockAlgoV2/data/historical_multiday_minute_DF/SPY_historical_multiday_min.csv",
        )
        ssh_client.close()
    logger.info(f"Main.py ended at utc time: {datetime.utcnow()}")
