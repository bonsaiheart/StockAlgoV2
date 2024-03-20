import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import PrivateData.tradier_info
import aiohttp
import numpy as np
import pandas as pd
import ta
from UTILITIES.logger_config import logger


async def get_ta(session, ticker):
    start = (datetime.today() - timedelta(days=5)).strftime("%Y-%m-%d %H:%M")
    end = datetime.today().strftime("%Y-%m-%d %H:%M")
    headers = {f"Authorization": f"Bearer {real_auth}", "Accept": "application/json"}
    # TODO move this into getoptions data?  then I can run it thru calc ANYTIME b/c it will have all data.  This can be processpooled.
    time_sale_response = await fetch(
        session,
        "https://api.tradier.com/v1/markets/timesales",
        params={
            "symbol": ticker,
            "interval": "1min",
            "start": start,
            "end": end,
            "session_filter": "all",
        },
        headers=headers,
    )

    if (
        time_sale_response
        and "series" in time_sale_response
        and "data" in time_sale_response["series"]
    ):
        df = pd.DataFrame(time_sale_response["series"]["data"]).set_index("time")
    else:
        print(
            f"Failed to retrieve TA data for ticker {ticker}: time_sale_response or required keys are missing or None"
        )
        return None

    df.index = pd.to_datetime(df.index)

    def safe_calculation(df, column_name, calculation_function, *args, **kwargs):
        """
        Safely perform a calculation for a DataFrame and handle exceptions.
        If an exception occurs, the specified column is filled with NaN.
        """
        try:
            df[column_name] = calculation_function(*args, **kwargs)
        except Exception as e:
            logger.warning(
                f"{ticker} - Problem with: column_name={column_name}, function={calculation_function.__name__},error={e}.  This is usually caused by missing data from yfinance."
            )
            df[column_name] = pd.NA

    # Usage of safe_calculation function for each indicator
    safe_calculation(
        df,
        "AwesomeOsc",
        ta.momentum.awesome_oscillator,
        high=df["high"],
        low=df["low"],
        window1=1,
        window2=5,
        fillna=False,
    )
    safe_calculation(
        df,
        "AwesomeOsc5_34",
        ta.momentum.awesome_oscillator,
        high=df["high"],
        low=df["low"],
        window1=5,
        window2=34,
        fillna=False,
    )

    # For MACD
    macd_object = ta.trend.MACD(
        close=df["close"], window_slow=26, window_fast=12, window_sign=9, fillna=False
    )
    signal_line = macd_object.macd_signal
    safe_calculation(df, "MACD", macd_object.macd)
    safe_calculation(df, "Signal_Line", signal_line)

    # For EMAs
    safe_calculation(
        df, "EMA_50", ta.trend.ema_indicator, close=df["close"], window=50, fillna=False
    )
    safe_calculation(
        df,
        "EMA_200",
        ta.trend.ema_indicator,
        close=df["close"],
        window=200,
        fillna=False,
    )

    # For RSI
    safe_calculation(
        df, "RSI1", ta.momentum.rsi, close=df["close"], window=1, fillna=False
    )

    safe_calculation(
        df, "RSI2", ta.momentum.rsi, close=df["close"], window=2, fillna=False
    )
    safe_calculation(
        df, "RSI3", ta.momentum.rsi, close=df["close"], window=3, fillna=False
    )
    safe_calculation(
        df, "RSI4", ta.momentum.rsi, close=df["close"], window=4, fillna=False
    )
    safe_calculation(
        df, "RSI", ta.momentum.rsi, close=df["close"], window=5, fillna=False
    )
    safe_calculation(
        df, "RSI5", ta.momentum.rsi, close=df["close"], window=5, fillna=False
    )  # column "RSI" == RSI5.. I'm trying to swap over but hte old data is still using RSI; so Thsi is to make swap easy. eventually when i ddrop old stuff or mergy? idk its late.
    safe_calculation(
        df, "RSI6", ta.momentum.rsi, close=df["close"], window=6, fillna=False
    )
    safe_calculation(
        df, "RSI7", ta.momentum.rsi, close=df["close"], window=7, fillna=False
    )

    safe_calculation(
        df, "RSI14", ta.momentum.rsi, close=df["close"], window=14, fillna=False
    )

    # Additional Indicators
    safe_calculation(
        df, "SMA_20", ta.trend.sma_indicator, close=df["close"], window=20, fillna=False
    )
    safe_calculation(
        df,
        "ADX",
        ta.trend.adx,
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
        fillna=False,
    )
    safe_calculation(
        df,
        "CCI",
        ta.trend.cci,
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=20,
        fillna=False,
    )
    williams_r_object = ta.momentum.WilliamsRIndicator(
        high=df["high"], low=df["low"], close=df["close"], lbp=14, fillna=False
    )
    safe_calculation(df, "Williams_R", williams_r_object.williams_r)
    pvo_object = ta.momentum.PercentageVolumeOscillator(
        volume=df["volume"], window_slow=26, window_fast=12, window_sign=9, fillna=False
    )
    safe_calculation(df, "PVO", pvo_object.pvo)
    ppo_object = ta.momentum.PercentagePriceOscillator(
        close=df["close"], window_slow=26, window_fast=12, window_sign=9, fillna=False
    )
    safe_calculation(df, "PPO", ppo_object.ppo)

    safe_calculation(
        df,
        "CMF",
        ta.volume.chaikin_money_flow,
        high=df["high"],
        low=df["low"],
        close=df["close"],
        volume=df["volume"],
        window=20,
        fillna=False,
    )
    safe_calculation(
        df,
        "EoM",
        ta.volume.ease_of_movement,
        high=df["high"],
        low=df["low"],
        volume=df["volume"],
        window=14,
        fillna=False,
    )
    safe_calculation(
        df,
        "OBV",
        ta.volume.on_balance_volume,
        close=df["close"],
        volume=df["volume"],
        fillna=False,
    )
    safe_calculation(
        df,
        "MFI",
        ta.volume.money_flow_index,
        high=df["high"],
        low=df["low"],
        close=df["close"],
        volume=df["volume"],
        window=14,
        fillna=False,
    )
    keltner_channel = ta.volatility.KeltnerChannel(
        high=df["high"], low=df["low"], close=df["close"], window=20, window_atr=10
    )
    safe_calculation(
        df,
        "Keltner_Upper",
        keltner_channel.keltner_channel_hband_indicator,
    )
    safe_calculation(
        df,
        "Keltner_Lower",
        keltner_channel.keltner_channel_lband_indicator,
    )
    safe_calculation(
        df,
        "VPT",
        ta.volume.volume_price_trend,
        close=df["close"],
        volume=df["volume"],
        fillna=False,
    )
    # aroon = ta.trend.AroonIndicator(close=df["close"], window=25)
    # safe_calculation(df, "Aroon_Oscillator", aroon.aroon_oscillator, fillna=False)

    groups = df.groupby(df.index.date)
    group_dates = list(groups.groups.keys())
    lastgroup = group_dates[-1]
    ta_data = groups.get_group(lastgroup)
    this_minute_ta_frame = ta_data.tail(1).reset_index(drop=False)
    return this_minute_ta_frame


class OptionChainError(Exception):
    pass


# import webullAPI
# Add a small constant to denominators taper_acc = PrivateData.tradier_info.paper_acc
paper_auth = PrivateData.tradier_info.paper_auth
real_acc = PrivateData.tradier_info.real_acc
real_auth = PrivateData.tradier_info.real_auth

###TODO time and sales (will be used for awesome ind. and ta.
YYMMDD = datetime.today().strftime("%y%m%d")

# TODO for now this ignores the divede by zero warnings.
np.seterr(divide="ignore", invalid="ignore")

sem = asyncio.Semaphore(2)  # Adjust the number as appropriate was10


async def get_option_chain(session, ticker, exp_date, headers):
    url = "https://api.tradier.com/v1/markets/options/chains"
    params = {"symbol": ticker, "expiration": exp_date, "greeks": "true"}
    try:
        async with sem:

            json_response = await fetch(session, url, params, headers)

            if json_response is None:
                raise OptionChainError(
                    f"NONE chain data for {ticker} (in traideierapi.get_options_chain"
                )

            if (
                json_response
                and "options" in json_response
                and "option" in json_response["options"]
            ):
                optionchain_df = pd.DataFrame(json_response["options"]["option"])
                return optionchain_df
            else:
                logger.error(
                    f"Failed to retrieve option chain data for ticker {ticker}: json_response or required keys are missing"
                )
                return None  # Or another appropriate response to indicate failure
    except Exception as e:
        raise


async def get_option_chains_concurrently(session, ticker, expiration_dates, headers):
    try:
        tasks = [
            get_option_chain(session, ticker, exp_date, headers)
            for exp_date in expiration_dates
        ]
        all_option_chains = await asyncio.gather(*tasks)

        # Check if any of the option chains is None and return None immediately
        if any(chain is None for chain in all_option_chains):
            return None

        return all_option_chains
    except Exception as e:
        raise  # Re-raise the exception to the caller


async def fetch(session, url, params, headers):
    rate_limit_allowed, rate_limit_used = None, None
    try:
        timeout = aiohttp.ClientTimeout(total=45)

        async with session.get(
            url, params=params, headers=headers, timeout=timeout
        ) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            rate_limit_allowed = int(response.headers.get("X-Ratelimit-Allowed", "0"))
            rate_limit_used = int(response.headers.get("X-Ratelimit-Used", "0"))
            # print(rate_limit_allowed,rate_limit_used)
            # Check if rate limit used exceeds allowed
            # limit
            if rate_limit_used >= (rate_limit_allowed * 0.99):
                logger.error(
                    f"{url},{params}----Rate limit exceeded: Used {rate_limit_used} out of {rate_limit_allowed}"
                )
                # await asyncio.sleep(5)
            if "application/json" in content_type:
                return await response.json()
            else:
                raise OptionChainError(
                    f"Fetch error: {content_type} with params {params} {url}"
                )
    # except asyncio.TimeoutError as e:
    #     logger.error(
    #         f"{rate_limit_used, rate_limit_allowed}The request timed out, {e} {params, url}"
    #     )
    #     raise OptionChainError(f"Timeout occurred for {params, url}")
    except Exception as e:
        raise OptionChainError(f"Fetch error: {e} with params {params} {url}")


async def get_options_data(session, ticker, YYMMDD_HHMM):
    headers = {f"Authorization": f"Bearer {real_auth}", "Accept": "application/json"}

    tasks = []
    # Add tasks to tasks list

    tasks.append(
        fetch(
            session,
            "https://api.tradier.com/v1/markets/quotes",
            params={"symbols": ticker, "greeks": "false"},
            headers=headers,
        )
    )

    tasks.append(
        fetch(
            session,
            "https://api.tradier.com/v1/markets/options/expirations",
            params={"symbol": ticker, "includeAllRoots": "true", "strikes": "true"},
            headers=headers,
        )
    )

    # # Wait for all tasks to complete
    # responses = await asyncio.gather(*tasks, return_exceptions=False) #was true
    try:
        responses = await asyncio.gather(*tasks)  # Default is return_exceptions=False
    except Exception as e:
        logger.error(f"An error occurred in fetching data: {e}")
        raise
    # # Check for exceptions in responses
    # for response in responses:
    #     if isinstance(response, Exception):
    #         logger.error(f"An error occurred in fetching data: {response}")
    #         raise response

    # Process responses
    quotes_response = responses[0]
    expirations_response = responses[1]

    quote_df = pd.DataFrame.from_dict(
        quotes_response["quotes"]["quote"], orient="index"
    ).T
    LAC = quote_df.at[0, "prevclose"]
    open = quote_df.at[0, "open"]
    # print(open)
    high = quote_df.at[0, "high"]
    low = quote_df.at[0, "low"]
    average_volume = quote_df.at[0, "average_volume"]
    last_volume = quote_df.at[0, "last_volume"]

    # print(high)
    CurrentPrice = quote_df.at[0, "last"]
    # price_change_percent = quote_df["change_percentage"][0]  Assuming this same as lac to current price
    StockLastTradeTime = quote_df["trade_date"][0]
    StockLastTradeTime = StockLastTradeTime / 1000  # Convert milliseconds to seconds
    StockLastTradeTime_datetime = datetime.fromtimestamp(StockLastTradeTime)
    StockLastTradeTime_str = StockLastTradeTime_datetime.strftime("%y%m%d_%H%M")
    StockLastTradeTime_YMD = StockLastTradeTime_datetime.strftime("%y%m%d")

    StockLastTradeTime = datetime.fromtimestamp(StockLastTradeTime).strftime(
        "%y%m%d_%H%M"
    )

    expirations = expirations_response["expirations"]["expiration"]
    expiration_dates = [expiration["date"] for expiration in expirations]

    callsChain = []
    putsChain = []
    try:
        all_option_chains = await get_option_chains_concurrently(
            session, ticker, expiration_dates, headers
        )
    except OptionChainError as e:
        logger.error(e)
        raise  # Re-raise the exception to the caller

    for optionchain_df in all_option_chains:
        if optionchain_df is not None:
            grouped = optionchain_df.groupby("option_type")
            call_group = grouped.get_group("call").copy()
            put_group = grouped.get_group("put").copy()
            callsChain.append(call_group)
            putsChain.append(put_group)

    calls_df = pd.concat(callsChain, ignore_index=True)
    puts_df = pd.concat(putsChain, ignore_index=True)
    # Columns to keep

    # Calculate new columns
    for df in [calls_df, puts_df]:
        df["dollarsFromStrike"] = abs(df["strike"] - LAC)
        df["ExpDate"] = df["symbol"].str[-15:-9]
        df["Strike"] = df["strike"]
        df["dollarsFromStrikeXoi"] = df["dollarsFromStrike"] * df["open_interest"]
        df["lastPriceXoi"] = df["last"] * df["open_interest"]
        # df["impliedVolatility"] = df["greeks"].str.get("mid_iv")
        # df["delta"] = df["greeks"].str.get("delta")
    # calls_df["lastContractPricexOI"] = calls_df["last"] * calls_df["open_interest"]
    # calls_df["impliedVolatility"] = calls_df["greeks"].str.get("mid_iv")
    columns_to_keep = [
        "symbol",
        "trade_date",
        "last",
        "bid",
        "ask",
        "change",
        "change_percentage",
        "volume",
        "open_interest",
        "greeks",
        "delta",
        "ExpDate",
        "Strike",
        "lastPriceXoi",
        "impliedVolatility",
        "dollarsFromStrikeXoi",
    ]
    # TODO commetned this out 240105
    # # Columns to drop (all columns that are not in 'columns_to_keep')
    # columns_to_drop_calls = [
    #     col for col in calls_df.columns if col not in columns_to_keep
    # ]
    # columns_to_drop_puts = [
    #     col for col in puts_df.columns if col not in columns_to_keep
    # ]
    columns_to_drop = [
        "description",
        "type",
        "exch",
        "underlying",
        "bidexch",
        "askexch",
        "expiration_date",
        "root_symbol",
    ]
    # # Drop unnecessary columns
    calls_df = calls_df.drop(columns_to_drop, axis=1)
    puts_df = puts_df.drop(columns_to_drop, axis=1)
    # Format date
    # Rename columns
    rename_dict = {
        "symbol": "contractSymbol",
        "trade_date": "lastTrade",
        "last": "lastPrice",
        "bid": "bid",
        "ask": "ask",
        "change": "change",
        "change_percentage": "percentChange",
        "volume": "volume",
        "open_interest": "openInterest",
        # "delta": "delta",
        "greeks": "greeks",
        # "impliedVolatility": "impliedVolatility",
        "dollarsFromStrike": "dollarsFromStrike",
        "dollarsFromStrikeXoi": "dollarsFromStrikeXoi",
        "lastPriceXoi": "lastPriceXoi",
    }
    # TODO change all columns to use standasrdized.. some are c_ some are Calls_ etc.
    calls_df.rename(columns={k: f"c_{v}" for k, v in rename_dict.items()}, inplace=True)
    puts_df.rename(columns={k: f"p_{v}" for k, v in rename_dict.items()}, inplace=True)

    # Merge dataframes
    combined = pd.merge(puts_df, calls_df, on=["ExpDate", "Strike"])
    # Update renaming dictionary for the combined DataFrame
    rename_dict_combined = {
        "c_lastPrice": "Call_LastPrice",
        "c_percentChange": "Call_PercentChange",
        "c_volume": "Call_Volume",
        "c_openInterest": "Call_OI",
        # "c_impliedVolatility": "Call_IV",
        "c_dollarsFromStrike": "Calls_dollarsFromStrike",
        "c_dollarsFromStrikeXoi": "Calls_dollarsFromStrikeXoi",
        "c_lastPriceXoi": "Calls_lastPriceXoi",
        "p_lastPrice": "Put_LastPrice",
        "p_volume": "Put_Volume",
        "p_openInterest": "Put_OI",
        # "p_impliedVolatility": "Put_IV",
        "p_dollarsFromStrike": "Puts_dollarsFromStrike",
        "p_dollarsFromStrikeXoi": "Puts_dollarsFromStrikeXoi",
        "p_lastPriceXoi": "Puts_lastPriceXoi",
    }
    # combined["LAC"] = LAC
    # combined["CurrentPrice"] = CurrentPrice
    # combined["open"] = open
    # combined["high"] = high
    # combined["low"] = low
    # combined["average_volume"] = average_volume
    # combined["last_volume"] = last_volume
    for column in [
        "LAC",
        "CurrentPrice",
        "open",
        "high",
        "low",
        "average_volume",
        "last_volume",
    ]:
        combined[column] = np.nan

    # Populate values only in the first row
    first_index = combined.index[0]  # Get the first index of the DataFrame
    combined.loc[
        first_index,
        ["LAC", "CurrentPrice", "open", "high", "low", "average_volume", "last_volume"],
    ] = [LAC, CurrentPrice, open, high, low, average_volume, last_volume]

    combined.rename(columns=rename_dict_combined, inplace=True)
    ####################
    # for option in json_response["options"]["option"]:
    #     print(option["symbol"], option["open_interest"])
    ##weighted total iv for contract
    # Total IV = (bid IV * bid volume + mid IV * mid volume + ask IV * ask volume) / (bid volume + mid volume + ask volume)
    # vega measures response to IV change.
    try:
        this_minute_ta_frame = await get_ta(session, ticker)
        for column in this_minute_ta_frame.columns:
            combined[column] = this_minute_ta_frame[column]
    except Exception as e:
        logger.warning(f"{ticker} has an error {e}")

    output_dir = Path(f"data/optionchain/{ticker}/{YYMMDD}")
    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

    # Make sure it's the same day's data.
    # if YYMMDD == StockLastTradeTime_YMD:#TDDO for test mode comment this
    # print(f"{YYMMDD_HHMM}: ${ticker} last Trade Time: {StockLastTradeTime_str}")

    try:
        file_path = f"data/optionchain/{ticker}/{YYMMDD}/{ticker}_{YYMMDD_HHMM}.csv"
        combined.to_csv(file_path, mode="w", index=False)
        return LAC, CurrentPrice, StockLastTradeTime_str, YYMMDD, combined

    except FileExistsError as e:
        logger.error(
            f"File already exists: {e} TIME:{YYMMDD_HHMM}. {ticker} {YYMMDD_HHMM}"
        )
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error: {e} TIME:{YYMMDD_HHMM}. {ticker} {YYMMDD_HHMM}"
        )
        raise
    # else:
    #     logger limit price
    #                     ).warning(
    #         f"{ticker} date:{YYMMDD} is not equal to stocklasttrade date{StockLastTradeTime_YMD}"
    #     )


# TODO should be able to get rid of the returns, ive added lac/currentprice to the csv for longer storatge.  SLTT and YYMMDD are in the filename.


# TODO added open high low close avg vol last vol... but didnt do anything with it yet
