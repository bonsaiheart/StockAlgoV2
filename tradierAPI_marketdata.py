import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import ta

import PrivateData.tradier_info
from UTILITIES.logger_config import logger

concurrency_limit = 500
# import webullAPI
# Add a small constant to denominators taper_acc = PrivateData.tradier_info.paper_acc
paper_auth = PrivateData.tradier_info.paper_auth
real_acc = PrivateData.tradier_info.real_acc
real_auth = PrivateData.tradier_info.real_auth

###TODO time and sales (will be used for awesome ind. and ta.
YYMMDD = datetime.today().strftime("%y%m%d")

# TODO for now this ignores the divede by zero warnings.
np.seterr(divide='ignore', invalid='ignore')


async def get_option_chain(session, ticker, exp_date, headers):
    response = await session.get(
        "https://api.tradier.com/v1/markets/options/chains",
        params={"symbol": ticker, "expiration": exp_date, "greeks": "true"},
        headers=headers,
        timeout=40  # Set the timeout to 10 seconds
    )
    json_response = await response.json()
    # print(response.status_code)
    # print("Option Chain: ",json_response)
    optionchain_df = pd.DataFrame(json_response["options"]["option"])
    return optionchain_df


async def get_option_chains_concurrently(session, ticker, expiration_dates, headers):
    tasks = []
    for exp_date in expiration_dates:
        tasks.append(get_option_chain(session, ticker, exp_date, headers))
    all_option_chains = await asyncio.gather(*tasks)
    return all_option_chains


async def fetch(session, url, params, headers):
    try:
        async with session.get(url, params=params, headers=headers) as response:
            # print("Rate Limit Headers:")
            # print("Allowed:", response.headers.get("X-Ratelimit-Allowed"))
            # print("Used:", response.headers.get("X-Ratelimit-Used"))
            return await response.json()
    except Exception as e:
        print(f"Connection error to {url}: {e}.")
        logger.exception(f"An error occurred while fetching data: {e}")


async def get_options_data(session, ticker):
    start = (datetime.today() - timedelta(days=5)).strftime("%Y-%m-%d %H:%M")
    end = datetime.today().strftime("%Y-%m-%d %H:%M")
    headers = {f"Authorization": f"Bearer {real_auth}", "Accept": "application/json"}

    tasks = []
    # Add tasks to tasks list
    tasks.append(fetch(session, "https://api.tradier.com/v1/markets/timesales",
                       params={"symbol": ticker, "interval": "1min", "start": start, "end": end,
                               "session_filter": "all"}, headers=headers))

    tasks.append(fetch(session, "https://api.tradier.com/v1/markets/quotes",
                       params={"symbols": ticker, "greeks": "false"}, headers=headers))

    tasks.append(fetch(session, "https://api.tradier.com/v1/markets/options/expirations",
                       params={"symbol": ticker, "includeAllRoots": "true", "strikes": "true"}, headers=headers))

    # Wait for all tasks to complete
    responses = await asyncio.gather(*tasks)
    # Process responses
    time_sale_response = responses[0]
    quotes_response = responses[1]
    expirations_response = responses[2]

    json_response = time_sale_response
    # print(response.status_code)
    # print(json_response)
    if json_response and "series" in json_response and "data" in json_response["series"]:
        df = pd.DataFrame(json_response["series"]["data"]).set_index("time")
    else:
        print(
            f"Failed to retrieve options data for ticker {ticker}: json_response or required keys are missing or None")
        return None  # Or another appropriate response to indicate failure
    # df.set_index('time', inplace=True)
    ##change index to datetimeindex
    df.index = pd.to_datetime(df.index)



    def safe_calculation(df, column_name, calculation_function, *args, **kwargs):
        """
        Safely perform a calculation for a DataFrame and handle exceptions.
        If an exception occurs, the specified column is filled with NaN.
        """
        try:
            df[column_name] = calculation_function(*args, **kwargs)
        except Exception:
            df[column_name] = pd.NA  # or pd.nan

    # Usage of safe_calculation function for each indicator
    safe_calculation(df, "AwesomeOsc", ta.momentum.awesome_oscillator, high=df["high"], low=df["low"], window1=1,
                     window2=5, fillna=False)
    safe_calculation(df, "AwesomeOsc5_34", ta.momentum.awesome_oscillator, high=df["high"], low=df["low"], window1=5,
                     window2=34, fillna=False)
    # For MACD
    macd_object = ta.trend.MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9, fillna=False)
    safe_calculation(df, "MACD", macd_object.macd)
    safe_calculation(df, "Signal_Line", macd_object.signal)

    # For EMAs
    safe_calculation(df, "EMA_50", ta.trend.ema_indicator, close=df["close"], window=50, fillna=False)
    safe_calculation(df, "EMA_200", ta.trend.ema_indicator, close=df["close"], window=200, fillna=False)

    # For RSI
    safe_calculation(df, "RSI", ta.momentum.rsi, close=df["close"], window=5, fillna=False)
    safe_calculation(df, "RSI2", ta.momentum.rsi, close=df["close"], window=2, fillna=False)
    safe_calculation(df, "RSI14", ta.momentum.rsi, close=df["close"], window=14, fillna=False)


    groups = df.groupby(df.index.date)
    group_dates = list(groups.groups.keys())
    lastgroup = group_dates[-1]
    ta_data = groups.get_group(lastgroup)
    this_minute_ta_frame = ta_data.tail(1).reset_index(drop=False)

    json_response = quotes_response

    quote_df = pd.DataFrame.from_dict(json_response["quotes"]["quote"], orient="index").T
    LAC = quote_df.at[0, "prevclose"]

    CurrentPrice = quote_df.at[0, "last"]
    price_change_percent = quote_df["change_percentage"][0]
    StockLastTradeTime = quote_df["trade_date"][0]
    StockLastTradeTime = StockLastTradeTime / 1000  # Convert milliseconds to seconds
    StockLastTradeTime = datetime.fromtimestamp(StockLastTradeTime).strftime("%y%m%d_%H%M")
    print(f"${ticker} last Trade Time: {StockLastTradeTime}")


    expirations = expirations_response["expirations"]["expiration"]
    expiration_dates = [expiration["date"] for expiration in expirations]

    callsChain = []
    putsChain = []
    all_option_chains = await get_option_chains_concurrently(session, ticker, expiration_dates, headers)

    for optionchain_df in all_option_chains:
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
        df["impliedVolatility"] = df["greeks"].str.get("mid_iv")
    # calls_df["lastContractPricexOI"] = calls_df["last"] * calls_df["open_interest"]
    # calls_df["impliedVolatility"] = calls_df["greeks"].str.get("mid_iv")
    columns_to_keep = ['symbol', 'trade_date', 'last', 'bid', 'ask', 'change', 'change_percentage', 'volume',
                       'open_interest', 'ExpDate', 'Strike', 'lastPriceXoi', 'impliedVolatility',
                       'dollarsFromStrikeXoi']

    # Columns to drop (all columns that are not in 'columns_to_keep')
    columns_to_drop_calls = [col for col in calls_df.columns if col not in columns_to_keep]
    columns_to_drop_puts = [col for col in puts_df.columns if col not in columns_to_keep]

    # Drop unnecessary columns
    calls_df = calls_df.drop(columns_to_drop_calls, axis=1)
    puts_df = puts_df.drop(columns_to_drop_puts, axis=1)
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
        "greeks": "greeks",
        "impliedVolatility": "impliedVolatility",
        "dollarsFromStrike": "dollarsFromStrike",

        "dollarsFromStrikeXoi": "dollarsFromStrikeXoi",
        "lastPriceXoi": "lastPriceXoi",
    }

    calls_df.rename(columns={k: f"c_{v}" for k, v in rename_dict.items()}, inplace=True)
    puts_df.rename(columns={k: f"p_{v}" for k, v in rename_dict.items()}, inplace=True)

    # Merge dataframes
    combined_optionchain_df = pd.merge(puts_df, calls_df, on=["ExpDate", "Strike"])
    # Update renaming dictionary for the combined DataFrame
    rename_dict_combined = {
        "c_lastPrice": "Call_LastPrice",
        "c_percentChange": "Call_PercentChange",
        "c_volume": "Call_Volume",
        "c_openInterest": "Call_OI",
        "c_impliedVolatility": "Call_IV",
        "c_dollarsFromStrike": "Calls_dollarsFromStrike",
        "c_dollarsFromStrikeXoi": "Calls_dollarsFromStrikeXoi",
        "c_lastPriceXoi": "Calls_lastPriceXoi",
        "p_lastPrice": "Put_LastPrice",
        "p_volume": "Put_Volume",
        "p_openInterest": "Put_OI",
        "p_impliedVolatility": "Put_IV",
        "p_dollarsFromStrike": "Puts_dollarsFromStrike",
        "p_dollarsFromStrikeXoi": "Puts_dollarsFromStrikeXoi",
        "p_lastPriceXoi": "Puts_lastPriceXoi",
    }

    combined_optionchain_df.rename(columns=rename_dict_combined, inplace=True)

    ####################
    # for option in json_response["options"]["option"]:
    #     print(option["symbol"], option["open_interest"])
    ##weighted total iv for contract
    # Total IV = (bid IV * bid volume + mid IV * mid volume + ask IV * ask volume) / (bid volume + mid volume + ask volume)
    # vega measures response to IV change.

    output_dir = Path(f"data/optionchain/{ticker}/{YYMMDD}")
    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    try:
        combined_optionchain_df.to_csv(f"data/optionchain/{ticker}/{YYMMDD}/{ticker}_{StockLastTradeTime}.csv", mode="x")
    except Exception as e:
        if FileExistsError:
            if StockLastTradeTime == 1600:
                combined_optionchain_df.to_csv(f"data/optionchain/{ticker}/{YYMMDD}/{ticker}_{StockLastTradeTime}(2).csv")
        else:
            print(f"An error occurred while writing the CSV file,: {e}")
            combined_optionchain_df.to_csv(f"data/optionchain/{ticker}/{YYMMDD}/{ticker}_{StockLastTradeTime}(2).csv")
    # combined.to_csv(f"combined_tradier.csv")
    ###strike, exp, call last price, call oi, iv,vol, $ from strike, dollars from strike x OI, last price x OI

    return LAC, CurrentPrice, price_change_percent, StockLastTradeTime, this_minute_ta_frame, combined_optionchain_df,YYMMDD
