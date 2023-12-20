import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import PrivateData.tradier_info
from UTILITIES.logger_config import logger


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
    url = "https://api.tradier.com/v1/markets/options/chains"
    params = {"symbol": ticker, "expiration": exp_date, "greeks": "true"}
    json_response = await fetch(session, url, params, headers)

    if json_response and "options" in json_response and "option" in json_response["options"]:
        optionchain_df = pd.DataFrame(json_response["options"]["option"])
        return optionchain_df
    else:
        print(f"Failed to retrieve option chain data for ticker {ticker}: json_response or required keys are missing")
        return None  # Or another appropriate response to indicate failure



async def get_option_chains_concurrently(session, ticker, expiration_dates, headers):
    tasks = [get_option_chain(session, ticker, exp_date, headers) for exp_date in expiration_dates]
    all_option_chains = await asyncio.gather(*tasks)
    return all_option_chains


async def fetch(session, url, params, headers):
    try:
        async with session.get(url, params=params, headers=headers) as response:
            # print("Rate Limit Headers Allowed:", response.headers.get("X-Ratelimit-Allowed"),"Used:", response.headers.get("X-Ratelimit-Used"))
            return await response.json()
    except Exception as e:
        print(f"Connection error to {url}: {e}.")
        logger.exception(f"An error occurred while fetching data: {e} At URL {url}")



async def get_options_data(session, ticker,YYMMDD_HHMM):
    headers = {f"Authorization": f"Bearer {real_auth}", "Accept": "application/json"}

    tasks = []
    # Add tasks to tasks list


    tasks.append(fetch(session, "https://api.tradier.com/v1/markets/quotes",
                       params={"symbols": ticker, "greeks": "false"}, headers=headers))

    tasks.append(fetch(session, "https://api.tradier.com/v1/markets/options/expirations",
                       params={"symbol": ticker, "includeAllRoots": "true", "strikes": "true"}, headers=headers))

    # Wait for all tasks to complete
    responses = await asyncio.gather(*tasks)
    # Process responses
    quotes_response = responses[0]
    expirations_response = responses[1]



    quote_df = pd.DataFrame.from_dict(quotes_response["quotes"]["quote"], orient="index").T
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

    StockLastTradeTime = datetime.fromtimestamp(StockLastTradeTime).strftime("%y%m%d_%H%M")
    print(f"${ticker} last Trade Time: {StockLastTradeTime_str}")


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
    combined = pd.merge(puts_df, calls_df, on=["ExpDate", "Strike"])
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
    combined["LAC"] = LAC
    combined["CurrentPrice"] = CurrentPrice
    combined["open"] = open
    combined["high"] = high
    combined["low"] = low
    combined["average_volume"] = average_volume
    combined["last_volume"] = last_volume

    combined.rename(columns=rename_dict_combined, inplace=True)
    ####################
    # for option in json_response["options"]["option"]:
    #     print(option["symbol"], option["open_interest"])
    ##weighted total iv for contract
    # Total IV = (bid IV * bid volume + mid IV * mid volume + ask IV * ask volume) / (bid volume + mid volume + ask volume)
    # vega measures response to IV change.

    output_dir = Path(f"data/optionchain/{ticker}/{YYMMDD}")
    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)


    if YYMMDD == StockLastTradeTime_YMD:
         try:
             combined.to_csv(f"data/optionchain/{ticker}/{YYMMDD}/{ticker}_{YYMMDD_HHMM}.csv", mode="x")
             return LAC, CurrentPrice, StockLastTradeTime_str, YYMMDD

         except Exception as e:

             logger.error(f"{e} TIME:{YYMMDD_HHMM}. {ticker} file aready exists using lasttradetime: {YYMMDD_HHMM}, using current YYMMDD_HHMM: {e}")
             return None, None, None, None  # IF its getting outdated info, skip

             # combined.to_csv(f"data/optionchain/{ticker}/{YYMMDD}/{ticker}_{YYMMDD_HHMM}.csv", mode="x")

#TODO should be able to get rid of the returns, ive added lac/currentprice to the csv for longer storatge.  SLTT and YYMMDD are in the filename.


#TODO added open high low close avg vol last vol... but didnt do anything with it yet