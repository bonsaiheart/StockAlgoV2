import asyncio
import traceback
from datetime import datetime, timedelta
from pathlib import Path
import PrivateData.tradier_info
import aiohttp
import numpy as np
import pandas as pd
import ta
from ta import momentum,trend,volume,volatility
from UTILITIES.logger_config import logger

# Mapping of intervals to the corresponding number of days to fetch data for


# Define your get_ta function
async def get_ta(session, ticker):
    days_to_fetch = {
        "1min": 5,
        "5min": 20,
        "15min": 40,
    }

    def safe_calculation(df, column_name, calculation_function, *args, **kwargs):
        try:
           df[column_name] = calculation_function(*args, **kwargs)
        except Exception as e:
            logger.warning(
                f"{ticker} - Problem with: column_name={column_name}, function={calculation_function.__name__}, error={e}. This is usually caused by missing data from yfinance."
            )
            df[column_name] = np.nan

    async def fetch_and_process_data(interval):
        start = (datetime.today() - timedelta(days=days_to_fetch[interval])).strftime("%Y-%m-%d %H:%M")
        end = datetime.today().strftime("%Y-%m-%d %H:%M")

        headers = {"Authorization": f"Bearer {real_auth}", "Accept": "application/json"}
        time_sale_response = await fetch(
            session,
            "https://api.tradier.com/v1/markets/timesales",
            params={
                "symbol": ticker,
                "interval": interval,
                "start": start,
                "end": end,
                "session_filter": "all",
            },
            headers=headers,
        )

        if not time_sale_response or "series" not in time_sale_response or "data" not in time_sale_response["series"]:
            logger.warning(f"Failed to retrieve TA data for ticker {ticker} and interval {interval}")
            return pd.DataFrame()

        df = pd.DataFrame(time_sale_response["series"]["data"]).set_index("time")
        df.index = pd.to_datetime(df.index)
        latest_minute_data = df.tail(1).copy()

        # Define the calculations
        def perform_calculations():
            ema_windows = [5, 14, 20, 50, 200]
            rsi_windows = [2, 7, 14, 21]
            macd_windows = [(12, 26, 9)]
            cci_window = 20
            adx_window = 7
            williams_r_params = 14
            cmf_window = 20
            eom_window = 14
            mfi_window = 14
            keltner_window = 20
            keltner_atr_window = 10
            for window in ema_windows:
                safe_calculation(df, f"EMA_{window}_{interval}", ta.trend.ema_indicator, close=df["close"], window=window)

            for window in rsi_windows:
                safe_calculation(df, f"RSI_{window}_{interval}", ta.momentum.rsi, close=df["close"], window=window)


            for fast_window, slow_window, signal_window in macd_windows:
                macd_object = ta.trend.MACD(
                    close=df["close"],
                    window_slow=slow_window,
                    window_fast=fast_window,
                    window_sign=signal_window,
                    fillna=False
                )
                latest_minute_data[f"MACD_{fast_window}_{slow_window}"] = macd_object.macd().iloc[-1]
                latest_minute_data[f"Signal_Line_{fast_window}_{slow_window}"] = macd_object.macd_signal().iloc[-1]
                latest_minute_data[f"MACD_diff_{fast_window}_{slow_window}"] = macd_object.macd_diff().iloc[-1]
                latest_minute_data[f"MACD_diff_prev_{fast_window}_{slow_window}"] = macd_object.macd_diff().shift(1).iloc[-1]

                bullish_cross = (
                    (latest_minute_data[f"MACD_diff_{fast_window}_{slow_window}"] > 0) &
                    (latest_minute_data[f"MACD_diff_prev_{fast_window}_{slow_window}"] <= 0)
                )

                bearish_cross = (
                    (latest_minute_data[f"MACD_diff_{fast_window}_{slow_window}"] < 0) &
                    (latest_minute_data[f"MACD_diff_prev_{fast_window}_{slow_window}"] >= 0)
                )

                latest_minute_data[f"MACD_signal_{fast_window}_{slow_window}"] = np.where(
                    bullish_cross, "buy", np.where(bearish_cross, "sell", "hold")
                )

            safe_calculation(df, f"AwesomeOsc", ta.momentum.awesome_oscillator, high=df["high"], low=df["low"], window1=5, window2=34)
            latest_minute_data[f"AwesomeOsc"] = df[f"AwesomeOsc"].iloc[-1]
            safe_calculation(df, f"SMA_20", ta.trend.sma_indicator, close=df["close"], window=20)
            latest_minute_data[f"SMA_20"] = df[f"SMA_20"].iloc[-1]

            # ADX Calculation (handle potential insufficient data)
            if len(df) >= adx_window:
                safe_calculation(df, f"ADX", ta.trend.adx, high=df["high"], low=df["low"], close=df["close"], window=adx_window, fillna=False)
                latest_minute_data[f"ADX"] = df[f"ADX"].iloc[-1]
            else:
                print("Insufficient data for ADX calculation.")
                latest_minute_data[f"ADX"] = pd.NA  # or any other default value

            safe_calculation(df, f"CCI", ta.trend.cci, high=df["high"], low=df["low"], close=df["close"], window=cci_window)
            latest_minute_data[f"CCI"] = df[f"CCI"].iloc[-1]

            williams_r_object = ta.momentum.WilliamsRIndicator(high=df["high"], low=df["low"], close=df["close"], lbp=williams_r_params,fillna=False)
            # print(williams_r_object.williams_r()) NEED TO ALLIGN THE TIMESTAMP FOR EACH
            latest_minute_data[f"Williams_R"] = williams_r_object.williams_r().iloc[-1]


            # PVO Calculation (using the Williams %R-like approach)
            pvo_object = ta.momentum.PercentageVolumeOscillator(
                volume=df["volume"], window_slow=26, window_fast=12, window_sign=9, fillna=False
            )
            latest_minute_data[f"PVO"] = pvo_object.pvo().iloc[-1]

            # PPO Calculation (using the Williams %R-like approach)
            ppo_object = ta.momentum.PercentagePriceOscillator(
                close=df["close"], window_slow=26, window_fast=12, window_sign=9, fillna=False
            )
            latest_minute_data[f"PPO"] = ppo_object.ppo().iloc[-1]


            safe_calculation(df, f"CMF", ta.volume.chaikin_money_flow, high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=cmf_window)
            latest_minute_data[f"CMF"] = df[f"CMF"].iloc[-1]

            safe_calculation(df, f"EoM", ta.volume.ease_of_movement, high=df["high"], low=df["low"], volume=df["volume"], window=eom_window)
            latest_minute_data[f"EoM"] = df[f"EoM"].mean()  # Use mean to get single value

            safe_calculation(df, f"OBV", ta.volume.on_balance_volume, close=df["close"], volume=df["volume"])
            latest_minute_data[f"OBV"] = df[f"OBV"].iloc[-1]

            safe_calculation(df, f"MFI", ta.volume.money_flow_index, high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=mfi_window)
            latest_minute_data[f"MFI"] = df[f"MFI"].iloc[-1]


            keltner_channel = ta.volatility.KeltnerChannel(
                high=latest_minute_data["high"],
                low=latest_minute_data["low"],
                close=latest_minute_data["close"],
                window=keltner_window,
                window_atr=keltner_atr_window
            )
            safe_calculation(latest_minute_data, f"Keltner_Upper", keltner_channel.keltner_channel_hband_indicator)
            safe_calculation(latest_minute_data, f"Keltner_Lower", keltner_channel.keltner_channel_lband_indicator)
            safe_calculation(latest_minute_data, f"VPT_{interval}", ta.volume.volume_price_trend, close=latest_minute_data["close"], volume=latest_minute_data["volume"])
            # Bollinger Bands Calculation (with enough historical data)
            bb_windows = [20]  # Standard BB window, add more if needed
            for window in bb_windows:
                bb_object = ta.volatility.BollingerBands(close=df["close"], window=window, window_dev=2)
                latest_minute_data[f"BB_high_{window}"] = bb_object.bollinger_hband().iloc[-1]
                latest_minute_data[f"BB_mid_{window}"] = bb_object.bollinger_mavg().iloc[-1]
                latest_minute_data[f"BB_low_{window}"] = bb_object.bollinger_lband().iloc[-1]
                # VPT Calculation (using the Williams %R-like approach)
            vpt_object = ta.volume.VolumePriceTrendIndicator(close=df["close"], volume=df["volume"])
            latest_minute_data[f"VPT"] = vpt_object.volume_price_trend().iloc[-1]



        perform_calculations()
        latest_minute_data[f"timestamp"] = latest_minute_data.index
        return latest_minute_data.reset_index(drop=True)

    intervals = ["1min", "5min", "15min"]
    results = await asyncio.gather(*[fetch_and_process_data(interval) for interval in intervals])

    processed_results = []
    for i, result in enumerate(results):
        result.columns = [f"{col}_{intervals[i]}" for col in result.columns]
        result = result.dropna(how='all')
        processed_results.append(result)

    # Concatenate the DataFrames horizontally (side by side)
    final_df = pd.concat(processed_results, axis=1)

    # Drop columns with all NaN values after concatenation
    final_df.dropna(axis=1, how='all', inplace=True)

    # print("final df = ", final_df)
    return final_df

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

sem = asyncio.Semaphore(1)  # Adjust the number as appropriate was10

import math





async def post_market_quotes(session, ticker, real_auth):
    url = "https://api.tradier.com/v1/markets/quotes"
    headers = {"Authorization": f"Bearer {real_auth}", "Accept": "application/json"}
    all_contracts = await lookup_all_option_contracts(session, ticker, real_auth)

    # Batching logic
    BATCH_SIZE = 8000
    num_batches = math.ceil(len(all_contracts) / BATCH_SIZE)
    results = []
    timeout = aiohttp.ClientTimeout(total=45)
    for batch_num in range(num_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = (batch_num + 1) * BATCH_SIZE
        batch_contracts = all_contracts[start_idx:end_idx]
        symbols_str = ",".join(batch_contracts)

        payload = {"symbols": symbols_str, "greeks": "true"}

        try:
            async with sem:
                async with session.post(url, data=payload, headers=headers,timeout=timeout) as response:
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    data = await response.json()
                    rate_limit_allowed = int(response.headers.get("X-Ratelimit-Allowed", "0"))
                    rate_limit_used = int(response.headers.get("X-Ratelimit-Used", "0"))
                    # print(rate_limit_allowed,rate_limit_used)
                    # Check if rate limit used exceeds allowed
                    # limit
                    if rate_limit_used >= (rate_limit_allowed * 0.99):
                        logger.error(
                            f"POST_{url},----Rate limit exceeded: Used {rate_limit_used} out of {rate_limit_allowed}"
                        )
                    # print( f"POST_{url}, {ticker},----Rate limit Used {rate_limit_used} out of {rate_limit_allowed}")
                    if "quotes" in data and "quote" in data["quotes"]:
                        quotes = data["quotes"]["quote"]
                        # if isinstance(quotes, dict):  Dont think i need this... was to make sure whern theres 1 batch, its still list.
                        #     quotes = [quotes]
                        # print(type(quotes))
                        results.append(pd.DataFrame(quotes))

                    else:
                        print(f"Error: No market quote data found for {ticker} (batch {batch_num}).")
                        raise Exception
            await asyncio.sleep(0.1)  # Short delay to avoid overwhelming API (adjust as needed)

        except aiohttp.ClientError as e:
            print(f"Error fetching market quotes (batch {batch_num}): {e}")

    # Combine results and handle potential empty results
    combined_df = pd.concat(results, ignore_index=True) if results else None
    return combined_df



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
            # print( f"get_{url},{params},----Rate limit Used {rate_limit_used} out of {rate_limit_allowed}")
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


async def lookup_all_option_contracts(session, underlying, real_auth):  # Add 'sem' as argument
    async with sem:  # Acquire the semaphore before making the request
        url = "https://api.tradier.com/v1/markets/options/lookup"
        params = {"underlying": underlying}
        headers = {"Authorization": f"Bearer {real_auth}", "Accept": "application/json"}

        try:
            async with session.get(url, params=params, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                rate_limit_allowed = int(response.headers.get("X-Ratelimit-Allowed", "0"))
                rate_limit_used = int(response.headers.get("X-Ratelimit-Used", "0"))
                # print( f"lookup_{url},{params},----Rate limit Used {rate_limit_used} out of {rate_limit_allowed}")

                if "symbols" in data and data["symbols"]:
                    option_contracts = data["symbols"][0]["options"]
                    return sorted(option_contracts)
                else:
                    raise Exception(f"No option lookup data found for {underlying}.")
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching option contracts: {e}")
            return None
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


    all_contract_quotes = await post_market_quotes(session,ticker,real_auth)
    # # Wait for all tasks to complete
    try:
        quotes_response = await asyncio.gather(*tasks)  # Default is return_exceptions=False
    except Exception as e:
        logger.error(f"An error occurred in fetching data: {e}")
        raise
    # # Check for exceptions in responses
    # for response in responses:
    #     if isinstance(response, Exception):
    #         logger.error(f"An error occurred in fetching data: {response}")
    #         raise response

    ticker_quote = quotes_response[0]

    quote_df = pd.DataFrame.from_dict(
        ticker_quote["quotes"]["quote"], orient="index"
    ).T
    LAC = quote_df.at[0, "prevclose"]
    open = quote_df.at[0, "open"]
    # print(open)
    high = quote_df.at[0, "high"]
    low = quote_df.at[0, "low"]
    close = quote_df.at[0, "close"]
    average_volume = quote_df.at[0, "average_volume"]
    # if average_volume < 1:
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

    # expirations = expirations_response["expirations"]["expiration"]
    # expiration_dates = [expiration["date"] for expiration in expirations]

    # callsChain = []
    # putsChain = []

    # print(all_contract_quotes)
    if all_contract_quotes is not None:
        # print("all_contrats", type(all_contract_quotes),all_contract_quotes)
        grouped = all_contract_quotes.groupby("option_type")
        calls_df = grouped.get_group("call").copy()
        puts_df = grouped.get_group("put").copy()
        # callsChain.append(call_group)
        # putsChain.append(put_group)



    # Calculate new columns
    for df in [calls_df, puts_df]:
        df['Moneyness'] = np.where(df['option_type'] == 'call',
                                   df['strike'] - CurrentPrice,
                                   CurrentPrice - df['strike'])

        # df["LACdollarsFromStrike"] = abs(df["strike"] - LAC)
        df["dollarsFromStrike"] = abs(df["strike"] - LAC)
        df["ExpDate"] = df["symbol"].str[-15:-9]
        df["Strike"] = df["strike"]

        # df["LACdollarsFromStrikeXoi"] = df["LACdollarsFromStrike"] * df["open_interest"]
        df["dollarsFromStrikeXoi"] = df["dollarsFromStrike"] * df["open_interest"]
        df["MoneynessXoi"] = df["Moneyness"] * df["open_interest"]
        df["lastPriceXoi"] = df["last"] * df["open_interest"]
        # df["impliedVolatility"] = df["greeks"].str.get("mid_iv")
        # df["delta"] = df["greeks"].str.get("delta")
    # calls_df["lastContractPricexOI"] = calls_df["last"] * calls_df["open_interest"]
    # calls_df["impliedVolatility"] = calls_df["greeks"].str.get("mid_iv")

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
        # "LACdollarsFromStrike": "LACdollarsFromStrike",
        "dollarsFromStrikeXoi": "dollarsFromStrikeXoi",
        # "LACdollarsFromStrikeXoi": "LACdollarsFromStrikeXoi",
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
        # "c_LACdollarsFromStrike": "Calls_LACdollarsFromStrike",
        "c_dollarsFromStrike": "Calls_dollarsFromStrike",
        # "c_LACdollarsFromStrikeXoi": "Calls_LACdollarsFromStrikeXoi",
        "c_dollarsFromStrikeXoi": "Calls_dollarsFromStrikeXoi",
        "c_lastPriceXoi": "Calls_lastPriceXoi",
        "p_lastPrice": "Put_LastPrice",
        "p_volume": "Put_Volume",
        "p_openInterest": "Put_OI",
        # "p_impliedVolatility": "Put_IV",
        # "p_LACdollarsFromStrike": "Puts_LACdollarsFromStrike",
        "p_dollarsFromStrike": "Puts_dollarsFromStrike",
        # "p_LACdollarsFromStrikeXoi": "Puts_LACdollarsFromStrikeXoi",
        "p_dollarsFromStrikeXoi": "Puts_dollarsFromStrikeXoi",
        "p_lastPriceXoi": "Puts_lastPriceXoi",
    }
    combined.rename(columns=rename_dict_combined, inplace=True)
    for column in [
        "LAC",
        "CurrentPrice",
        "open",
        "high",
        "low",
        "close",
        "average_volume",
        "last_volume",
    ]:
        combined[column] = np.nan

    # Populate values only in the first row
    first_index = combined.index[0]  # Get the first index of the DataFrame
    combined.loc[
        first_index,
        ["LAC", "CurrentPrice", "open", "high", "low","close", "average_volume", "last_volume"],
    ] = [LAC, CurrentPrice, open, high, low,close, average_volume, last_volume]


    ####################
    # for option in json_response["options"]["option"]:
    #     print(option["symbol"], option["open_interest"])
    ##weighted total iv for contract
    # Total IV = (bid IV * bid volume + mid IV * mid volume + ask IV * ask volume) / (bid volume + mid volume + ask volume)
    # vega measures response to IV change.
    try:
        this_minute_ta_frame = await get_ta(session, ticker)
        # logger.info(f"Successfully retrieved TA data for {ticker}")

        # Efficiently combine all columns
        combined = pd.concat([combined, this_minute_ta_frame], axis=1)
        logger.info(f"Successfully concatenated TA data for {ticker}")

        # Optionally, drop NaN columns if they are not relevant for your analysis
        combined.dropna(axis=1, how="all", inplace=True)
        # logger.info(f"Dropped NaN columns for {ticker} if any")
    except Exception as e:
        logger.error(f"Error combining TA data for {ticker}: {e}\nTraceback: {traceback.format_exc()}")
    output_dir = Path(f"data/optionchain/{ticker}/{YYMMDD}")
    output_dir.mkdir(mode=0o755, parents =True, exist_ok=True)

    # Make sure it's the same day's data.
    # if YYMMDD == StockLastTradeTime_YMD:#TDDO for test mode comment this

    # file_path2 = f"data/optionchain/{ticker}/{YYMMDD}/aaaicker_{YYMMDD_HHMM}.csv"
    # print(f"{YYMMDD_HHMM}: ${ticker} last Trade Time: {StockLastTradeTime_str}")
    # this_minute_ta_frame.to_csv(file_path2, mode="w", index=False)
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
