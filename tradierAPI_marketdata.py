import requests
from datetime import datetime, timedelta
import pandas as pd
import ta
from pathlib import Path
import numpy as np
import PrivateData.tradier_info
concurrency_limit=2
# import webullAPI
# Add a small constant to denominators taper_acc = PrivateData.tradier_info.paper_acc
paper_auth = PrivateData.tradier_info.paper_auth
real_acc = PrivateData.tradier_info.real_acc
real_auth = PrivateData.tradier_info.real_auth

###TODO time and sales (will be used for awesome ind. and ta.
YYMMDD = datetime.today().strftime("%y%m%d")
import aiohttp
import asyncio

#TODO for now this ignores the divede by zero warnings.
np.seterr(divide='ignore', invalid='ignore')
async def get_option_chain(session, ticker, exp_date, headers):
    response = await session.get(
        "https://api.tradier.com/v1/markets/options/chains",
        params={"symbol": ticker, "expiration": exp_date, "greeks": "true"},
        headers=headers,
    )
    json_response = await response.json()
    # print(response.status_code)
    # print("Option Chain: ",json_response)
    optionchain_df = pd.DataFrame(json_response["options"]["option"])
    return optionchain_df

async def get_option_chains_concurrently(session,ticker, expiration_dates, headers):
    tasks = []
    for exp_date in expiration_dates:
        tasks.append(get_option_chain(session, ticker, exp_date, headers))
    all_option_chains = await asyncio.gather(*tasks)
    return all_option_chains
async def fetch(session, url, params, headers):
    async with session.get(url, params=params, headers=headers) as response:
        # print("Rate Limit Headers:")
        # print("Allowed:", response.headers.get("X-Ratelimit-Allowed"))
        # print("Used:", response.headers.get("X-Ratelimit-Used"))
        return await response.json()


async def get_options_data(session,ticker):
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
    df = pd.DataFrame(json_response["series"]["data"]).set_index("time")
    # df.set_index('time', inplace=True)
    ##change index to datetimeindex
    df.index = pd.to_datetime(df.index)

    close = df["close"]
    df["AwesomeOsc"] = ta.momentum.awesome_oscillator(
        high=df["high"], low=df["low"], window1=1, window2=5, fillna=False
    )
    df["AwesomeOsc5_34"] = ta.momentum.awesome_oscillator(
        high=df["high"], low=df["low"], window1=5, window2=34, fillna=False
    )
    df["RSI"] = ta.momentum.rsi(close=close, window=5, fillna=False)
    df["RSI2"] = ta.momentum.rsi(close=close, window=2, fillna=False)
    df["RSI14"] = ta.momentum.rsi(close=close, window=14, fillna=False)
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
    # print(f"LAC: ${LAC}")
    # print(f"{ticker} Current Price: ${CurrentPrice}")

    expirations = expirations_response["expirations"]["expiration"]
    expiration_dates = [expiration["date"] for expiration in expirations]

    callsChain = []
    putsChain = []
    all_option_chains = await get_option_chains_concurrently(session,ticker, expiration_dates, headers)

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
                       'open_interest', 'ExpDate', 'Strike','lastPriceXoi','impliedVolatility','dollarsFromStrikeXoi']

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

    combined.rename(columns=rename_dict_combined, inplace=True)
####################
    # for option in json_response["options"]["option"]:
    #     print(option["symbol"], option["open_interest"])
    ##weighted total iv for contract
    # Total IV = (bid IV * bid volume + mid IV * mid volume + ask IV * ask volume) / (bid volume + mid volume + ask volume)
    # vega measures response to IV change.

    output_dir = Path(f"data/optionchain/{ticker}/{YYMMDD}")
    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

    try:
        combined.to_csv(f"data/optionchain/{ticker}/{YYMMDD}/{ticker}_{StockLastTradeTime}.csv", mode="x")
    except Exception as e:
        if FileExistsError:
            if StockLastTradeTime == 1600:
                combined.to_csv(f"data/optionchain/{ticker}/{YYMMDD}/{ticker}_{StockLastTradeTime}(2).csv")
        else:
            print(f"An error occurred while writing the CSV file,: {e}")
            combined.to_csv(f"data/optionchain/{ticker}/{YYMMDD}/{ticker}_{StockLastTradeTime}(2).csv")
    # combined.to_csv(f"combined_tradier.csv")
    ###strike, exp, call last price, call oi, iv,vol, $ from strike, dollars from strike x OI, last price x OI

    return LAC, CurrentPrice, price_change_percent, StockLastTradeTime, this_minute_ta_frame, expiration_dates


#
def perform_operations(
        ticker,
        last_adj_close,
        current_price,
        price_change_percent,
        StockLastTradeTime,
        this_minute_ta_frame,
        expiration_dates,
):
    results = []

    data = pd.read_csv(f"data/optionchain/{ticker}/{YYMMDD}/{ticker}_{StockLastTradeTime}.csv")

    groups = data.groupby("ExpDate")
    # divide into groups by exp date, call info from group.
    for exp_date, group in groups:
        pain_list = []
        strike_LASTPRICExOI_list = []
        call_LASTPRICExOI_list = []
        put_LASTPRICExOI_list = []
        call_price_dict = (
            group.loc[group["Call_LastPrice"] >= 0, ["Strike", "Call_LastPrice"]].set_index("Strike").to_dict()
        )

        strike = group["Strike"]
        # print("strike column for group",strike)
        # pain is ITM puts/calls
        calls_LASTPRICExOI_dict = (
            group.loc[group["Calls_lastPriceXoi"] >= 0, ["Strike", "Calls_lastPriceXoi"]].set_index("Strike").to_dict()
        )
        puts_LASTPRICExOI_dict = (
            group.loc[group["Puts_lastPriceXoi"] >= 0, ["Strike", "Puts_lastPriceXoi"]].set_index("Strike").to_dict()
        )
        # calls_DFSxOI_dict = (
        #     group.loc[group["Calls_dollarsFromStrikeXoi"] >= 0, ["Strike", "Calls_dollarsFromStrikeXoi"]]
        #     .set_index("Strike")
        #     .to_dict()
        # )
        # puts_DFSxOI_dict = (
        #     group.loc[group["Puts_dollarsFromStrikeXoi"] >= 0, ["Strike", "Puts_dollarsFromStrikeXoi"]]
        #     .set_index("Strike")
        #     .to_dict()
        # )


        ITM_CallsVol = group.loc[(group["Strike"] <= current_price), "Call_Volume"].sum()
        ITM_PutsVol = group.loc[(group["Strike"] >= current_price), "Put_Volume"].sum()
        ITM_CallsOI = group.loc[(group["Strike"] <= current_price), "Call_OI"].sum()
        ITM_PutsOI = group.loc[(group["Strike"] >= current_price), "Put_OI"].sum()

        all_CallsVol = group.Call_Volume.sum()
        all_PutsVol = group.Put_Volume.sum()

        all_CallsOI = group.Call_OI.sum()
        all_PutsOI = group.Put_OI.sum()

        all_OI = all_PutsOI + all_CallsOI
        ITM_OI = ITM_CallsOI + ITM_PutsOI

        ITM_Call_IV = group.loc[(group["Strike"] <= current_price), "Call_IV"].sum()
        ITM_Put_IV = group.loc[(group["Strike"] >= current_price), "Put_IV"].sum()

        Call_IV = group["Call_IV"].sum()
        Put_IV = group["Put_IV"].sum()

        # Now that we have calculated sums, we use them for further calculations

        ITM_Avg_Net_IV = ITM_Call_IV - ITM_Put_IV
        Net_IV = Call_IV - Put_IV

        PC_Ratio_Vol = all_PutsVol / all_CallsVol if all_CallsVol != 0 and not np.isnan(all_CallsVol) else np.nan
        ITM_PC_Ratio_Vol = ITM_PutsVol / ITM_CallsVol if ITM_CallsVol != 0 and not np.isnan(ITM_CallsVol) else np.nan
        PC_Ratio_OI = all_PutsOI / all_CallsOI if all_CallsOI != 0 and not np.isnan(all_CallsOI) else np.nan
        ITM_PC_Ratio_OI = ITM_PutsOI / ITM_CallsOI if ITM_CallsOI != 0 and not np.isnan(ITM_CallsOI) else np.nan

        # DFSxOI_dict = (
        #     group.loc[group["Puts_dollarsFromStrikeXoi"] >= 0, ["Strike", "Puts_dollarsFromStrikeXoi"]]
        #     .set_index("Strike")
        #     .to_dict()
        # )
        # Money_weighted_PC_Ratio =
        ###TODO add highest premium puts/calls, greeks corelation?
        ###TODO correlate volume and IV, high volume high iv = contracts being bought, high volume, low vol. = contracts being sold.
        # print("wassup123",strike)
        for strikeprice in strike:
            itmCalls_dollarsFromStrikeXoiSum = group.loc[
                (group["Strike"] < strikeprice), "Calls_dollarsFromStrikeXoi"
            ].sum()
            itmPuts_dollarsFromStrikeXoiSum = group.loc[
                (group["Strike"] > strikeprice), "Puts_dollarsFromStrikeXoi"
            ].sum()
            call_LASTPRICExOI = calls_LASTPRICExOI_dict.get("Calls_lastPriceXoi", {}).get(strikeprice, 0)
            put_LASTPRICExOI = puts_LASTPRICExOI_dict.get("Puts_lastPriceXoi", {}).get(strikeprice, 0)
            # call_DFSxOI = calls_DFSxOI_dict.get("Calls_dollarsFromStrikeXoi", {}).get(strikeprice, 0)
            # put_DFSxOI = puts_DFSxOI_dict.get("Puts_dollarsFromStrikeXoi", {}).get(strikeprice, 0)
            pain_value = itmPuts_dollarsFromStrikeXoiSum + itmCalls_dollarsFromStrikeXoiSum
            pain_list.append((strikeprice, pain_value))
            strike_LASTPRICExOI = call_LASTPRICExOI + put_LASTPRICExOI
            strike_LASTPRICExOI_list.append((strikeprice, strike_LASTPRICExOI))
            call_LASTPRICExOI_list.append((strikeprice, call_LASTPRICExOI))
            put_LASTPRICExOI_list.append((strikeprice, put_LASTPRICExOI))

        highest_premium_strike = max(strike_LASTPRICExOI_list, key=lambda x: x[1])[0]
        highest_premium_call = max(call_LASTPRICExOI_list, key=lambda x: x[1])[0]
        highest_premium_put = max(put_LASTPRICExOI_list, key=lambda x: x[1])[0]
        max_pain = min(pain_list, key=lambda x: x[1])[0]
        top_five_calls = group.loc[group["Call_OI"] > 0].sort_values(by="Call_OI", ascending=False).head(5)
        top_five_calls_dict = top_five_calls[["Strike", "Call_OI"]].set_index("Strike").to_dict()["Call_OI"]
        top_five_puts = group.loc[group["Put_OI"] > 0].sort_values(by="Put_OI", ascending=False).head(5)
        top_five_puts_dict = top_five_puts[["Strike", "Put_OI"]].set_index("Strike").to_dict()["Put_OI"]

        ### FINDING CLOSEST STRIKE TO LAc
        # target number from column A
        # calculate difference between target and each value in column B
        data["strike_lac_diff"] = group["Strike"].apply(lambda x: abs(x - last_adj_close))
        ###############################
        if not group.empty:
            smallest_change_from_lac = data["strike_lac_diff"].abs().idxmin()
            closest_strike_lac = group.loc[smallest_change_from_lac, "Strike"]

            # Find index of row with the closest strike to the current price
            current_price_index = group["Strike"].sub(current_price).abs().idxmin()

            # Create a list of higher and lower strike indexes
            higher_strike_indexes = [i for i in range(current_price_index + 1, current_price_index + 5) if
                                     i in group.index]
            lower_strike_indexes = [i for i in range(current_price_index - 1, current_price_index - 5, -1) if
                                    i in group.index]

            # Initialize the lists for the closest strikes
            closest_higher_strikes = group.loc[higher_strike_indexes, "Strike"].tolist()
            closest_lower_strikes = group.loc[lower_strike_indexes, "Strike"].tolist()

            # Append None values to the lists to ensure they have a length of 4
            closest_higher_strikes += [None] * (4 - len(closest_higher_strikes))
            closest_lower_strikes += [None] * (4 - len(closest_lower_strikes))

            closest_strike_currentprice = group.loc[current_price_index, "Strike"]
        else:
            closest_strike_lac = None
            closest_strike_currentprice = None
            closest_higher_strikes = [None] * 4
            closest_lower_strikes = [None] * 4

        # Create the strikeindex_abovebelow list
        strikeindex_abovebelow = closest_lower_strikes[::-1] + [closest_strike_currentprice] + closest_higher_strikes
        closest_lower_strike4 = strikeindex_abovebelow[0]
        closest_lower_strike3 = strikeindex_abovebelow[1]
        closest_lower_strike2 = strikeindex_abovebelow[2]
        closest_lower_strike1 = strikeindex_abovebelow[3]
        closest_higher_strike1= strikeindex_abovebelow[5]
        closest_higher_strike2= strikeindex_abovebelow[6]
        closest_higher_strike3=strikeindex_abovebelow[7]
        closest_higher_strike4=strikeindex_abovebelow[8]
#################

        ##Gettting pcr-vol for individual strikes above/below CP(closest to current price strike)
        def calculate_pcr_ratio(put_data, call_data):
            if np.isnan(put_data) or np.isnan(call_data) or call_data == 0:
                return np.inf if call_data == 0 and put_data !=0 and not np.isnan(put_data) else np.nan
            else:
                return put_data / call_data

        group_strike = group.groupby('Strike')

        # Initialize dictionaries for storing PCR values
        strike_PCRv_dict = {}
        strike_PCRoi_dict = {}
        strike_ITMPCRv_dict = {}
        strike_ITMPCRoi_dict = {}

        # Calculate PCR values for all strikes in strikeindex_abovebelow
        for strikeabovebelow in strikeindex_abovebelow:
            strike_data = group_strike.get_group(strikeabovebelow) if strikeabovebelow is not None else None

            if strike_data is None:
                strike_PCRv_dict[strikeabovebelow] = np.nan
                strike_PCRoi_dict[strikeabovebelow] = np.nan
                strike_ITMPCRv_dict[strikeabovebelow] = np.nan
                strike_ITMPCRoi_dict[strikeabovebelow] = np.nan
                continue
            strike_PCRv_dict[strikeabovebelow] = calculate_pcr_ratio(strike_data["Put_Volume"].values[0],
                                                           strike_data["Call_Volume"].values[0])
            strike_PCRoi_dict[strikeabovebelow] = calculate_pcr_ratio(strike_data["Put_OI"].values[0],
                                                            strike_data["Call_OI"].values[0])

            # Calculate ITM PCR values for strikes above and below the current strike
            # For puts, the strike is higher



            itm_put_strike_data = group.loc[group["Strike"] >= strikeabovebelow]
            itm_call_strike_data = group.loc[group["Strike"] <= strikeabovebelow]

            itm_put_volume = itm_put_strike_data["Put_Volume"].sum()
            itm_call_volume = itm_call_strike_data["Call_Volume"].sum()
            if itm_call_volume == 0:
                strike_ITMPCRv_dict[strikeabovebelow] = np.nan
            else:
                strike_ITMPCRv_dict[strikeabovebelow] = itm_put_volume / itm_call_volume

            itm_put_oi = itm_put_strike_data["Put_OI"].sum()
            itm_call_oi = itm_call_strike_data["Call_OI"].sum()
            if itm_call_oi == 0:
                strike_ITMPCRoi_dict[strikeabovebelow] = np.nan
            else:
                strike_ITMPCRoi_dict[strikeabovebelow] = itm_put_oi / itm_call_oi

#         ####################
#         for strikeabovebelow in strikeindex_abovebelow:
#             if strikeabovebelow == None:
#                 strike_ITMPCRv_dict[strikeabovebelow] = np.nan
#             else:
#                 strike_ITMPCRvput_volume = group.loc[(group["Strike"] >= strikeabovebelow), "Put_Volume"].sum()
#                 strike_ITMPCRvcall_volume = group.loc[(group["Strike"] <= strikeabovebelow), "Call_Volume"].sum()
#                 if strike_ITMPCRvcall_volume == 0:
#                     strike_ITMPCRv_dict[strikeabovebelow] = np.nan
#                 else:
#                     strike_ITMPCRv_dict[strikeabovebelow] = strike_ITMPCRvput_volume / strike_ITMPCRvcall_volume
#
#         for strikeabovebelow in strikeindex_abovebelow:
#             if strikeabovebelow == None:
#                 strike_ITMPCRoi_dict[strikeabovebelow] = np.nan
#             else:
#                 strike_ITMPCRoiput_volume = group.loc[(group["Strike"] >= strikeabovebelow), "Put_OI"].sum()
#                 strike_ITMPCRoicall_volume = group.loc[(group["Strike"] <= strikeabovebelow), "Call_OI"].sum()
#                 if strike_ITMPCRoicall_volume == 0:
#                     strike_ITMPCRoi_dict[strikeabovebelow] = np.nan
#                 else:
#                     strike_ITMPCRoi_dict[strikeabovebelow] = strike_ITMPCRoiput_volume / strike_ITMPCRoicall_volume
# #####################

        def get_ratio_and_iv(strike):
            if strike is None:
                # handle the case where strike is None
                return None
            else:
                strike_data = group_strike.get_group(strike)
                ratio_v = calculate_pcr_ratio(strike_data["Put_Volume"].values[0], strike_data["Call_Volume"].values[0])
                ratio_oi = calculate_pcr_ratio(strike_data["Put_OI"].values[0], strike_data["Call_OI"].values[0])
                call_iv = strike_data["Call_IV"].sum()
                put_iv = strike_data["Put_IV"].sum()
                net_iv = call_iv - put_iv
                return ratio_v, ratio_oi, call_iv, put_iv, net_iv
        # Calculate PCR values for the closest strike to LAC
        PC_Ratio_Vol_Closest_Strike_LAC, PC_Ratio_OI_Closest_Strike_LAC, Call_IV_Closest_Strike_LAC, Put_IV_Closest_Strike_LAC, Net_IV_Closest_Strike_LAC = get_ratio_and_iv(
            closest_strike_lac)
        # Calculate PCR values for the closest strike to CP
        PCRv_cp_strike, PCRoi_cp_strike, _, _, _ = get_ratio_and_iv(closest_strike_currentprice)



        # Calculate PCR values for Max Pain strike
        PC_Ratio_Vol_atMP, PC_Ratio_OI_atMP, Net_Call_IV_at_MP, Net_Put_IV_at_MP, Net_IV_at_MP = get_ratio_and_iv(
            max_pain)

        NIV_CurrentStrike = get_ratio_and_iv(closest_strike_currentprice)[
            4] if closest_strike_currentprice is not None else np.nan
        NIV_1HigherStrike = get_ratio_and_iv(closest_higher_strike1)[
            4] if closest_higher_strike1 is not None else np.nan
        NIV_2HigherStrike = get_ratio_and_iv(closest_higher_strike2)[
            4] if closest_higher_strike2 is not None else np.nan
        NIV_3HigherStrike = get_ratio_and_iv(closest_higher_strike3)[
            4] if closest_higher_strike3 is not None else np.nan
        NIV_4HigherStrike = get_ratio_and_iv(closest_higher_strike4)[
            4] if closest_higher_strike4 is not None else np.nan
        NIV_1LowerStrike = get_ratio_and_iv(closest_lower_strike1)[4] if closest_lower_strike1 is not None else np.nan
        NIV_2LowerStrike = get_ratio_and_iv(closest_lower_strike2)[4] if closest_lower_strike2 is not None else np.nan
        NIV_3LowerStrike = get_ratio_and_iv(closest_lower_strike3)[4] if closest_lower_strike3 is not None else np.nan
        NIV_4LowerStrike = get_ratio_and_iv(closest_lower_strike4)[4] if closest_lower_strike4 is not None else np.nan

        ###TODO error handling for scalar divide of zero denominator


        Bonsai_Ratio = ((ITM_PutsVol / all_PutsVol) * (ITM_PutsOI / all_PutsOI)) / (
                (ITM_CallsVol / all_CallsVol) * (ITM_CallsOI / all_CallsOI))
        Bonsai2_Ratio = ((all_PutsVol / ITM_PutsVol) / (all_PutsOI / ITM_PutsOI)) * (
                (all_CallsVol / ITM_CallsVol) / (all_CallsOI / ITM_CallsOI))
        round(strike_PCRv_dict[closest_higher_strike1], 3),
        results.append(
            {
                ###TODO change all price data to percentage change?
                ###TODO change closest strike to average of closest above/closest below
                "ExpDate": exp_date,
                "LastTradeTime": StockLastTradeTime,
                "Current Stock Price": float(current_price),
                "Current SP % Change(LAC)": round(float(price_change_percent), 2),
                # 'IV 30': iv30,
                # 'IV 30 % change': iv30_change_percent,
                "Maximum Pain": max_pain,
                "Bonsai Ratio": round(Bonsai_Ratio, 5),
                # 'Bonsai %change': bonsai_percent_change,
                "Bonsai Ratio 2": round(Bonsai2_Ratio, 5),
                "B1/B2": round((Bonsai_Ratio / Bonsai2_Ratio), 4),
                "B2/B1": round((Bonsai2_Ratio / Bonsai_Ratio), 4),
                # TODO ITM contract $ %
                "PCR-Vol": round(PC_Ratio_Vol, 3),
                "PCR-OI": round(PC_Ratio_OI, 3),
                "PCRv @CP Strike": round(PCRv_cp_strike, 3),
                "PCRoi @CP Strike": round(PCRoi_cp_strike, 3),
                "PCRv Up1": round(strike_PCRv_dict[closest_higher_strike1], 3),
                "PCRv Up2": round(strike_PCRv_dict[closest_higher_strike2], 3),
                "PCRv Up3": round(strike_PCRv_dict[closest_higher_strike3], 3),
                "PCRv Up4": round(strike_PCRv_dict[closest_higher_strike4], 3),
                "PCRv Down1": round(strike_PCRv_dict[closest_lower_strike1], 3),
                "PCRv Down2": round(strike_PCRv_dict[closest_lower_strike2], 3),
                "PCRv Down3": round(strike_PCRv_dict[closest_lower_strike3], 3),
                "PCRv Down4": round(strike_PCRv_dict[closest_lower_strike4], 3),
                "PCRoi Up1": round(strike_PCRoi_dict[closest_higher_strike1], 3),
                "PCRoi Up2": round(strike_PCRoi_dict[closest_higher_strike2], 3),
                "PCRoi Up3": round(strike_PCRoi_dict[closest_higher_strike3], 3),
                "PCRoi Up4": round(strike_PCRoi_dict[closest_higher_strike4], 3),
                "PCRoi Down1": round(strike_PCRoi_dict[closest_lower_strike1], 3),
                "PCRoi Down2": round(strike_PCRoi_dict[closest_lower_strike2], 3),
                "PCRoi Down3": round(strike_PCRoi_dict[closest_lower_strike3], 3),
                "PCRoi Down4": round(strike_PCRoi_dict[closest_lower_strike4], 3),
                "ITM PCR-Vol": round(ITM_PC_Ratio_Vol, 2),
                "ITM PCR-OI": round(ITM_PC_Ratio_OI, 3),
                "ITM PCRv Up1": strike_ITMPCRv_dict[closest_higher_strike1],
                "ITM PCRv Up2": strike_ITMPCRv_dict[closest_higher_strike2],
                "ITM PCRv Up3": strike_ITMPCRv_dict[closest_higher_strike3],
                "ITM PCRv Up4": strike_ITMPCRv_dict[closest_higher_strike4],
                "ITM PCRv Down1": strike_ITMPCRv_dict[closest_lower_strike1],
                "ITM PCRv Down2": strike_ITMPCRv_dict[closest_lower_strike2],
                "ITM PCRv Down3": strike_ITMPCRv_dict[closest_lower_strike3],
                "ITM PCRv Down4": strike_ITMPCRv_dict[closest_lower_strike4],
                "ITM PCRoi Up1": strike_ITMPCRoi_dict[closest_higher_strike1],
                "ITM PCRoi Up2": strike_ITMPCRoi_dict[closest_higher_strike2],
                "ITM PCRoi Up3": strike_ITMPCRoi_dict[closest_higher_strike3],
                "ITM PCRoi Up4": strike_ITMPCRoi_dict[closest_higher_strike4],
                "ITM PCRoi Down1": strike_ITMPCRoi_dict[closest_lower_strike1],
                "ITM PCRoi Down2": strike_ITMPCRoi_dict[closest_lower_strike2],
                "ITM PCRoi Down3": strike_ITMPCRoi_dict[closest_lower_strike3],
                "ITM PCRoi Down4": strike_ITMPCRoi_dict[closest_lower_strike4],
                "ITM OI": ITM_OI,
                "Total OI": all_OI,
                "ITM Contracts %": ITM_OI / all_OI,
                "Net_IV": round(Net_IV, 3),
                "Net ITM IV": round(ITM_Avg_Net_IV, 3),
                "Net IV MP": round(Net_IV_at_MP, 3),
                "Net IV LAC": round(Net_IV_Closest_Strike_LAC, 3),
                "NIV Current Strike": round(NIV_CurrentStrike, 3),
                "NIV 1Higher Strike": round(NIV_1HigherStrike, 3),
                "NIV 1Lower Strike": round(NIV_1LowerStrike, 3),
                "NIV 2Higher Strike": round(NIV_2HigherStrike, 3),
                "NIV 2Lower Strike": round(NIV_2LowerStrike, 3),
                "NIV 3Higher Strike": round(NIV_3HigherStrike, 3),
                "NIV 3Lower Strike": round(NIV_3LowerStrike, 3),
                "NIV 4Higher Strike": round(NIV_4HigherStrike, 3),
                "NIV 4Lower Strike": round(NIV_4LowerStrike, 3),
                ###Positive number means NIV highers are higher, and price will drop.
                # TODO should do as percentage change from total niv numbers to see if its big diff.
                "NIV highers(-)lowers1-2": (NIV_1HigherStrike + NIV_2HigherStrike)
                                           - (NIV_1LowerStrike + NIV_2LowerStrike),
                "NIV highers(-)lowers1-4": (
                                                   NIV_1HigherStrike + NIV_2HigherStrike + NIV_3HigherStrike + NIV_4HigherStrike
                                           )
                                           - (
                                                       NIV_1LowerStrike + NIV_2LowerStrike + NIV_3LowerStrike + NIV_4LowerStrike),
                "NIV 1-2 % from mean": (
                                               ((NIV_1HigherStrike + NIV_2HigherStrike) - (
                                                           NIV_1LowerStrike + NIV_2LowerStrike))
                                               / ((
                                                              NIV_1HigherStrike + NIV_2HigherStrike + NIV_1LowerStrike + NIV_2LowerStrike) / 4)
                                       )
                                       * 100,
                "NIV 1-4 % from mean": (
                                               (
                                                           NIV_1HigherStrike + NIV_2HigherStrike + NIV_3HigherStrike + NIV_4HigherStrike)
                                               - (
                                                           NIV_1LowerStrike + NIV_2LowerStrike + NIV_3LowerStrike + NIV_4LowerStrike)
                                               / (
                                                       (
                                                               NIV_1HigherStrike
                                                               + NIV_2HigherStrike
                                                               + NIV_3HigherStrike
                                                               + NIV_4HigherStrike
                                                               + NIV_1LowerStrike
                                                               + NIV_2LowerStrike
                                                               + NIV_3LowerStrike
                                                               + NIV_4LowerStrike
                                                       )
                                                       / 8
                                               )
                                       )
                                       * 100,
                ##TODO swap (/) with result = np.divide(x, y)
                "Net_IV/OI": Net_IV / all_OI,
                "Net ITM_IV/ITM_OI": ITM_Avg_Net_IV / ITM_OI,
                "Closest Strike to CP": closest_strike_currentprice,
                "Closest Strike Above/Below(below to above,4 each) list": strikeindex_abovebelow,
            }
        )
    df = pd.DataFrame(results)
    df["RSI"] = this_minute_ta_frame["RSI"]
    df["RSI2"] = this_minute_ta_frame["RSI2"]
    df["RSI14"] = this_minute_ta_frame["RSI14"]
    df["AwesomeOsc"] = this_minute_ta_frame["AwesomeOsc"]

    df["AwesomeOsc5_34"] = this_minute_ta_frame["AwesomeOsc5_34"]  # this_minute_ta_frame['exp_date'] = '230427.0'

    output_dir = Path(f"data/ProcessedData/{ticker}/{YYMMDD}/")

    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    output_dir_dailyminutes = Path(f"data/DailyMinutes/{ticker}")
    output_file_dailyminutes = Path(f"data/DailyMinutes/{ticker}/{ticker}_{YYMMDD}.csv")
    output_dir_dailyminutes_w_algo_results = Path(f"data/DailyMinutes_w_algo_results/{ticker}")
    output_dir_dailyminutes_w_algo_results.mkdir(mode=0o755, parents=True, exist_ok=True)
    output_dir_dailyminutes.mkdir(mode=0o755, parents=True, exist_ok=True)
    if output_file_dailyminutes.exists():
        dailyminutes = pd.read_csv(output_file_dailyminutes)
        dailyminutes = dailyminutes.dropna().drop_duplicates(subset="LastTradeTime")
        dailyminutes = pd.concat([dailyminutes, df.head(1)], ignore_index=True)
        dailyminutes.to_csv(output_file_dailyminutes, index=False)
    else:
        dailyminutes = pd.concat([df.head(1)], ignore_index=True)
        dailyminutes.to_csv(output_file_dailyminutes, index=False)

    try:
        df.to_csv(f"data/ProcessedData/{ticker}/{YYMMDD}/{ticker}_{StockLastTradeTime}.csv", mode="x", index=False)
    ###TODO could use this fileexists as a trigger to tell algos not to send(market clesed)
    except FileExistsError:
        print(f"data/ProcessedData/{ticker}/{YYMMDD}/{ticker}_{StockLastTradeTime}.csv", "File Already Exists.")
        # exit()
    return (
        f"data/optionchain/{ticker}/{YYMMDD}/{ticker}_{StockLastTradeTime}.csv",
        f"data/DailyMinutes/{ticker}/{ticker}_{YYMMDD}.csv",
        df,
        ticker,
    )
##df is processeddata