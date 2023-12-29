import asyncio
from pathlib import Path
import yfinance as yf
import pytz
import aiohttp
import numpy as np
import pandas as pd
import ta
import os
import requests
import pandas_market_calendars as mcal
from datetime import datetime, timedelta

import PrivateData.tradier_info
from UTILITIES.logger_config import logger
from tradierAPI_marketdata import fetch
from datetime import datetime, timedelta
import pytz

paper_auth = PrivateData.tradier_info.paper_auth
real_acc = PrivateData.tradier_info.real_acc
real_auth = PrivateData.tradier_info.real_auth
from datetime import datetime, timedelta

def is_within_last_30_days(date):#ACtaully targeting 11/23 to 11/28 because thats what was missing from processed/daily.
    current_date = datetime.now(pytz.utc)  # Making current_date offset-aware
    thirty_days_ago = current_date - timedelta(days=30)
    return thirty_days_ago <= date <= current_date

def is_within_target_dates(date):
    # Define the start and end dates of the target range
    start_date = datetime(2023, 11, 26, tzinfo=pytz.utc)  # assuming the year is 2023
    end_date = datetime(2023, 11, 28, tzinfo=pytz.utc)

    # Check if the given date is within the target date range
    return start_date <= date <= end_date



async def fetch(session, url, params, headers):
    try:
        async with session.get(url, params=params, headers=headers) as response:
            content_type = response.headers.get('Content-Type', '')

            if 'application/json' in content_type:
                response =  await response.json()
                print(response)
                return response
            elif 'application/xml' in content_type or 'text/xml' in content_type:
                # XML parsing logic here
                xml_response = await response.text()
                print(xml_response)
                # ...
            else:
                print(f"Unsupported content type: {content_type}")
                return None

    except Exception as e:
        print(f"Connection error to {url}: {e}.")
        logger.exception(f"An error occurred while fetching data: {e} At URL {url}")

# TODO could overhaul so anything can be done from optinchain.. lots work hmm
from datetime import datetime



def str_YYMMDD_HHMM_to_datetime(date_str):
    # Parse the input string into a datetime object
    date = datetime.strptime(date_str, '%y%m%d_%H%M')
    # date_dt = date.strftime("%Y-%m-%d %H:%M")
    return date
semaphore = asyncio.Semaphore(5000)  # Adjust the number 10 to your desired concurrency limit

async def get_ta(session, ticker,date):
    # async with semaphore:  # This will limit the number of concurrent get_ta calls

    date = str_YYMMDD_HHMM_to_datetime(date)
    start = (date - timedelta(days=5)).strftime("%Y-%m-%d %H:%M")
    end = (date).strftime("%Y-%m-%d %H:%M")
    headers = {f"Authorization": f"Bearer {real_auth}", "Accept": "application/json"}
    #start date must be on or after 2023-11-20 00:00:00.?  as of f 1218
    print(start,end)
    time_sale_response = await fetch(session, "https://api.tradier.com/v1/markets/timesales",
                                     params={"symbol": ticker, "interval": "1min", "start": start, "end": end,
                                             "session_filter": "all"}, headers=headers)

    if time_sale_response and "series" in time_sale_response and "data" in time_sale_response["series"]:
        df = pd.DataFrame(time_sale_response["series"]["data"]).set_index("time")
    else:
        logger.warning(f"Bad time_sale_response data. {ticker},{date}")
        print(
            f"Bad time_sale_response data. {ticker},{date}")

    df.index = pd.to_datetime(df.index)

    def safe_calculation(df, column_name, calculation_function, *args, **kwargs):
        """
        Safely perform a calculation for a DataFrame and handle exceptions.
        If an exception occurs, the specified column is filled with NaN.
        """
        try:

            df[column_name] = calculation_function(*args, **kwargs)
        except Exception as e:
            print(column_name, ticker, e)
            df[column_name] = pd.NA  # or pd.nan
            raise

    # Usage of safe_calculation function for each indicator
    safe_calculation(df, "AwesomeOsc", ta.momentum.awesome_oscillator, high=df["high"], low=df["low"], window1=1,
                     window2=5, fillna=False)
    safe_calculation(df, "AwesomeOsc5_34", ta.momentum.awesome_oscillator, high=df["high"], low=df["low"], window1=5,
                     window2=34, fillna=False)
    # For MACD
    macd_object = ta.trend.MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9, fillna=False)
    signal_line = macd_object.macd_signal
    safe_calculation(df, "MACD", macd_object.macd)
    safe_calculation(df, "Signal_Line", signal_line)

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
    return this_minute_ta_frame


async def perform_operations(session,
                             ticker,
                             last_adj_close,
                             current_price,
                             StockLastTradeTime,
                             YYMMDD, optionchain_df
                             ):
    results = []
    price_change_percent = ((current_price - last_adj_close) / last_adj_close) * 100
    # TODO could pass in optionchain.

    groups = optionchain_df.groupby("ExpDate")
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
        optionchain_df["strike_lac_diff"] = group["Strike"].apply(lambda x: abs(x - last_adj_close))
        ###############################
        if not group.empty:
            smallest_change_from_lac = optionchain_df["strike_lac_diff"].abs().idxmin()
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
        closest_higher_strike1 = strikeindex_abovebelow[5]
        closest_higher_strike2 = strikeindex_abovebelow[6]
        closest_higher_strike3 = strikeindex_abovebelow[7]
        closest_higher_strike4 = strikeindex_abovebelow[8]

        #################

        ##Gettting pcr-vol for individual strikes above/below CP(closest to current price strike)
        def calculate_pcr_ratio(put_data, call_data):
            if np.isnan(put_data) or np.isnan(call_data) or call_data == 0:
                return np.inf if call_data == 0 and put_data != 0 and not np.isnan(put_data) else np.nan
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

        NIV_CurrentStrike =  get_ratio_and_iv(closest_strike_currentprice)[
            4] if closest_strike_currentprice is not None else np.nan
        NIV_1HigherStrike =  get_ratio_and_iv(closest_higher_strike1)[
            4] if closest_higher_strike1 is not None else np.nan
        NIV_2HigherStrike =  get_ratio_and_iv(closest_higher_strike2)[
            4] if closest_higher_strike2 is not None else np.nan
        NIV_3HigherStrike =  get_ratio_and_iv(closest_higher_strike3)[
            4] if closest_higher_strike3 is not None else np.nan
        NIV_4HigherStrike =  get_ratio_and_iv(closest_higher_strike4)[
            4] if closest_higher_strike4 is not None else np.nan
        NIV_1LowerStrike =  get_ratio_and_iv(closest_lower_strike1)[4] if closest_lower_strike1 is not None else np.nan
        NIV_2LowerStrike =  get_ratio_and_iv(closest_lower_strike2)[4] if closest_lower_strike2 is not None else np.nan
        NIV_3LowerStrike =  get_ratio_and_iv(closest_lower_strike3)[4] if closest_lower_strike3 is not None else np.nan
        NIV_4LowerStrike =  get_ratio_and_iv(closest_lower_strike4)[4] if closest_lower_strike4 is not None else np.nan

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
    processed_data_df = pd.DataFrame(results)
    this_minute_ta_frame = await get_ta(session, ticker,date=StockLastTradeTime)
    processed_data_df["RSI"] = this_minute_ta_frame["RSI"]
    processed_data_df["RSI2"] = this_minute_ta_frame["RSI2"]
    processed_data_df["RSI14"] = this_minute_ta_frame["RSI14"]
    processed_data_df["AwesomeOsc"] = this_minute_ta_frame["AwesomeOsc"]
    processed_data_df["MACD"] = this_minute_ta_frame["MACD"]
    processed_data_df["Signal_Line"] = this_minute_ta_frame["Signal_Line"]

    # Calculate 50-Day EMA
    processed_data_df["EMA_50"] = this_minute_ta_frame["EMA_50"]

    # Calculate 200-Day EMA
    processed_data_df["EMA_200"] = this_minute_ta_frame["EMA_200"]

    processed_data_df["AwesomeOsc5_34"] = this_minute_ta_frame[
        "AwesomeOsc5_34"]  # this_minute_ta_frame['exp_date'] = '230427.0'
    processed_data_df["MACD"] = this_minute_ta_frame["MACD"]
    processed_data_df["Signal_Line"] = this_minute_ta_frame["Signal_Line"]

    # Calculate 50-Day EMA
    processed_data_df["EMA_50"] = this_minute_ta_frame["EMA_50"]

    # Calculate 200-Day EMA
    processed_data_df["EMA_200"] = this_minute_ta_frame["EMA_200"]

    output_dir = Path(f"data/ProcessedData/{ticker}/{YYMMDD}/")
    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    output_dir_dailyminutes = Path(f"data/DailyMinutes/{ticker}")
    output_file_dailyminutes = Path(f"data/DailyMinutes/{ticker}/{ticker}_{YYMMDD}.csv")#TODO changed theis from .../ which is the actual dir.
    # output_dir_dailyminutes_w_algo_results = Path(f"data/DailyMinutes_w_algo_results/{ticker}")
    # output_dir_dailyminutes_w_algo_results.mkdir(mode=0o755, parents=True, exist_ok=True)
    output_dir_dailyminutes.mkdir(mode=0o755, parents=True, exist_ok=True)

    def replace_inf(df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return  # No numeric columns to process

        epsilon = 1e-7  # small value

        for col in numeric_cols:
            is_pos_inf = df[col] == np.inf
            is_neg_inf = df[col] == -np.inf

            if is_pos_inf.any():
                finite_max = df.loc[~is_pos_inf, col].max() + epsilon
                df.loc[is_pos_inf, col] = finite_max

            if is_neg_inf.any():
                finite_min = df.loc[~is_neg_inf, col].min() - epsilon
                df.loc[is_neg_inf & (finite_min < 0), col] = finite_min * 1.5
                df.loc[is_neg_inf & (finite_min >= 0), col] = finite_min

    # Use the function
    if output_file_dailyminutes.exists():
        dailyminutes_df = pd.read_csv(output_file_dailyminutes)
        dailyminutes_df = dailyminutes_df.drop_duplicates(subset="LastTradeTime")
        dailyminutes_df = pd.concat([dailyminutes_df, processed_data_df.head(1)], ignore_index=True)
        replace_inf(dailyminutes_df)  # It will only run if inf or -inf values are present
    else:
        dailyminutes_df = pd.concat([processed_data_df.head(1)], ignore_index=True)
        replace_inf(dailyminutes_df)  # It will only run if inf or -inf values are present

    dailyminutes_df.to_csv(output_file_dailyminutes, index=False)

    try:
        processed_data_df.to_csv(f"data/ProcessedData/{ticker}/{YYMMDD}/{ticker}_{StockLastTradeTime}.csv", mode="x",
                                 index=False)
    ###TODO could use this fileexists as a trigger to tell algos not to send(market clesed)
    except FileExistsError as e:
        print(f"data/ProcessedData/{ticker}/{YYMMDD}/{ticker}_{StockLastTradeTime}.csv", "File Already Exists.")

    return (
        optionchain_df,
        dailyminutes_df,
        processed_data_df,
        ticker,

    )



async def get_previous_trading_day(date_str):
    # Create a calendar for NYSE
    nyse = mcal.get_calendar('NYSE')

    # Convert string to datetime object
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    # Set the date range
    start_date = date_obj - timedelta(days=30)
    end_date = date_obj

    # Fetch the trading days
    trading_days = nyse.valid_days(start_date=start_date, end_date=end_date)

    # Find the last trading day before your date of interest
    last_trading_day = None
    for day in reversed(trading_days):
        if day.date() < end_date.date():
            last_trading_day = day
            break

    return last_trading_day.strftime("%Y-%m-%d") if last_trading_day else None

async def get_last_adjusted_close(api_key, symbol, date):
    url = "https://api.tradier.com/v1/markets/history"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }
    params = {
        'symbol': symbol,
        'start': date,
        'end': date
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if data is not None:
            days_data = data.get('history', {}).get('day', [])
            return days_data['close']
        else:
            return "Response JSON is None."
    else:
        return f"Error fetching data: {response.status_code}"


from datetime import datetime
import pytz

async def format_stock_last_trade_time(input_str, timezone='America/New_York', time_difference=5):
    # Split the filename by underscore

    # Convert the input string to datetime object (assuming the input is in UTC)
    stock_last_trade_time_utc = datetime.strptime(input_str, "%y%m%d_%H%M")
    print(stock_last_trade_time_utc)
    stock_last_trade_time_utc = pytz.utc.localize(stock_last_trade_time_utc)
    print("stock_last_trade_time_utc21",stock_last_trade_time_utc)
    # Convert the datetime object to Eastern Time
    tz = pytz.timezone(timezone)
    print(tz)
    stock_last_trade_time_et = stock_last_trade_time_utc.astimezone(tz)+ timedelta(hours=time_difference)
    # stock_last_trade_time_et = stock_last_trade_time_utc + timedelta(hours=time_difference)
    return stock_last_trade_time_et
#TODO why this has -4 offset insetad of 5
async def fetch_current_price(ticker, stock_last_trade_time):
    # Set the date range to the specific day you are interested in
    print("stock last trade time",stock_last_trade_time)
    date_str = stock_last_trade_time.strftime("%Y-%m-%d")
    end_date = stock_last_trade_time + timedelta(days=1)
    end_date_str = end_date.strftime("%Y-%m-%d")

    # Load data from yfinance
    data = yf.download(ticker, start=date_str, end=end_date_str, interval='1m')
    if data.empty:
        return None
    print(data)
    # Check if the exact minute is available in the data
    if stock_last_trade_time in data.index:
        return data.loc[stock_last_trade_time]['Adj Close']
    else:
        return None



# Example usage
api_key = real_auth
optionchain_dir = r"C:\Users\del_p\PycharmProjects\StockAlgoV2\data\optionchain"
async def main():
    async with aiohttp.ClientSession() as session:

        for ticker_folder in os.listdir(optionchain_dir):
            # if ticker_folder in ["TSLA","SPY","GOOGL"]:
                print(ticker_folder)
                ticker_dir = os.path.join(optionchain_dir, ticker_folder)
                if os.path.isdir(ticker_dir):  # Ensure it's a directory
                    for date_folder in os.listdir(ticker_dir):  # Iterate over files/directories in ticker_dir
                        date_dir = os.path.join(ticker_dir, date_folder)
                        if os.path.isdir(date_dir):  # Check if it's a directory
                            print(date_folder)
                            print(date_dir)

                        formatted_date = datetime.strptime(date_folder, "%y%m%d").strftime("%Y-%m-%d")
                        YYMMDD = date_folder
                        print(YYMMDD)
                        previous_trading_day =  await get_previous_trading_day(formatted_date)
                        # Fetch last adjusted close for this previous trading day
                        #TOTO get from the dict.
                        try:
                            last_adj_close = await get_last_adjusted_close(api_key, ticker_folder, previous_trading_day)
                        except Exception as e:
                            print(e)
                            continue
                        print(f"Ticker: {date_folder}, Last Trading Day: {previous_trading_day}, Last Adjusted Close: {last_adj_close}")

                        for optionchain_min_csv_file in os.listdir(date_dir):
                            print("optionchain file",optionchain_min_csv_file)
                            #min_csv files in forlmat "SPY_230915_1116.csv" for ticker_YYMMDD_HHMM
                            parts = optionchain_min_csv_file.split('_')
                            # Concatenate to form YYMMDD_HHMM
                            if len(parts) >= 3:
                                 YYMMDD_HHMM = parts[1] + "_" + parts[2].split('.')[0]  # Removes the .csv extension
                            if parts[1] != YYMMDD  or int(parts[2].split('.')[0]) > 1600 or int(parts[2].split('.')[0]) < 930:#cut off at open and close.
                                continue
                            formatted_stock_last_trade_time = await format_stock_last_trade_time(YYMMDD_HHMM)
                            # if not is_within_last_30_days(formatted_stock_last_trade_time):
                            if not is_within_target_dates(formatted_stock_last_trade_time):
                                print(f"Date {formatted_stock_last_trade_time} is not within the targeted dates.. Skipping download.")
                                continue
                            else:
                                current_price = await fetch_current_price(ticker_folder, formatted_stock_last_trade_time)
                                print(f"Adjusted current price for {ticker_folder} at {optionchain_min_csv_file}: {current_price}")
                                optionchain_df = pd.read_csv(os.path.join(date_dir, optionchain_min_csv_file))
                                if current_price is not None and last_adj_close is not None:
                                    try:
                                        await perform_operations(session, ticker_folder, last_adj_close, current_price, YYMMDD_HHMM, YYMMDD, optionchain_df)
                                    except Exception as e:
                                        logger.exception(msg=e)
                                        continue
                                else:
                                    break
asyncio.get_event_loop().run_until_complete(main())
