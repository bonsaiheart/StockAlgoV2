import requests
from datetime import datetime, timedelta
import pandas as pd
import ta
from pathlib import Path
import numpy as np
import PrivateData.tradier_info

# import webullAPI

paper_acc = PrivateData.tradier_info.paper_acc
paper_auth = PrivateData.tradier_info.paper_auth
real_acc = PrivateData.tradier_info.real_acc
real_auth = PrivateData.tradier_info.real_auth

###TODO time and sales (will be used for awesome ind. and ta.
YYMMDD = datetime.today().strftime("%y%m%d")


def get_options_data(ticker):
    start = (datetime.today() - timedelta(days=5)).strftime("%Y-%m-%d %H:%M")
    end = datetime.today().strftime("%Y-%m-%d %H:%M")
    response = requests.get('https://api.tradier.com/v1/markets/timesales',
                            params={'symbol': ticker, 'interval': '1min', 'start': start, 'end': end,
                                    'session_filter': 'all'},
                            headers={f'Authorization': f'Bearer {real_auth}', 'Accept': 'application/json'}
                            )
    json_response = response.json()
    # print(response.status_code)
    # print(json_response)
    df = pd.DataFrame(json_response['series']['data']).set_index('time')
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
    # print(this_minute_ta_frame)
    # df.to_csv('df.csv')

    # quote
    response = requests.get('https://api.tradier.com/v1/markets/quotes',
                            params={'symbols': ticker, 'greeks': 'false'},
                            headers={f'Authorization': f'Bearer {real_auth}', 'Accept': 'application/json'}
                            )
    json_response = response.json()
    # print(response.status_code)
    # print(json_response)
    # print(json_response)
    quote_df = pd.DataFrame.from_dict(json_response['quotes']['quote'], orient='index').T
    LAC = quote_df.at[0, 'prevclose']

    CurrentPrice = quote_df.at[0, 'last']
    price_change_percent = quote_df['change_percentage'][0]
    StockLastTradeTime = quote_df['trade_date'][0]
    StockLastTradeTime = StockLastTradeTime / 1000  # Convert milliseconds to seconds
    StockLastTradeTime = datetime.fromtimestamp(StockLastTradeTime).strftime("%y%m%d_%H%M")
    print(f"${ticker} last Trade Time: {StockLastTradeTime}")
    print(f"LAC: ${LAC}")
    print(f"Current Price: ${CurrentPrice}")

    # quote_df.to_csv('tradierquote.csv')

    ###TODO finish making this df look just like yfinance df.
    response = requests.get('https://api.tradier.com/v1/markets/options/expirations',
                            params={'symbol': ticker, 'includeAllRoots': 'true', 'strikes': 'true'},
                            headers={'Authorization': f'Bearer {real_auth}', 'Accept': 'application/json'}
                            )
    json_response = response.json()

    expirations = json_response['expirations']['expiration']
    expiration_dates = [expiration['date'] for expiration in expirations]

    closest_exp_date = expiration_dates[0]

    callsChain = []
    putsChain = []

    for exp_date in expiration_dates:
        response = requests.get('https://api.tradier.com/v1/markets/options/chains',
                                params={'symbol': ticker, 'expiration': exp_date, 'greeks': 'true'},
                                headers={'Authorization': f'Bearer {real_auth}', 'Accept': 'application/json'}
                                )
        json_response = response.json()
        # print(response.status_code)
        # print("Option Chain: ",json_response)
        optionchain_df = pd.DataFrame(json_response['options']['option'])
        grouped = optionchain_df.groupby('option_type')
        call_group = grouped.get_group('call').copy()
        put_group = grouped.get_group('put').copy()

        # print(call_group.columns)
        # print(put_group.columns)
        callsChain.append(call_group)
        putsChain.append(put_group)

    # combined_call_put_optionchain_df = pd.concat([call_group.set_index('strike'), put_group.set_index('strike')], axis=1, keys=['call', 'put'])

    ###CALLS OPS

    # calls_df["lastTrade"] = pd.to_datetime(calls_df["lastTradeDate"])
    # calls_df["lastTrade"] = calls_df["lastTrade"].dt.strftime("%y%m%d_%H%M")

    calls_df = pd.concat(callsChain, ignore_index=True)
    # for greek in calls_df['greeks']:
    #     print(greek.get('mid_iv'))
    calls_df["expDate"] = calls_df["symbol"].str[-15:-9]
    calls_df["dollarsFromStrike"] = abs(calls_df["strike"] - LAC)
    calls_df["dollarsFromStrikexOI"] = calls_df["dollarsFromStrike"] * calls_df["open_interest"]
    calls_df["lastContractPricexOI"] = calls_df["last"] * calls_df["open_interest"]
    calls_df["impliedVolatility"] = calls_df['greeks'].str.get('mid_iv')
    # print("callsiv",calls_df['impliedVolatility'])

    calls_df.rename(
        columns={
            "symbol": "c_contractSymbol",
            "trade_date": "c_lastTrade",
            "last": "c_lastPrice",
            "bid": "c_bid",
            "ask": "c_ask",
            "change": "c_change",
            "change_percentage": "c_percentChange",
            "volume": "c_volume",
            "open_interest": "c_openInterest",
            "greeks": "c_greeks",
            "impliedVolatility": "c_impliedVolatility",
            # "inTheMoney": "c_inTheMoney",
            # "lastTrade": "c_lastTrade",
            "dollarsFromStrike": "c_dollarsFromStrike",
            "dollarsFromStrikexOI": "c_dollarsFromStrikexOI",
            "lastContractPricexOI": "c_lastContractPricexOI",
        },
        inplace=True,
    )
    ###PUTS OPS

    puts_df = pd.concat(putsChain, ignore_index=True)
    # puts_df["lastTrade"] = pd.to_datetime(puts_df["lastTradeDate"])
    # puts_df["lastTrade"] = puts_df["lastTrade"].dt.strftime("%y%m%d_%H%M")
    puts_df["expDate"] = puts_df["symbol"].str[-15:-9]
    ###TODO for puts, use current price - strike i think.
    puts_df["dollarsFromStrike"] = abs(puts_df["strike"] - LAC)
    puts_df["dollarsFromStrikexOI"] = puts_df["dollarsFromStrike"] * puts_df["open_interest"]
    puts_df["lastContractPricexOI"] = puts_df["last"] * puts_df["open_interest"]
    puts_df["impliedVolatility"] = puts_df['greeks'].str.get('mid_iv')
    # print("PUTSsiv", puts_df['impliedVolatility'])
    puts_df.rename(
        columns={
            "symbol": "p_contractSymbol",
            "trade_date": "p_lastTrade",
            "last": "p_lastPrice",
            "bid": "p_bid",
            "ask": "p_ask",
            "change": "p_change",
            "change_percentage": "p_percentChange",
            "volume": "p_volume",
            "open_interest": "p_openInterest",
            "impliedVolatility": "p_impliedVolatility",
            "greeks": "p_greeks",
            # "inTheMoney": "p_inTheMoney",
            # "lastTrade": "p_lastTrade",
            "dollarsFromStrike": "p_dollarsFromStrike",
            "dollarsFromStrikexOI": "p_dollarsFromStrikexOI",
            "lastContractPricexOI": "p_lastContractPricexOI",
        },
        inplace=True,
    )

    combined = pd.merge(puts_df, calls_df, on=["expDate", "strike"])

    combined = combined[
        [
            "expDate",
            "strike",
            "c_contractSymbol",
            "c_lastTrade",
            "c_lastPrice",
            "c_bid",
            "c_ask",
            "c_change",
            "c_percentChange",
            "c_volume",
            "c_openInterest",
            "c_impliedVolatility",
            "c_greeks",
            # "c_inTheMoney",
            "c_lastTrade",
            "c_dollarsFromStrike",
            "c_dollarsFromStrikexOI",
            "c_lastContractPricexOI",
            "p_contractSymbol",
            "p_lastTrade",
            "p_lastPrice",
            "p_bid",
            "p_ask",
            "p_change",
            "p_percentChange",
            "p_volume",
            "p_openInterest",
            "p_impliedVolatility",
            "c_greeks",
            # "p_inTheMoney",
            "p_lastTrade",
            "p_dollarsFromStrike",
            "p_dollarsFromStrikexOI",
            "p_lastContractPricexOI",
        ]
    ]
    combined.rename(
        columns={
            "expDate": "ExpDate",
            "strike": "Strike",
            "c_lastPrice": "Call_LastPrice",
            "c_percentChange": "Call_PercentChange",
            "c_volume": "Call_Volume",
            "c_openInterest": "Call_OI",
            "c_impliedVolatility": "Call_IV",
            "c_dollarsFromStrike": "Calls_dollarsFromStrike",
            "c_dollarsFromStrikexOI": "Calls_dollarsFromStrikeXoi",
            "c_lastContractPricexOI": "Calls_lastPriceXoi",
            "p_lastPrice": "Put_LastPrice",
            "p_volume": "Put_Volume",
            "p_openInterest": "Put_OI",
            "p_impliedVolatility": "Put_IV",
            "p_dollarsFromStrike": "Puts_dollarsFromStrike",
            "p_dollarsFromStrikexOI": "Puts_dollarsFromStrikeXoi",
            "p_lastContractPricexOI": "Puts_lastPriceXoi",
        },
        inplace=True,
    )

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
    # print(type(LAC), type(CurrentPrice), type(price_change_percent), type(StockLastTradeTime),
    #       type(this_minute_ta_frame), type(closest_exp_date))
    # print(LAC, CurrentPrice,"fffff", price_change_percent,"asdfdasf", StockLastTradeTime, this_minute_ta_frame, closest_exp_date)
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
    closest_exp_date = expiration_dates[0]
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
        calls_DFSxOI_dict = (
            group.loc[group["Calls_dollarsFromStrikeXoi"] >= 0, ["Strike", "Calls_dollarsFromStrikeXoi"]]
            .set_index("Strike")
            .to_dict()
        )
        puts_DFSxOI_dict = (
            group.loc[group["Puts_dollarsFromStrikeXoi"] >= 0, ["Strike", "Puts_dollarsFromStrikeXoi"]]
            .set_index("Strike")
            .to_dict()
        )

        ITM_CallsVol = group.loc[(group["Strike"] <= current_price), "Call_Volume"].sum()
        ITM_PutsVol = group.loc[(group["Strike"] >= current_price), "Put_Volume"].sum()
        ITM_CallsOI = group.loc[(group["Strike"] <= current_price), "Call_OI"].sum()
        ITM_PutsOI = group.loc[(group["Strike"] >= current_price), "Put_OI"].sum()
        ITM_OI = ITM_CallsOI + ITM_PutsOI
        all_CallsVol = group.Call_Volume.sum()
        all_PutsVol = group.Put_Volume.sum()

        all_CallsOI = group.Call_OI.sum()
        all_PutsOI = group.Put_OI.sum()
        all_OI = all_PutsOI + all_CallsOI

        ITM_Call_IV = group.loc[(group["Strike"] <= current_price), "Call_IV"].sum()
        ITM_Put_IV = group.loc[(group["Strike"] >= current_price), "Put_IV"].sum()
        Call_IV = group["Call_IV"].sum()
        Put_IV = group["Put_IV"].sum()
        ITM_Avg_Net_IV = ITM_Call_IV - ITM_Put_IV
        Net_IV = Call_IV - Put_IV

        if all_CallsVol != 0 and not np.isnan(all_CallsVol):
            PC_Ratio_Vol = all_PutsVol / all_CallsVol
        else:
            PC_Ratio_Vol = np.nan

        if ITM_CallsVol != 0 and not np.isnan(ITM_CallsVol):
            ITM_PC_Ratio_Vol = ITM_PutsVol / ITM_CallsVol
        else:
            ITM_PC_Ratio_Vol = np.nan

        if all_CallsOI != 0 and not np.isnan(all_CallsOI):
            PC_Ratio_OI = all_PutsOI / all_CallsOI
        else:
            PC_Ratio_OI = np.nan

        if ITM_CallsOI != 0 and not np.isnan(ITM_CallsOI):
            ITM_PC_Ratio_OI = ITM_PutsOI / ITM_CallsOI
        else:
            ITM_PC_Ratio_OI = np.nan

        DFSxOI_dict = (
            group.loc[group["Puts_dollarsFromStrikeXoi"] >= 0, ["Strike", "Puts_dollarsFromStrikeXoi"]]
            .set_index("Strike")
            .to_dict()
        )

        # All_PC_Ratio =
        # Money_weighted_PC_Ratio =
        ###TODO figure out WHEN this needs to run... probalby after 6pm eastern and before mrkt open.  remove otm
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

        ### FINDING CLOSEST STRIKE TO LAC
        # target number from column A
        # calculate difference between target and each value in column B
        data["strike_lac_diff"] = group["Strike"].apply(lambda x: abs(x - last_adj_close))
        # find index of row with smallest difference

        if not group.empty:
            smallest_change_from_lac = data["strike_lac_diff"].abs().idxmin()
            closest_strike_lac = group.loc[smallest_change_from_lac, "Strike"]

            current_price_index = group["Strike"].sub(current_price).abs().idxmin()
            # print("currentprice index", current_price_index)
            ###RETURNS index of strike closest to CP

            higher_strike_index1 = current_price_index + 1

            higher_strike_index2 = current_price_index + 2
            higher_strike_index3 = current_price_index + 3
            higher_strike_index4 = current_price_index + 4
            # get the index of the row with the closest lower strike
            lower_strike_index1 = current_price_index - 1
            lower_strike_index2 = current_price_index - 2
            lower_strike_index3 = current_price_index - 3

            lower_strike_index4 = current_price_index - 4
            # get the strikes for the closest higher and lower strikes
            try:

                closest_strike_currentprice = group.loc[current_price_index, "Strike"]

            except KeyError as e:
                closest_strike_currentprice = None

                ("KeyError:", e)
            try:
                closest_higher_strike1 = group.loc[higher_strike_index1, "Strike"]
                # print(closest_higher_strike1)
            except KeyError:
                closest_higher_strike1 = None

            try:
                closest_higher_strike2 = group.loc[higher_strike_index2, "Strike"]
            except KeyError:
                closest_higher_strike2 = None

            try:
                closest_higher_strike3 = group.loc[higher_strike_index3, "Strike"]
                # print('closesthigherstrike3',closest_higher_strike3)
            except KeyError:
                closest_higher_strike3 = None

            try:
                closest_higher_strike4 = group.loc[higher_strike_index4, "Strike"]
            except KeyError:
                closest_higher_strike4 = None

            try:
                closest_lower_strike1 = group.loc[lower_strike_index1, "Strike"]
            except KeyError:
                closest_lower_strike1 = None

            try:
                closest_lower_strike2 = group.loc[lower_strike_index2, "Strike"]
            except KeyError:
                closest_lower_strike2 = None

            try:
                closest_lower_strike3 = group.loc[lower_strike_index3, "Strike"]
            except KeyError:
                closest_lower_strike3 = None

            try:
                closest_lower_strike4 = group.loc[lower_strike_index4, "Strike"]
                # print("closest_lowerstrike4",closest_lower_strike4)
            except KeyError:
                closest_lower_strike4 = None

        else:
            # handle empty dataframe here
            closest_strike_lac = None
            closest_strike_currentprice = None
            closest_higher_strike1 = None
            closest_higher_strike2 = None
            closest_higher_strike3 = None
            closest_higher_strike4 = None
            closest_lower_strike1 = None
            closest_lower_strike2 = None
            closest_lower_strike3 = None
            closest_lower_strike4 = None

        # closest_strike_currentprice_dict[exp_date] = closest_strike_currentprice

        strikeindex_abovebelow = [
            closest_lower_strike4,
            closest_lower_strike3,
            closest_lower_strike2,
            closest_lower_strike1,
            closest_strike_currentprice,
            closest_higher_strike1,
            closest_higher_strike2,
            closest_higher_strike3,
            closest_higher_strike4,
        ]
        strike_PCRv_dict = {}
        strike_PCRoi_dict = {}
        strike_ITMPCRv_dict = {}
        strike_ITMPCRoi_dict = {}
        ##Gettting pcr-vol for individual strikes above/below CP(closest to current price strike)
        for strike in strikeindex_abovebelow:
            if strike == None:
                strike_PCRv_dict[strike] = np.nan
            else:
                strikeindex_abovebelowput_volume = group.loc[group["Strike"] == strike, "Put_Volume"].values[0]
                strikeindex_abovebelowcall_volume = group.loc[group["Strike"] == strike, "Call_Volume"].values[0]
                if strikeindex_abovebelowcall_volume == 0:
                    strike_PCRv_dict[strike] = np.nan
                else:
                    strike_PCRv_dict[strike] = strikeindex_abovebelowput_volume / strikeindex_abovebelowcall_volume
        ##Get pcr-oi for individual strikes above/below CP strieki
        for strike in strikeindex_abovebelow:
            if strike == None:
                strike_PCRoi_dict[strike] = np.nan
            else:
                strikeindex_abovebelowput_oi = group.loc[group["Strike"] == strike, "Put_OI"].values[0]
                strikeindex_abovebelowcall_oi = group.loc[group["Strike"] == strike, "Call_OI"].values[0]
                if strikeindex_abovebelowcall_oi == 0:
                    strike_PCRoi_dict[strike] = np.nan
                else:
                    strike_PCRoi_dict[strike] = strikeindex_abovebelowput_oi / strikeindex_abovebelowcall_oi

        ##CP PCR V/oi
        cp_put_vol = group.loc[group["Strike"] == closest_strike_currentprice, "Put_Volume"].values[0]
        cp_call_vol = group.loc[group["Strike"] == closest_strike_currentprice, "Call_Volume"].values[0]
        if np.isnan(cp_put_vol) or np.isnan(cp_call_vol) or cp_call_vol == 0:
            PCRv_cp_strike = np.nan
        else:
            PCRv_cp_strike = cp_put_vol / cp_call_vol

        cp_put_OI = group.loc[group["Strike"] == closest_strike_currentprice, "Put_OI"].values[0]
        cp_call_OI = group.loc[group["Strike"] == closest_strike_currentprice, "Call_OI"].values[0]
        if np.isnan(cp_put_OI) or np.isnan(cp_call_OI) or cp_call_OI == 0:
            PCRoi_cp_strike = np.nan
        else:
            PCRoi_cp_strike = cp_put_OI / cp_call_OI

        ###MP V PCR
        mp_put_vol = group.loc[group["Strike"] == max_pain, "Put_Volume"].values[0]
        mp_call_vol = group.loc[group["Strike"] == max_pain, "Call_Volume"].values[0]

        if np.isnan(mp_put_vol) or np.isnan(mp_call_vol) or mp_call_vol == 0:
            PC_Ratio_Vol_atMP = np.nan
        else:
            PC_Ratio_Vol_atMP = mp_put_vol / mp_call_vol
        ##MP OI PCR
        mp_put_OI = group.loc[group["Strike"] == max_pain, "Put_OI"].values[0]
        mp_call_OI = group.loc[group["Strike"] == max_pain, "Call_OI"].values[0]
        if np.isnan(mp_put_OI) or np.isnan(mp_call_OI) or mp_call_OI == 0:
            PC_Ratio_OI_atMP = np.nan
        else:
            PC_Ratio_OI_atMP = mp_put_OI / mp_call_OI

        ####ITM for four up and four down.

        for strikeabovebelow in strikeindex_abovebelow:
            if strikeabovebelow == None:
                strike_ITMPCRv_dict[strikeabovebelow] = np.nan
            else:
                strike_ITMPCRvput_volume = group.loc[(group["Strike"] >= strikeabovebelow), "Put_Volume"].sum()
                strike_ITMPCRvcall_volume = group.loc[(group["Strike"] <= strikeabovebelow), "Call_Volume"].sum()
                if strike_ITMPCRvcall_volume == 0:
                    strike_ITMPCRv_dict[strikeabovebelow] = np.nan
                else:
                    strike_ITMPCRv_dict[strikeabovebelow] = strike_ITMPCRvput_volume / strike_ITMPCRvcall_volume

        for strikeabovebelow in strikeindex_abovebelow:
            if strikeabovebelow == None:
                strike_ITMPCRoi_dict[strikeabovebelow] = np.nan
            else:
                strike_ITMPCRoiput_volume = group.loc[(group["Strike"] >= strikeabovebelow), "Put_OI"].sum()
                strike_ITMPCRoicall_volume = group.loc[(group["Strike"] <= strikeabovebelow), "Call_OI"].sum()
                if strike_ITMPCRoicall_volume == 0:
                    strike_ITMPCRoi_dict[strikeabovebelow] = np.nan
                else:
                    strike_ITMPCRoi_dict[strikeabovebelow] = strike_ITMPCRoiput_volume / strike_ITMPCRoicall_volume

        ###TODO use above/below lac strikes instead of just closest.
        ##LAC V/oi PCR
        lac_put_vol = group.loc[group["Strike"] == closest_strike_lac, "Put_Volume"].values[0]
        lac_call_vol = group.loc[group["Strike"] == closest_strike_lac, "Call_Volume"].values[0]
        if np.isnan(lac_put_vol) or np.isnan(lac_call_vol) or lac_call_vol == 0:
            PC_Ratio_Vol_Closest_Strike_LAC = np.nan
        else:
            PC_Ratio_Vol_Closest_Strike_LAC = lac_put_vol / lac_call_vol

        lac_put_OI = group.loc[group["Strike"] == closest_strike_lac, "Put_OI"].values[0]
        lac_call_OI = group.loc[group["Strike"] == closest_strike_lac, "Call_OI"].values[0]
        if np.isnan(lac_put_OI) or np.isnan(lac_call_OI) or lac_call_OI == 0:
            PC_Ratio_OI_Closest_Strike_LAC = np.nan
        else:
            PC_Ratio_OI_Closest_Strike_LAC = lac_put_OI / lac_call_OI

        if np.isnan(PC_Ratio_Vol_atMP) or np.isnan(PC_Ratio_OI_atMP) or PC_Ratio_OI_atMP == 0:
            PCR_vol_OI_at_MP = np.nan
        else:
            PCR_vol_OI_at_MP = round((PC_Ratio_Vol_atMP / PC_Ratio_OI_atMP), 3)

        if (
                np.isnan(PC_Ratio_Vol_Closest_Strike_LAC)
                or np.isnan(PC_Ratio_OI_Closest_Strike_LAC)
                or PC_Ratio_OI_Closest_Strike_LAC == 0
        ):
            PCR_vol_OI_at_LAC = np.nan
        else:
            PCR_vol_OI_at_LAC = round((PC_Ratio_Vol_Closest_Strike_LAC / PC_Ratio_OI_Closest_Strike_LAC), 3)

        Net_Call_IV_at_MP = group.loc[(group["Strike"] == max_pain), "Call_IV"].sum()
        Net_Put_IV_at_MP = group.loc[(group["Strike"] == max_pain), "Put_IV"].sum()
        Net_IV_at_MP = Net_Call_IV_at_MP - Net_Put_IV_at_MP
        NIV_CurrentStrike = (group.loc[(group["Strike"] == closest_strike_currentprice), "Call_IV"].sum()) - (
            group.loc[(group["Strike"] == closest_strike_currentprice), "Put_IV"].sum()
        )
        NIV_1HigherStrike = (group.loc[(group["Strike"] == closest_higher_strike1), "Call_IV"].sum()) - (
            group.loc[(group["Strike"] == closest_higher_strike1), "Put_IV"].sum()
        )
        NIV_2HigherStrike = (group.loc[(group["Strike"] == closest_higher_strike2), "Call_IV"].sum()) - (
            group.loc[(group["Strike"] == closest_higher_strike2), "Put_IV"].sum()
        )
        NIV_3HigherStrike = (group.loc[(group["Strike"] == closest_higher_strike3), "Call_IV"].sum()) - (
            group.loc[(group["Strike"] == closest_higher_strike3), "Put_IV"].sum()
        )
        NIV_4HigherStrike = (group.loc[(group["Strike"] == closest_higher_strike4), "Call_IV"].sum()) - (
            group.loc[(group["Strike"] == closest_higher_strike4), "Put_IV"].sum()
        )
        NIV_1LowerStrike = (group.loc[(group["Strike"] == closest_lower_strike1), "Call_IV"].sum()) - (
            group.loc[(group["Strike"] == closest_lower_strike1), "Put_IV"].sum()
        )
        NIV_2LowerStrike = (group.loc[(group["Strike"] == closest_lower_strike2), "Call_IV"].sum()) - (
            group.loc[(group["Strike"] == closest_lower_strike2), "Put_IV"].sum()
        )
        NIV_3LowerStrike = (group.loc[(group["Strike"] == closest_lower_strike3), "Call_IV"].sum()) - (
            group.loc[(group["Strike"] == closest_lower_strike3), "Put_IV"].sum()
        )
        NIV_4LowerStrike = (group.loc[(group["Strike"] == closest_lower_strike4), "Call_IV"].sum()) - (
            group.loc[(group["Strike"] == closest_lower_strike4), "Put_IV"].sum()
        )

        Call_IV_Closest_Strike_LAC = group.loc[(group["Strike"] == closest_strike_lac), "Call_IV"].sum()
        Put_IV_Closest_Strike_LAC = group.loc[(group["Strike"] == closest_strike_lac), "Put_IV"].sum()
        Net_IV_Closest_Strike_LAC = Call_IV_Closest_Strike_LAC - Put_IV_Closest_Strike_LAC
        ###TODO error handling for scalar divide of zero denominator

        Bonsai_Ratio = ((ITM_PutsVol / all_PutsVol) * (ITM_PutsOI / all_PutsOI)) / (
                (ITM_CallsVol / all_CallsVol) * (ITM_CallsOI / all_CallsOI)
        )
        Bonsai2_Ratio = (
            # (all_PutsOI == 0 or ITM_PutsOI == 0 or all_CallsOI == 0 or ITM_CallsVol == 0 or ITM_CallsOI == 0)
            # and float("inf")
            # or
                ((all_PutsVol / ITM_PutsVol) / (all_PutsOI / ITM_PutsOI))
                * ((all_CallsVol / ITM_CallsVol) / (all_CallsOI / ITM_CallsOI))
        )

        # Calculate the percentage change###TODO figure out how to look at bonsai %change, will need to transform to timesheet.
        # if last_Bonsai_Ratio is not None:
        #     bonsai_percent_change = ((Bonsai_Ratio - last_Bonsai_Ratio) / last_Bonsai_Ratio) * 100
        # else:
        #     bonsai_percent_change = 0.0
        # if last_Bonsai_Ratio_2 is not None:
        #     bonsai2_percent_change = ((Bonsai2_Ratio - last_Bonsai_Ratio_2) / last_Bonsai_Ratio_2) * 100
        # else:
        #     bonsai2_percent_change = 0.0

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
                # 'Bonsai_2 %change': bonsai2_percent_change,
                # 'Maximum Pain % -LAC': round((last_adj_close-max_pain)/max_pain,2),
                # 'Implied % Move -previousMP': 0,
                # 'Implied % Move-LAC': round(implied_percentage_move, 2),
                # TODO ITM contract $ %
                # PCR
                "PCR-Vol": round(PC_Ratio_Vol, 3),
                "PCR-OI": round(PC_Ratio_OI, 3),
                # 'PCR Vol/OI': round(PC_Ratio_Vol / PC_Ratio_OI, 3),
                # 'ITM PCR Vol/OI': float('inf') if ITM_PC_Ratio_OI == 0 else 0 if ITM_PC_Ratio_Vol == 0 else round(ITM_PC_Ratio_Vol / ITM_PC_Ratio_OI, 3),
                # 'PCR @MP Vol/OI ': round((PC_Ratio_Vol_atMP / PC_Ratio_OI_atMP), 3),
                # 'PCR @LAC Vol/OI ': round(PCR_vol_OI_at_LAC, 3),
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
                "NIV highers(-)lowers1-2": (
                                                   NIV_1HigherStrike + NIV_2HigherStrike) - (
                                                   NIV_1LowerStrike + NIV_2LowerStrike),

                "NIV highers(-)lowers1-4": (
                                                       NIV_1HigherStrike + NIV_2HigherStrike + NIV_3HigherStrike + NIV_4HigherStrike) - (
                                                       NIV_1LowerStrike + NIV_2LowerStrike + NIV_3LowerStrike + NIV_4LowerStrike),
                "NIV 1-2 % from mean": (
                                               ((NIV_1HigherStrike + NIV_2HigherStrike) - (
                                                       NIV_1LowerStrike + NIV_2LowerStrike)) / ((
                                                                                                            NIV_1HigherStrike + NIV_2HigherStrike + NIV_1LowerStrike + NIV_2LowerStrike) / 4)) * 100,

                "NIV 1-4 % from mean": (
                                               (
                                                           NIV_1HigherStrike + NIV_2HigherStrike + NIV_3HigherStrike + NIV_4HigherStrike) - (
                                                       NIV_1LowerStrike + NIV_2LowerStrike + NIV_3LowerStrike + NIV_4LowerStrike) / (
                                                           (
                                                                       NIV_1HigherStrike + NIV_2HigherStrike + NIV_3HigherStrike + NIV_4HigherStrike + NIV_1LowerStrike + NIV_2LowerStrike + NIV_3LowerStrike + NIV_4LowerStrike) / 8)) * 100,
                ##TODO swap (/) with result = np.divide(x, y)
                "Net_IV/OI": Net_IV / all_OI,
                "Net ITM_IV/ITM_OI": ITM_Avg_Net_IV / ITM_OI,
                "Closest Strike to CP": closest_strike_currentprice,
                "Closest Strike Above/Below(below to above,4 each) list": strikeindex_abovebelow
            }
        )

        # if len(results) >= 2:
        #     # print(results[-2]['Maximum Pain'])
        #     implied_move_from_last_MP = ((max_pain - results[-2]['Maximum Pain']) / results[-2]['Maximum Pain']) * 100
        #     # print(implied_move_fom_last_MP)
        #     results[-1]['Implied % Move -previousMP'] = round(implied_move_from_last_MP,2)

        ##TODO Signal strength 1-3, bull/bear.
        #
        # if results('Bonsai Ratio') < 1:
        #     if results('Maximum Pain') < results('Current Stock Price'):

        # if results('Bonsai Ratio') > 1:

    df = pd.DataFrame(results)
    # print("hello",type(df["Closest Strike Above/Below(below to above,4 each) list"][0]))
    df["RSI"] = this_minute_ta_frame["RSI"]
    df["RSI2"] = this_minute_ta_frame["RSI2"]
    df["RSI14"] = this_minute_ta_frame["RSI14"]
    df["AwesomeOsc"] = this_minute_ta_frame["AwesomeOsc"]

    df["AwesomeOsc5_34"] = this_minute_ta_frame["AwesomeOsc5_34"]# this_minute_ta_frame['exp_date'] = '230427.0'
    # df = pd.concat([this_minute_ta_frame,df])
    # df['']
    output_dir = Path(f"data/ProcessedData/{ticker}/{YYMMDD}/")

    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    output_dir_dailyminutes = Path(f"data/DailyMinutes/{ticker}")
    output_dir_dailyminutes_w_algo_results = Path(f"data/DailyMinutes_w_algo_results/{ticker}")
    output_file_dailyminutes = Path(f"data/DailyMinutes/{ticker}/{ticker}_{YYMMDD}.csv")
    output_file_dailyminutes_w_algo_results =Path(f"data/DailyMinutes_w_algo_results/{ticker}/{ticker}_{YYMMDD}.csv")
    output_dir_dailyminutes.mkdir(mode=0o755, parents=True, exist_ok=True)
    output_dir_dailyminutes_w_algo_results.mkdir(mode=0o755, parents=True, exist_ok=True)
    if output_file_dailyminutes.exists():
        # Load the existing DataFrame from the file
        dailyminutes = pd.read_csv(output_file_dailyminutes)
        dailyminutes= dailyminutes.dropna().drop_duplicates(subset='LastTradeTime')
        dailyminutes = pd.concat([dailyminutes,df.head(1)], ignore_index=True)
        # print(dailyminutes)
        dailyminutes.to_csv(output_file_dailyminutes, index=False)
        dailyminutes.to_csv(output_file_dailyminutes_w_algo_results, index=False)
    else:

        # dailyminutes = df.head(1)
        dailyminutes = pd.concat([df.head(1)], ignore_index=True)
        # Append the new row to the existing DataFrame

        # Save the updated DataFrame back to the file

        dailyminutes.to_csv(output_file_dailyminutes, index=False)
        dailyminutes.to_csv(output_file_dailyminutes_w_algo_results, index=False)

    try:
        df.to_csv(f"data/ProcessedData/{ticker}/{YYMMDD}/{ticker}_{StockLastTradeTime}.csv", mode="x", index=False)
###TODO could use this fileexists as a trigger to tell algos not to send(market clesed)
    except FileExistsError:
        df.to_csv(f"data/ProcessedData/{ticker}/{YYMMDD}/{ticker}_{StockLastTradeTime}.csv", index=False)
    return (
        f"data/optionchain/{ticker}/{YYMMDD}/{ticker}_{StockLastTradeTime}.csv",
        f"data/DailyMinutes/{ticker}/{ticker}_{YYMMDD}.csv", output_file_dailyminutes_w_algo_results,
         df,
        ##df is processeddata
        ticker,
    )
