import asyncio
import re
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

import IB.ibAPI
from Strategy_Testing.Trained_Models import trained_minute_models, pytorch_trained_minute_models
from UTILITIES.Send_Notifications import send_notifications as send_notifications
from UTILITIES.logger_config import logger


async def place_option_order_sync(CorP, ticker, exp, strike, contract_current_price, orderRef,quantity=10,
                                  custom_takeprofit=None, custom_trailamount=None, loop=None):
    try:
        await IB.ibAPI.placeOptionBracketOrder(CorP, ticker, exp, strike, contract_current_price, quantity,
                                               orderRef, custom_takeprofit, custom_trailamount)
    except Exception as e:
        logger.error(f"An error occurred in place_option_order_sync. {ticker} : {e}", exc_info=True)


async def place_buy_order_sync(ticker, current_price,  orderRef,quantity=10, custom_takeprofit=None,
                               custom_trailamount=None, loop=None):
    try:
        await IB.ibAPI.placeBuyBracketOrder(ticker, current_price, quantity, orderRef, custom_takeprofit, custom_trailamount)
    except Exception as e:
        logger.error(f"An error occurred in place_buy_order_sync. {ticker} : {e}", exc_info=True)


def check_interval_match(model_name):
    pattern = re.compile(r"\d+(min|hr)")


    interval_match = pattern.search(model_name)
    if interval_match:
        time_till_expectedprofit = interval_match.group()
        if time_till_expectedprofit.endswith("min"):
            seconds_till_expectedprofit = int(time_till_expectedprofit[:-3]) * 60
        elif time_till_expectedprofit.endswith("hr"):
            seconds_till_expectedprofit = int(time_till_expectedprofit[:-2]) * 3600
        # elif time_till_expectedprofit.endswith("day"):
        #     seconds_till_expectedprofit = int(time_till_expectedprofit[:-3]) * 86400
        return time_till_expectedprofit, seconds_till_expectedprofit

    else:
        logger.error(f"Invalid model function name: {model_name}")


def get_contract(optionchain, processeddata, ticker, model_name):
    ###strikeindex_abovebelow is a list [lowest,3 lower,2 lower, 1 lower, 1 higher,2 higher,3 higher, 4 higher]
    now = datetime.now()
    formatted_time = now.strftime("%y%m%d %H:%M EST")
    expdates_strikes_dict = {}
    for exp_date, row in processeddata.iterrows():
        exp_date = row['ExpDate']
        closest_strikes_list = row["Closest Strike Above/Below(below to above,4 each) list"]
        expdates_strikes_dict[exp_date] = closest_strikes_list
    closest_exp_date = list(expdates_strikes_dict.keys())[0]
    strikeindex_closest_expdate = expdates_strikes_dict[closest_exp_date]
    optionchain = pd.read_csv(optionchain)
    # print(ticker, current_price)
    date_string = str(closest_exp_date)
    date_object = datetime.strptime(date_string, "%y%m%d")
    new_date_string = date_object.strftime("%y%m%d")
    IB_option_date = date_object.strftime("%Y%m%d")

    # Create a list of your variable names for indexing
    varnames = ["four_strike_below", "three_strike_below", "two_strike_below",
                "one_strike_below", "closest_strike", "one_strike_above",
                "two_strike_above", "three_strike_above", "four_strike_above"]

    # Initialize your dictionaries

    ##IB values for contract strikes is like 34.5 instead of 00034500
    ib_values = {}

    strike_values = {}

    # Now loop through your indices
    for idx, varname in enumerate(varnames):
        if strikeindex_closest_expdate[idx] != np.nan:
            ib_values[varname] = strikeindex_closest_expdate[idx]
            strike_values[varname] = int(strikeindex_closest_expdate[idx] * 1000)

    ###TODO add different exp date options in addition to diff strike optoins.
    ib_one_strike_below = ib_values["one_strike_below"]
    ib_one_strike_above = ib_values["one_strike_above"]

    one_strike_above_closest_cp_strike_int_num = strike_values["one_strike_above"]
    one_strike_below_closest_cp_strike_int_num = strike_values["one_strike_below"]
    # two_strike_above_closest_cp_strike_int_num = strike_values["two_strike_above"]
    # two_strike_below_closest_cp_strike_int_num = strike_values["two_strike_above"]
    # three_strike_above_closest_cp_strike_int_num = strike_values["three_strike_above"]
    # three_strike_below_closest_cp_strike_int_num = strike_values["three_strike_above"]
    # four_strike_above_closest_cp_strike_int_num = strike_values["four_strike_above"]
    # four_strike_below_closest_cp_strike_int_num = strike_values["four_strike_above"]
    closest_strike_exp_int_num = strike_values["closest_strike"]

    one_strike_above_CCPS = "{:08d}".format(one_strike_above_closest_cp_strike_int_num)
    one_strike_below_CCPS = "{:08d}".format(one_strike_below_closest_cp_strike_int_num)
    # two_strike_above_CCPS = "{:08d}".format(two_strike_above_closest_cp_strike_int_num)
    # two_strike_below_CCPS = "{:08d}".format(two_strike_below_closest_cp_strike_int_num)
    closest_contract_strike = "{:08d}".format(closest_strike_exp_int_num)
    # CCP_upone_call_contract = f"{ticker}{new_date_string}C{one_strike_above_CCPS}"
    CCP_upone_put_contract = f"{ticker}{new_date_string}P{one_strike_above_CCPS}"
    CCP_downone_call_contract = f"{ticker}{new_date_string}C{one_strike_below_CCPS}"
    # CCP_downone_put_contract = f"{ticker}{new_date_string}P{one_strike_below_CCPS}"
    # CCP_uptwo_call_contract = f"{ticker}{new_date_string}C{two_strike_above_CCPS}"
    # CCP_uptwo_put_contract = f"{ticker}{new_date_string}P{two_strike_above_CCPS}"
    # CCP_downtwo_call_contract = f"{ticker}{new_date_string}C{two_strike_below_CCPS}"
    # CCP_downtwo_put_contract = f"{ticker}{new_date_string}P{two_strike_below_CCPS}"
    CCP_call_contract = f"{ticker}{new_date_string}C{closest_contract_strike}"
    CCP_put_contract = f"{ticker}{new_date_string}P{closest_contract_strike}"

    CCP_Put_Price = optionchain.loc[optionchain["p_contractSymbol"] == CCP_put_contract]["Put_LastPrice"].values[0]
    CCP_Call_Price = optionchain.loc[optionchain["c_contractSymbol"] == CCP_call_contract]["Call_LastPrice"].values[0]
    try:
        DownOne_Call_Price = optionchain.loc[optionchain["c_contractSymbol"] == CCP_downone_call_contract][
            "Call_LastPrice"
        ].values[0]
        UpOne_Put_Price = optionchain.loc[optionchain["p_contractSymbol"] == CCP_upone_put_contract][
            "Put_LastPrice"
        ].values[0]
        # DownOne_Put_Price = optionchain.loc[optionchain["p_contractSymbol"] == CCP_downone_put_contract][
        #     "Put_LastPrice"
        # ].values[0]
        # UpOne_Call_Price = optionchain.loc[optionchain["c_contractSymbol"] == CCP_upone_call_contract][
        #     "Call_LastPrice"
        # ].values[0]
        # UpTwo_Call_Price = optionchain.loc[optionchain["c_contractSymbol"] == CCP_uptwo_call_contract][
        #     "Call_LastPrice"
        # ].values[0]
        # DownTwo_Put_Price = optionchain.loc[optionchain["p_contractSymbol"] == CCP_downtwo_put_contract][
        #     "Put_LastPrice"
        # ].values[0]
        # DownTwo_Call_Price = optionchain.loc[optionchain["c_contractSymbol"] == CCP_downtwo_call_contract][
        #     "Call_LastPrice"
        # ].values[0]
        # UpTwo_Put_Price = optionchain.loc[optionchain["p_contractSymbol"] == CCP_uptwo_put_contract][
        #     "Put_LastPrice"
        # ].values[0]

    except Exception as e:
        print(e)
        logger.error(f"An error occurred while getting options prices.{ticker},: {e}", exc_info=True)
        traceback.print_exc()
        pass
    if model_name.startswith("Buy"):
        CorP = "C"
    elif model_name.startswith("Sell"):
        CorP = "P"

    if CorP == "C":
        upordown = "up"
        callorput = "call"
        contractStrike = ib_one_strike_below
        contract_price = DownOne_Call_Price
    else:
        upordown = "down"
        callorput = "put"
        contractStrike = ib_one_strike_above
        contract_price = UpOne_Put_Price

    return upordown, CorP, callorput, contractStrike, contract_price, IB_option_date,formatted_time


async def actions(optionchain, dailyminutes, processeddata, ticker, current_price):
    model_list = [
        trained_minute_models.Buy_3hr_15minA2baseSPYA1,
        trained_minute_models.Sell_3hr_15minA2baseSPYA1,
        trained_minute_models.Buy_30min_15minA2SPY_A1_test,
        trained_minute_models.Sell_30min_15minA2SPY_A1_test,
        # trained_minute_models.Buy_2hr_RFSPYA2,
        # trained_minute_models.Sell_2hr_RFSPYA2,
        # trained_minute_models.Buy_2hr_RFSPYA1,
        # trained_minute_models.Sell_2hr_RFSPYA1,
        # pytorch_trained_minute_models.Buy_4hr_ffSPY230805,
        pytorch_trained_minute_models.Buy_1hr_ptminclassSPYA1,
        pytorch_trained_minute_models.Buy_3hr_PTminClassSPYA1,
        # pytorch_trained_minute_models.Buy_2hr_ptminclassSPYA2,
        pytorch_trained_minute_models.Buy_2hr_ptminclassSPYA1,
     ]
    dailyminutes_df = pd.read_csv(dailyminutes)

    # Store results in a dictionary for easy access
    results = {}
    for model in model_list:
        model_name = model.__name__
        model_output = model(dailyminutes_df)

        try:
            upordown, CorP, callorput, contractStrike, contract_price, IB_option_date,formatted_time = get_contract(optionchain,
                                                                                                     processeddata,
                                                                                                     ticker, model_name)

            if isinstance(model_output, tuple):
                (dailyminutes_df[model_name], custom_takeprofit, custom_trailingstop) = model_output
            else:
                dailyminutes_df[model_name], custom_takeprofit, custom_trailingstop = model_output, None, None
            results[model_name] = dailyminutes_df[model_name].iloc[-1]
            result = results[model_name]
            if result > 0.5:
                print(f'Positive result for {ticker} {model_name}')
                timetill_expectedprofit, seconds_till_expectedprofit = check_interval_match(model_name)
                loop = asyncio.get_event_loop()
                send_notifications.send_tweet_w_countdown_followup(
                    ticker,
                    current_price,
                    upordown,
                    f"${ticker} ${current_price}. {timetill_expectedprofit} to make money on a {callorput} #{model_name} {formatted_time}",
                    seconds_till_expectedprofit, model_name
                )
                # send_notifications.email_me_string(model_name, CorP, ticker)
                asyncio.gather(place_option_order_sync(CorP, ticker, IB_option_date, contractStrike, contract_price, f"{model_name}",10
                                                       , custom_takeprofit, custom_trailingstop, loop),
                               place_buy_order_sync(ticker, current_price,
                                                    model_name,10, None, None, loop))

        except Exception as e:
            logger.error(f"An error occurred after recieving positive result. {ticker}, {model_name}: {e}",
                         exc_info=True)
