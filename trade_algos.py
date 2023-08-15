import asyncio
from UTILITIES.logger_config import logger
import traceback
import pandas as pd
from datetime import datetime
import numpy as np
from Strategy_Testing.Trained_Models import trained_minute_models, pytorch_trained_minute_models
import re
import asyncio
import threading
import IB.ibAPI
from UTILITIES.Send_Notifications import send_notifications as send_notifications




def place_option_order_sync(CorP, ticker, exp, strike, contract_current_price, quantity, orderRef,
                            custom_takeprofit=None,
                            custom_trailamount=None):
    print(CorP, ticker, exp, strike, contract_current_price, quantity, orderRef)
    loop = asyncio.new_event_loop()
    print("placeoptionordersync", custom_trailamount, custom_takeprofit)
    # Set the new event loop as the current one
    asyncio.set_event_loop(loop)
    if quantity == None:
        quantity = 10
    try:

        IB.ibAPI.placeOptionBracketOrder(CorP=CorP,
                                         ticker=ticker,
                                         exp=exp,
                                         strike=strike,
                                         contract_current_price=contract_current_price,
                                         quantity=quantity,
                                         orderRef=orderRef, custom_takeprofit=custom_takeprofit,
                                         custom_trailamount=custom_trailamount)
    except Exception as e:
        print(f"Error in placeoptionordersync: {traceback.format_exc()}")

        logger.error(f"An error occurred in trade_algos palceoptionordersync. {ticker} : {e}", exc_info=True)

    finally:
        loop.close()


def place_buy_order_sync(ticker, current_price,
                         quantity,
                         orderRef,
                         custom_takeprofit=None,
                         custom_trailamount=None):
    loop = asyncio.new_event_loop()
    print("buying stocks")
    # Set the new event loop as the current one
    asyncio.set_event_loop(loop)
    if quantity == None:
        quantity = 10
    try:
        IB.ibAPI.placeBuyBracketOrder(ticker, current_price, quantity=quantity, orderRef=orderRef,
                                      custom_takeprofit=None, custom_trailamount=None)
    except Exception as e:
        print(f"Error in placebuyordersync: {traceback.format_exc()}")
        logger.error(f"An error occurred in place_buy_order_sync. {ticker}: {e}", exc_info=True)

    finally:
        loop.close()


# Then use it in your main code like this:

# loop = asyncio.get_event_loop()
# loop.run_in_executor(None, place_order_sync, corP, ticker, exp, strike, contract_current_price, quantity, orderRef)
#

async def actions(optionchain, dailyminutes, processeddata, ticker, current_price):
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
    dailyminutes_df = pd.read_csv(dailyminutes)
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
    # three_strike_above_CCPS = "{:08d}".format(three_strike_above_closest_cp_strike_int_num)
    # three_strike_below_CCPS = "{:08d}".format(three_strike_below_closest_cp_strike_int_num)
    # four_strike_above_CCPS = "{:08d}".format(four_strike_above_closest_cp_strike_int_num)
    # four_strike_below_CCPS = "{:08d}".format(four_strike_below_closest_cp_strike_int_num)
    closest_contract_strike = "{:08d}".format(closest_strike_exp_int_num)
    # CCP_upone_call_contract = f"{ticker}{new_date_string}C{one_strike_above_CCPS}"
    CCP_upone_put_contract = f"{ticker}{new_date_string}P{one_strike_above_CCPS}"
    CCP_downone_call_contract = f"{ticker}{new_date_string}C{one_strike_below_CCPS}"
    # CCP_downone_put_contract = f"{ticker}{new_date_string}P{one_strike_below_CCPS}"
    # CCP_uptwo_call_contract = f"{ticker}{new_date_string}C{two_strike_above_CCPS}"
    # CCP_uptwo_put_contract = f"{ticker}{new_date_string}P{two_strike_above_CCPS}"
    # CCP_downtwo_call_contract = f"{ticker}{new_date_string}C{two_strike_below_CCPS}"
    # CCP_downtwo_put_contract = f"{ticker}{new_date_string}P{two_strike_below_CCPS}"
    # CCP_upthree_call_contract = f"{ticker}{new_date_string}C{three_strike_above_CCPS}"
    # CCP_upthree_put_contract = f"{ticker}{new_date_string}P{three_strike_above_CCPS}"
    # CCP_downthree_call_contract = f"{ticker}{new_date_string}C{three_strike_below_CCPS}"
    # CCP_downthree_put_contract = f"{ticker}{new_date_string}P{three_strike_below_CCPS}"
    # CCP_upfour_call_contract = f"{ticker}{new_date_string}C{four_strike_above_CCPS}"
    # CCP_upfour_put_contract = f"{ticker}{new_date_string}P{four_strike_above_CCPS}"
    # CCP_downfour_call_contract = f"{ticker}{new_date_string}C{four_strike_below_CCPS}"
    # CCP_downfour_put_contract = f"{ticker}{new_date_string}P{four_strike_below_CCPS}"
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
        # DownThree_Put_Price = optionchain.loc[optionchain["p_contractSymbol"] == CCP_downthree_put_contract][
        #     "Put_LastPrice"
        # ].values[0]
        # DownThree_Call_Price = optionchain.loc[optionchain["c_contractSymbol"] == CCP_downthree_call_contract][
        #     "Call_LastPrice"
        # ].values[0]
        # DownFour_Call_Price = optionchain.loc[optionchain["c_contractSymbol"] == CCP_downfour_call_contract][
        #     "Call_LastPrice"
        # ].values[0]
        # UpThree_Call_Price = optionchain.loc[optionchain["c_contractSymbol"] == CCP_upthree_call_contract][
        #     "Call_LastPrice"
        # ].values[0]
        # Upfour_Call_Price = optionchain.loc[optionchain["c_contractSymbol"] == CCP_upfour_call_contract][
        #     "Call_LastPrice"
        # ].values[0]
        #
        # DownFour_Put_Price = optionchain.loc[optionchain["p_contractSymbol"] == CCP_downfour_put_contract][
        #     "Put_LastPrice"
        # ].values[0]
        # UpThree_Put_Price = optionchain.loc[optionchain["p_contractSymbol"] == CCP_upthree_put_contract][
        #     "Put_LastPrice"
        # ].values[0]
        # Upfour_Put_Price = optionchain.loc[optionchain["p_contractSymbol"] == CCP_upfour_put_contract][
        #     "Put_LastPrice"
        # ].values[0]
    except Exception as e:
        print(e)
        logger.error(f"An error occurred while getting options prices.{ticker},: {e}", exc_info=True)
        traceback.print_exc()
        pass
    """These Models are classifications and only need a single frame(current frame)"""

    model_list = [
        pytorch_trained_minute_models.Buy_1hr_ptminclassSPYA1,
        pytorch_trained_minute_models.Buy_3hr_PTminClassSPYA1,
        # pytorch_trained_minute_models.Buy_2hr_ptminclassSPYA2,
        pytorch_trained_minute_models.Buy_2hr_ptminclassSPYA1,
        pytorch_trained_minute_models.Buy_1hr_ptmin1A1,
        # trained_minute_models.Buy_4hr_nnSPYA1,  ##made 3 out of 3, >.25% change! wow
        # trained_minute_models.Sell_4hr_nnSPYA1,
        trained_minute_models.Buy_2hr_gsmmcA1,
        trained_minute_models.Sell_2hr_gsmmcA1,
        # trained_minute_models.Buy_2hr_nnA2,  ##made 3 out of 3, >.25% change! wow
        # trained_minute_models.Sell_2hr_nnA2,
        # trained_minute_models.Buy_90min_nnA2,  # WORKS GREAT?
        # trained_minute_models.Sell_90min_nnA2,
        # trained_minute_models.Buy_90min_nnA1,  # WORKS GREAT?
        # trained_minute_models.Sell_90min_nnA1,
        # trained_minute_models.Buy_1hr_nnA1,  # WORKS GREAT?
        # trained_minute_models.Sell_1hr_nnA1,
        trained_minute_models.Buy_2hr_A1,  ##made 3 out of 3, >.25% change! wow
        trained_minute_models.Sell_2hr_A1,
        trained_minute_models.Buy_2hr_A2,  ##made 3 out of 3, >.25% change! wow
        trained_minute_models.Sell_2hr_A2,

        trained_minute_models.Buy_90min_A2,
        trained_minute_models.Sell_90min_A2,

        # trained_minute_models.Buy_90min_A1,
        # trained_minute_models.Sell_90min_A1,
        # trained_minute_models.Buy_90min_A2,
        # trained_minute_models.Sell_90min_A2,
        trained_minute_models.Buy_90min_A3,
        trained_minute_models.Sell_90min_A3,
        trained_minute_models.Buy_90min_A4,
        trained_minute_models.Sell_90min_A4,
        # trained_minute_models.Buy_90min_A5,
        # trained_minute_models.Sell_90min_A5,
        trained_minute_models.Buy_1hr_A9,
        trained_minute_models.Sell_1hr_A9,
        # trained_minute_models.Buy_1hr_A8,
        # trained_minute_models.Sell_1hr_A8,
        trained_minute_models.Buy_1hr_A7,
        trained_minute_models.Sell_1hr_A7,  # got 2 outt of 3, and when it works its >.1%

        # trained_minute_models.Buy_1hr_A6,
        # trained_minute_models.Sell_1hr_A6,

        # trained_minute_models.Buy_1hr_A5,
        # trained_minute_models.Sell_1hr_A5,

        # trained_minute_models.Buy_1hr_A4,
        # trained_minute_models.Sell_1hr_A4,

        trained_minute_models.Buy_1hr_A3,
        trained_minute_models.Sell_1hr_A3,

        trained_minute_models.Buy_1hr_A2,
        trained_minute_models.Sell_1hr_A2,

        trained_minute_models.Buy_1hr_A1,  # WORKS GREAT?
        trained_minute_models.Sell_1hr_A1,  ###didn't seem to work accurately enough
        # WORKS GREAT?
        # trained_minute_models.Buy_45min_A1,
        # trained_minute_models.Sell_45min_A1,# only works ~50%?

        # trained_minute_models.Buy_30min_A1,  # WORKS GREAT?
        # trained_minute_models.Sell_30min_A1,  # seems to work well, expect .03-.1 drop.

        # trained_minute_models.Buy_20min_A1,  # WORKS GREAT?
        # trained_minute_models.Sell_20min_A1,
        # WORKS GREAT?
        trained_minute_models.Buy_15min_A2,  # works well?
        trained_minute_models.Sell_15min_A2,  # not sure
        trained_minute_models.Buy_15min_A1,  ##A1 picks up more moves, but more false positives - and more big moves
        trained_minute_models.Sell_15min_A1,  ##A1 picks up more moves, but more false positives - and more big moves
    ]
    # TODO convert to tensorobject here, instead of for each model definition.
    # TODO add logic so that if close is <x hours, use next day strike.
    # Convert input data to tensor

    # Store results in a dictionary for easy access
    results = {}
    pattern = re.compile(r"\d+(min|hr)")

    for model in model_list:
        model_name = model.__name__
        model_output = model(dailyminutes_df)

        if isinstance(model_output, tuple):
            (dailyminutes_df[model_name], custom_takeprofit, custom_trailingstop) = model_output
            print(custom_takeprofit, custom_trailingstop, "istuple!!!!!!!!!!!!!!!!!!!!!1")
        else:
            dailyminutes_df[model_name], custom_takeprofit, custom_trailingstop = model_output, None, None
        results[model_name] = dailyminutes_df[model_name].iloc[-1]
        print(model_name, results[model_name])
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

        interval_match = re.search(pattern, model_name)

        if interval_match:
            timetill_expectedprofit = interval_match.group()
            seconds = (
                int(timetill_expectedprofit[:-3]) * 60
                if timetill_expectedprofit.endswith("min")
                else int(timetill_expectedprofit[:-2]) * 3600
            )
        else:
            print(f"Invalid model function name: {model_name}")
            continue

        # Access result from the dictionary
        result = results[model_name]
        if result > 0.5:
            send_notifications.email_me_string(model_name, CorP, ticker)
            try:
                # Place order or perform other actions specific to the action
                print(f"(options)Sending {model_name} to IB.")
                loop = asyncio.get_event_loop()
                # TODO get custom tp and ts to work
                loop.run_in_executor(None, place_option_order_sync, CorP, ticker, IB_option_date, contractStrike,
                                     contract_price, 10, f"{model_name}", custom_takeprofit, custom_trailingstop)
                print(f"(stock)Sending {model_name} to IB.")
                loop.run_in_executor(None, place_buy_order_sync, ticker, current_price, 10,
                                     model_name)

            except Exception as e:
                logger.error(f"An error occurred after recieving positive result. {ticker}, {model_name}: {e}", exc_info=True)
                print("Error occurred while placing order:", str(e))
            print("sending tweet")
            send_notifications.send_tweet_w_countdown_followup(
                ticker,
                current_price,
                upordown,
                f"${ticker} ${current_price}. {timetill_expectedprofit} to make money on a {callorput} #{model_name} {formatted_time}",
                seconds, model_name
            )

    Algo1 = int(
        (dailyminutes_df["B2/B1"].iloc[-1] > 500)
        and (dailyminutes_df["Bonsai Ratio"].iloc[-1] < 0.0001)
        and (dailyminutes_df["ITM PCRv Up2"].iloc[-1] < 0.01)
        and (dailyminutes_df["ITM PCRv Down2"].iloc[-1] < 5)
        and (dailyminutes_df["NIV 1-2 % from mean"].iloc[-1] > dailyminutes_df["NIV 1-4 % from mean"].iloc[-1] > 0)
    )
    if Algo1:
        send_notifications.email_me_string(
            "B2/B1> 500 and B1 < 0.0001 and ITM PCRv Up2 < 0.01 and ITM PCRv Down2 < 5 and NIV 1-2 % from mean > NIV 1-4 % from mean",
            "Notsure", ticker
        )
    # 1.15-(hold until) 0 and <0.0, hold call until .3   (hold them until the b1/b2 doubles/halves?) with conditions to make sure its profitable.
    Algo2 = int(
        (dailyminutes_df["B1/B2"].iloc[-1] > 1.15) and (dailyminutes_df["RSI"].iloc[-1] < 30)
    )
    if Algo2:
        send_notifications.email_me_string(
            "B1/B2 > 1.15) and RSI < 30", "Call", ticker
        )  ####THis one is good for a very short term peak before drop.  Maybe tighter profit/loss
    if dailyminutes_df["B1/B2"].iloc[-1] < 0.25 and dailyminutes_df["RSI"].iloc[-1] > 70:
        send_notifications.email_me_string(
            "dailyminutes_df['B1/B2'][-1] < 0.25 and dailyminutes_df['RSI'][-1]>77:", "Put", ticker
        )

#

#         ####THis one is good for a very short term peak before drop.  Maybe tighter profit/loss
#     if dailyminutes_df['RSI'].iloc[-1] > 80 and dailyminutes_df['RSI14'].iloc[-1]>75:

###JUST b1/b2
# if dailyminutes_df['B1/B2'].iloc[-1] > 1.15 :

# ####THis one is good for a very short term peak before drop.  Maybe tighter profit/loss
# if dailyminutes_df['B1/B2'].iloc[-1] < 0.25 :
#
# dailyminutes_df["NIV 1-2 % from mean & NIV 1-4 % from mean"] = (
#     (dailyminutes_df["NIV 1-2 % from mean"] < -100) & (dailyminutes_df["NIV 1-4 % from mean"] < -200)
# ).astype(int)
#
# dailyminutes_df["NIV 1-2 % from mean & NIV 1-4 % from mean"] = (
#     (dailyminutes_df["NIV 1-2 % from mean"] > 100) & (dailyminutes_df["NIV 1-4 % from mean"] > 200)
# ).astype(int)
#
# dailyminutes_df["NIV highers(-)lowers1-4"] = (dailyminutes_df["NIV highers(-)lowers1-4"] < -20).astype(int)
#
# dailyminutes_df["NIV highers(-)lowers1-4"] = (dailyminutes_df["NIV highers(-)lowers1-4"] > 20).astype(int)
#
# dailyminutes_df["ITM PCR-Vol & RSI"] = (
#     (dailyminutes_df["ITM PCR-Vol"] > 1.3) & (dailyminutes_df["RSI"] > 70)
# ).astype(int)
#
# dailyminutes_df["Bonsai Ratio & ITM PCR-Vol & RSI"] = (
#     (dailyminutes_df["Bonsai Ratio"] < 0.8) & (dailyminutes_df["ITM PCR-Vol"] < 0.8) & (dailyminutes_df["RSI"] < 30)
# ).astype(int)
#
# dailyminutes_df["Bonsai Ratio & ITM PCR-Vol & RSI"] = (
#     (dailyminutes_df["Bonsai Ratio"] > 1.5) & (dailyminutes_df["ITM PCR-Vol"] > 1.2) & (dailyminutes_df["RSI"] > 70)
# ).astype(int)
#
# dailyminutes_df["Bonsai Ratio < 0.7 & Net_IV < -50 & Net ITM IV > -41"] = (
#     (dailyminutes_df["Bonsai Ratio"] < 0.7)
#     & (dailyminutes_df["Net_IV"] < -50)
#     & (dailyminutes_df["Net ITM IV"] > -41)
# ).astype(int)
