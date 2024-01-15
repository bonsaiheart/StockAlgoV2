import asyncio
import re
from datetime import datetime
import pandas as pd
import IB.ibAPI

from Strategy_Testing.Trained_Models import (
    trained_minute_models,
    pytorch_trained_minute_models,
)
from UTILITIES.Send_Notifications import send_notifications
from UTILITIES.logger_config import logger

order_manager = IB.ibAPI.IBOrderManager()


# Utility function to handle errors
def log_error(location, ticker, model_name, exception):
    logger.error(
        f"An error occurred in {location}. {ticker}, {model_name}: {exception}",
        exc_info=True,
    )
    raise


# Order placement functions
async def place_option_order_sync(
    CorP,
    ticker,
    exp,
    strike,
    contract_current_price,
    orderRef,
    quantity,
    take_profit_percent,
    trail_stop_percent,
):
    try:
        await order_manager.placeOptionBracketOrder(
            CorP,
            ticker,
            exp,
            strike,
            contract_current_price,
            quantity,
            orderRef,
            take_profit_percent,
            trail_stop_percent,
        )
    except Exception as e:
        log_error("place_option_order_sync", ticker, orderRef, e)


async def place_buy_order_sync(
    ticker, current_price, orderRef, quantity, take_profit_percent, trail_stop_percent
):
    try:  # TODO make create task
        await IB.ibAPI.placeBuyBracketOrder(
            ticker,
            current_price,
            quantity,
            orderRef,
            take_profit_percent,
            trail_stop_percent,
        )
    except Exception as e:
        log_error("place_buy_order_sync", ticker, orderRef, e)


# Function to extract the time interval from the model name
# TODO note: only set up for minutes/hours.  Add days.
def check_interval_match(model_name):
    pattern = re.compile(r"(\d+)(min|hr)")
    interval_match = pattern.search(model_name)
    if interval_match:
        # Extract the numeric part and the unit (min or hr)
        num_part, unit_part = interval_match.groups()
        seconds = int(num_part) * (60 if unit_part == "min" else 3600)
        return f"{num_part}{unit_part}", seconds
    else:
        logger.error(f"Invalid model function name: {model_name}")
        return None, None


async def handle_model_result(
    model_name,
    ticker,
    current_price,
    optionchain_df,
    processeddata_df,
    option_take_profit_percent,
    option_trail_stop_percent,
):
    # Retrieve the contract details
    try:
        result = await get_contract_details(optionchain_df, processeddata_df, ticker, model_name)
        if result is not None:
            (
                upordown,
                CorP,
                contractStrike,
                contract_price,
                IB_option_date,
                formatted_time,
                formatted_time_mdHR_MIN_only,
            ) = result
            callorput = "call" if CorP == "C" else "put"
            timetill_expectedprofit, seconds_till_expectedprofit = check_interval_match(
                model_name
            )
            # orderRef = ticker + "_" + model_name + "_" + formatted_time_mdHR_MIN_only

            if order_manager.ib.isConnected:
                try:
                    parent_trade_success = await place_option_order_sync(
                        CorP,
                        ticker,
                        IB_option_date,
                        contractStrike,
                        contract_price,
                        orderRef=ticker + "_" + model_name + "_" + formatted_time_mdHR_MIN_only,
                        quantity=3,
                        take_profit_percent=option_take_profit_percent,
                        trail_stop_percent=option_trail_stop_percent,
                    )
                except Exception as trade_e:
                    logger.exception(
                        f"An error occurred while creating option order task {trade_e}.",exc_info=True
                    )
            # try:
            #     await send_notifications.send_tweet_w_countdown_followup(
            #         ticker,
            #         current_price,
            #         upordown,
            #         f"${ticker} ${current_price}. {timetill_expectedprofit} to make money on a {callorput} #{model_name} {formatted_time}",
            #         seconds_till_expectedprofit,
            #         model_name,
            #     )
            # except Exception as e:
            #     print(f"Tweet error {e}.")
            #     logger.exception(f"An error occurred while creating tweeting task {e}")
            try:
                await send_notifications.email_me_string(model_name, CorP, ticker)
            except Exception as e:
                print(f"Email error {e}.")
                logger.exception(f"An error occurred while creating email task {e}")

    except Exception as e:
        raise e



# Define model pairs that require a combined signal sum over 1.5 to trigger an action
model_pairs = {
    "Buy_2hr_ensemble_pair1": [
        pytorch_trained_minute_models.SPY_2hr_50pct_Down_PTNNclass.__name__,
        pytorch_trained_minute_models.Buy_20min_05pctup_ptclass_B1.__name__,
    ],
    # "ModelPair2": ["ModelName3", "ModelName4"],
    # Add more pairs as needed
}

# Initialize a dictionary to store signal sums for model pairs
signal_sums = {pair: 0 for pair in model_pairs}


# Main function to handle model actions
async def actions(
    optionchain_df, dailyminutes_df, processeddata_df, ticker, current_price
):
    # Load your data into dataframes
    # Initialize a variable to keep track of evaluated models
    evaluated_models = set()
    # Initialize a dictionary to track executed models
    executed_models = set()

    # Iterate over each model in your model list
    for model in get_model_list():
        model_name = model.__name__
        model_output = model(
            dailyminutes_df.tail(1)
        )  # error wehen trying to use taill.. either missing data(and used to return whatever row had all features.
        evaluated_models.add(model_name)
        # print(model_output)
        try:
            # TODO make each model return signal, so they can have individual thressholds for buy/sell.
            if isinstance(model_output, tuple):
                (
                    model_output_df,
                    stock_take_profit_percent,
                    stock_trail_stop_percent,
                    option_take_profit_percent,
                    option_trail_stop_percent,
                ) = model_output
            else:
                model_output_df = (
                    model_output  # Assuming `model_output` is a DataFrame or similar
                )
                stock_take_profit_percent = None
                stock_trail_stop_percent = None
                option_take_profit_percent = None
                option_trail_stop_percent = None
            # model_output_df.to_csv('test.csv')
            # dailyminutes_df.to_csv('test_dailymin.csv')

            result = model_output_df.iloc[-1]
            tail = model_output_df.tail(1)
            tail_str = ", ".join(map(str, tail))
            print(
                f"{ticker} {model_name} last 3 results: {tail_str}"
            )  # TODO could use this avg. to make order!

            # print('evaluated',evaluated_models)
            # Check if model is part of any pair
            part_of_pair = False

            # Check if model is part of any pair
            for pair_name, pair_models in model_pairs.items():
                if model_name in pair_models:
                    part_of_pair = True
                    signal_sums[pair_name] += result

                    if evaluated_models.issuperset(pair_models):
                        if signal_sums[pair_name] > 0.5:
                            # Execute for pair
                            logger.info(
                                f"!!!positive pair result? {pair_name}: {signal_sums[pair_name]}"
                            )
                            # TODO change so that it uses a tp/trail fitted for the pair/combo.
                            successfultrade = await handle_model_result(
                                pair_name,
                                ticker,
                                current_price,
                                optionchain_df,
                                processeddata_df,
                                option_take_profit_percent,
                                option_trail_stop_percent,
                            )
                            signal_sums[pair_name] = 0
                            executed_models.update(pair_models)
                        else:
                            signal_sums[pair_name] = 0
                        break  # Exit the loop after handling pair
#TODO can onlly have 1 positive per contract!!! b/c the cancelling orders etc will interfere...but only if not in orderdumy
            # Execute for individual model if not part of a pair or not executed as part of a pair
            if not part_of_pair or model_name not in executed_models:
                if result >=0:#TODO change this
                    try:
                        successfultrade = await handle_model_result(
                            model_name,
                            ticker,
                            current_price,
                            optionchain_df,
                            processeddata_df,
                            option_take_profit_percent,
                            option_trail_stop_percent,
                        )
                        if successfultrade:
                            return successfultrade
                    except Exception as e:
                        logger.exception(f"Error in handle_model_result. {e}")

        except Exception as e:
            log_error("actions", ticker, model_name, e)


# TODO use this log error funciton globally?

# Define a function to send notifications (assuming you have this functionality in the send_notifications module)

##TODO add clause to track price after buy signal.. if it drops x% then rebuy/reaverage.


# Define the model list, this assumes that the model list is predefined
def get_model_list():
    return [
        # Add the actual models here
        pytorch_trained_minute_models.Buy_3hr_PTminClassSPYA1,
        pytorch_trained_minute_models.SPY_2hr_50pct_Down_PTNNclass,
        # pytorch_trained_minute_models.Buy_20min_1pctup_ptclass_B1,
        # pytorch_trained_minute_models.Buy_20min_05pctup_ptclass_B1,
        pytorch_trained_minute_models._3hr_40pt_down_FeatSet2_shuf_exc_test_onlyvalloss,
    ]


# TODO make it look for pairs first somehow?  store all orders, and take best?   PROCESSED DATA IS NOT USED
from datetime import datetime

async def get_contract_details(optionchain_df, processeddatadf, ticker, model_name, target_delta=1, gamma_threshold=(0.035, 0.9), max_bid_ask_spread_percent=3,min_volume=500):
    # Determine the type of contract based on the model name

    CorP = "C" if "Buy" in model_name or "Up" in model_name or "up" in model_name else "P"

    # Define liquidity thresholds
    min_volume = min_volume

    # Extract delta and gamma values, calculate bid-ask spread as a percentage of the contract price
    if CorP == "C":
        optionchain_df['c_delta'] = optionchain_df['c_greeks'].apply(lambda x: x.get('delta') if isinstance(x, dict) else None)
        optionchain_df['c_gamma'] = optionchain_df['c_greeks'].apply(lambda x: x.get('gamma') if isinstance(x, dict) else None)
        price_column = "Call_LastPrice"
        delta_column = 'c_delta'
        gamma_column = 'c_gamma'
        volume_column = 'Call_Volume'
        bid_column = 'c_bid'
        ask_column = 'c_ask'
    else:
        optionchain_df['p_delta'] = optionchain_df['p_greeks'].apply(lambda x: x.get('delta') if isinstance(x, dict) else None)
        optionchain_df['p_gamma'] = optionchain_df['p_greeks'].apply(lambda x: x.get('gamma') if isinstance(x, dict) else None)
        price_column = "Put_LastPrice"
        delta_column = 'p_delta'
        gamma_column = 'p_gamma'
        volume_column = 'Put_Volume'
        bid_column = 'p_bid'
        ask_column = 'p_ask'

    # Calculate bid-ask spread percentage and apply liquidity filter
    optionchain_df['bid_ask_spread_percent'] = ((optionchain_df[ask_column] - optionchain_df[bid_column]) / optionchain_df[price_column]) * 100
    liquidity_filter = (optionchain_df[volume_column] >= min_volume) & (optionchain_df['bid_ask_spread_percent'] <= max_bid_ask_spread_percent)

    # Apply gamma filter
    gamma_filter = (optionchain_df[gamma_column] >= gamma_threshold[0]) & (optionchain_df[gamma_column] <= gamma_threshold[1])

    # Apply combined filters
    relevant_df = optionchain_df[liquidity_filter & gamma_filter].dropna(subset=[delta_column, gamma_column])

    # Ensure the DataFrame is not empty
    if relevant_df.empty:
        print("contractdetails relevant_df empty")
        return None

    # Find the contract with delta closest to the target delta
    adjusted_target_delta = target_delta if CorP == "C" else -target_delta
    relevant_df['delta_diff'] = (relevant_df[delta_column] - adjusted_target_delta).abs()
    closest_delta_row = relevant_df.iloc[relevant_df['delta_diff'].argsort()[:1]]

    # Extract the closest expiration date, strike, and delta
    closest_exp_date = closest_delta_row["ExpDate"].iloc[0]
    contractStrike = closest_delta_row["Strike"].iloc[0]
    contract_price = closest_delta_row[price_column].iloc[0]
    delta_value = closest_delta_row[delta_column].iloc[0]
    gamma_value = closest_delta_row[gamma_column].iloc[0]

    # Format the expiration date for IB
    date_object = datetime.strptime(str(closest_exp_date), "%y%m%d")
    IB_option_date = date_object.strftime("%Y%m%d")

    # Construct the contract symbol
    formatted_contract_strike = int(contractStrike * 1000)
    contract_symbol = f"{ticker}{date_object.strftime('%y%m%d')}{CorP}{formatted_contract_strike:08d}"

    # Determine the direction for the notification message
    upordown = "up" if CorP == "C" else "down"

    # Get the current time formatted for the notification message
    current_time = datetime.now()
    formatted_time = current_time.strftime("%y%m%d %H:%M EST")
    formatted_time_mdHMonly = current_time.strftime("%m%d_%H:%M")
    print (
        upordown,
        CorP,
        contractStrike,
        contract_price,
        IB_option_date,
        formatted_time,
        formatted_time_mdHMonly,

    )
    return (
        upordown,
        CorP,
        contractStrike,
        contract_price,
        IB_option_date,
        formatted_time,
        formatted_time_mdHMonly,

    )

#
# TODO: Implement functionality to find option pairs
# async def get_contract_details(optionchain_df, processeddatadf, ticker, model_name, target_delta=1, gamma_threshold=(0.05, 0.1)):
#     # Determine the type of contract based on the model name
#     CorP = "C" if "Buy" in model_name or "Up" in model_name or "up" in model_name else "P"
#
#     # Define liquidity thresholds
#     min_volume = 1000
#     max_bid_ask_spread = 0.03
#
#     # Extract delta and gamma values, and calculate bid-ask spread
#     delta_column = f'{CorP.lower()}_delta'
#     gamma_column = f'{CorP.lower()}_gamma'
#     bid_ask_spread_column = f'{CorP.lower()}_bid_ask_spread'
#
#     optionchain_df[delta_column] = optionchain_df[f'{CorP.lower()}_greeks'].apply(lambda x: x.get('delta') if isinstance(x, dict) else None)
#     optionchain_df[gamma_column] = optionchain_df[f'{CorP.lower()}_greeks'].apply(lambda x: x.get('gamma') if isinstance(x, dict) else None)
#     optionchain_df[bid_ask_spread_column] = optionchain_df[f'{CorP.lower()}_ask'] - optionchain_df[f'{CorP.lower()}_bid']
#
#     # Apply liquidity filter
#     volume_column = f'{"Put" if CorP == "P" else "Call"}_Volume'
#     liquidity_filter = (optionchain_df[f'{CorP.lower()}_lastTrade'] > 0) & (optionchain_df[volume_column] >= min_volume) & (optionchain_df[bid_ask_spread_column] <= max_bid_ask_spread)
#     relevant_df = optionchain_df[liquidity_filter].dropna(subset=[delta_column, gamma_column])
#
#     # Ensure the DataFrame is not empty
#     if relevant_df.empty:
#         print("contractdetails relevant_df empty")
#         return None
#
#     # Adjust the target delta for puts if necessary
#     adjusted_target_delta = target_delta if CorP == "C" else -target_delta
#
#     # Filter contracts based on the gamma threshold
#     gamma_min, gamma_max = gamma_threshold
#     gamma_filter = (relevant_df[gamma_column] >= gamma_min) & (relevant_df[gamma_column] <= gamma_max)
#     relevant_df = relevant_df[gamma_filter]
#
#     # Ensure there are contracts within the specified gamma range
#     if relevant_df.empty:
#         print("No contracts within the specified gamma range")
#         return None
#
#     # Find the contract with delta closest to the target delta
#     delta_diff_column = f'{CorP.lower()}_delta_diff'
#     relevant_df[delta_diff_column] = (relevant_df[delta_column] - adjusted_target_delta).abs()
#     closest_delta_row = relevant_df.iloc[relevant_df[delta_diff_column].argsort()[:1]]
#
#     # Retrieve gamma value for the same contract
#     gamma_value = closest_delta_row[gamma_column].iloc[0]
#
#     # Extract the closest expiration date, strike, and delta
#     closest_exp_date = closest_delta_row["ExpDate"].iloc[0]
#     contractStrike = closest_delta_row["Strike"].iloc[0]
#     lastprice_column = f'{"Put" if CorP == "P" else "Call"}_LastPrice'
#
#     contract_price = closest_delta_row[lastprice_column].iloc[0]
#     delta_value = closest_delta_row[delta_column].iloc[0]
#
#     # Format the expiration date for IB
#     date_object = datetime.strptime(str(closest_exp_date), "%y%m%d")
#     IB_option_date = date_object.strftime("%Y%m%d")
#
#     # Construct the contract symbol
#     formatted_contract_strike = int(contractStrike * 1000)
#     contract_symbol = f"{ticker}{date_object.strftime('%y%m%d')}{CorP}{formatted_contract_strike:08d}"
#     print(contract_symbol)
#
#     # Determine the direction for the notification message
#     upordown = "up" if CorP == "C" else "down"
#
#     # Get the current time formatted for the notification message
#     current_time = datetime.now()
#     formatted_time = current_time.strftime("%y%m%d %H:%M EST")
#     formatted_time_mdHMonly = current_time.strftime("%m%d_%H:%M")
#
#     print(upordown, CorP, contractStrike, contract_price, IB_option_date, formatted_time, formatted_time_mdHMonly)
#     print(type(upordown), type(CorP), type(contractStrike), type(contract_price), type(IB_option_date), type(formatted_time), type(formatted_time_mdHMonly))
#
#     return (
#         upordown, CorP, contractStrike, contract_price, IB_option_date, formatted_time, formatted_time_mdHMonly
#     )

# Additional code for finding option pairs can be added here
#
# async def get_contract_details(optionchain_df, processeddata_df, ticker, model_name):
#     # Extract the closest expiration date and strikes list
#     closest_exp_date = processeddata_df["ExpDate"].iloc[0]
#     closest_strikes_list = processeddata_df[
#         "Closest Strike Above/Below(below to above,4 each) list"
#     ].iloc[0]
#     # Format the expiration date for IB
#     # low to high, indesx 4 is closest current strike
#     date_object = datetime.strptime(str(closest_exp_date), "%y%m%d")
#     # print(date_object)
#     formatted_contract_date = date_object.strftime("%y%m%d")
#
#     IB_option_date = date_object.strftime("%Y%m%d")
#     # print(IB_option_date)
#     # Determine the type of contract based on the model name
#     CorP = (
#         "C" if "Buy" in model_name or "Up" in model_name or "up" in model_name else "P"
#     )
#
#     # Calculate the contract strike and price
#     # contractStrike = closest_strikes_list[1] if CorP == "C" else closest_strikes_list[-2]
#     contractStrike = closest_strikes_list[4] if CorP == "C" else closest_strikes_list[4]
#
#     # has for mat 410.5
#     formatted_contract_strike = contractStrike * 1000
#     # print(contractStrike)
#     contract_symbol = (
#         f"{ticker}{formatted_contract_date}{CorP}{int(formatted_contract_strike):08d}"
#     )
#     print(contract_symbol)
#     # print("wowowow", optionchain_df.loc[optionchain_df["c_contractSymbol"] == contract_symbol]["Call_LastPrice"])
#     # Get the last price for the contract
#     # profit_loss = (delta * price_change) + (0.5 * gamma * price_change**2) - (theta * time_decay) + (vega * vol_change)
#     if CorP == "C":
#         contract_price = optionchain_df.loc[
#             optionchain_df["c_contractSymbol"] == contract_symbol ]["Call_LastPrice"].values[0]
#         # contract_greeks = optionchain_df[f"{contract_symbol}"][""]
#         contract_greeksdict = optionchain_df.loc[
#             optionchain_df["c_contractSymbol"] == contract_symbol, "p_greeks"].values[0]
#         delta_value = contract_greeksdict.get('delta')
#         print(delta_value,"Delta")
#
#     else:
#         contract_price = optionchain_df.loc[
#             optionchain_df["p_contractSymbol"] == contract_symbol
#         ]["Put_LastPrice"].values[0]
#         contract_greeksdict = optionchain_df.loc[
#             optionchain_df["p_contractSymbol"] == contract_symbol, "c_greeks"].values[0]
#         delta_value = contract_greeksdict.get('delta')
#         print(delta_value,"Delta")
#     # Determine the direction for the notification message
#     upordown = "up" if CorP == "C" else "down"
#
#     # Get the current time formatted for the notification message
#     current_time = datetime.now()
#
#     # Full date and time format
#     formatted_time = current_time.strftime("%y%m%d %H:%M EST")
#
#     # Only time format
#     formatted_time_mdHMonly = current_time.strftime("%m%d_%H:%M")
#
#     return (
#         upordown,
#         CorP,
#         contractStrike,
#         contract_price,
#         IB_option_date,
#         formatted_time,
#         formatted_time_mdHMonly,
#     )
#
#
