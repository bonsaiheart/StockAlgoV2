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
    (
        upordown,
        CorP,
        contractStrike,
        contract_price,
        IB_option_date,
        formatted_time,
        formatted_time_HR_MIN_only,
    ) = await get_contract_details(optionchain_df, processeddata_df, ticker, model_name)

    callorput = "call" if CorP == "C" else "put"
    timetill_expectedprofit, seconds_till_expectedprofit = check_interval_match(
        model_name
    )
    orderRef = ticker + "_" + model_name + "_" + formatted_time_HR_MIN_only

    if order_manager.ib.isConnected:
        try:
            parent_trade_success = await place_option_order_sync(
                CorP,
                ticker,
                IB_option_date,
                contractStrike,
                contract_price,
                orderRef=ticker + "_" + model_name + "_" + formatted_time_HR_MIN_only,
                quantity=10,
                take_profit_percent=option_take_profit_percent,
                trail_stop_percent=option_trail_stop_percent,
            )
        except Exception as trade_e:
            logger.exception(
                f"An error occurred while creating option order task {trade_e}."
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
    # try:
    #     await send_notifications.email_me_string(model_name, CorP, ticker)
    # except Exception as e:
    #     print(f"Email error {e}.")
    #     logger.exception(f"An error occurred while creating email task {e}")
    #

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
                        if signal_sums[pair_name] > 0.1:
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

            # Execute for individual model if not part of a pair or not executed as part of a pair
            if not part_of_pair or model_name not in executed_models:
                if result > 0.0:
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


# TODO make it look for pairs first somehow?  store all orders, and take best?
async def get_contract_details(optionchain_df, processeddata_df, ticker, model_name):
    # Extract the closest expiration date and strikes list
    closest_exp_date = processeddata_df["ExpDate"].iloc[0]
    closest_strikes_list = processeddata_df[
        "Closest Strike Above/Below(below to above,4 each) list"
    ].iloc[0]
    # Format the expiration date for IB
    # low to high, indesx 4 is closest current strike
    date_object = datetime.strptime(str(closest_exp_date), "%y%m%d")
    # print(date_object)
    formatted_contract_date = date_object.strftime("%y%m%d")

    IB_option_date = date_object.strftime("%Y%m%d")
    # print(IB_option_date)
    # Determine the type of contract based on the model name
    CorP = (
        "C" if "Buy" in model_name or "Up" in model_name or "up" in model_name else "P"
    )

    # Calculate the contract strike and price
    # contractStrike = closest_strikes_list[1] if CorP == "C" else closest_strikes_list[-2]
    contractStrike = closest_strikes_list[4] if CorP == "C" else closest_strikes_list[4]

    # has for mat 410.5
    formatted_contract_strike = contractStrike * 1000
    # print(contractStrike)
    contract_symbol = (
        f"{ticker}{formatted_contract_date}{CorP}{int(formatted_contract_strike):08d}"
    )
    # print(contract_symbol)
    # print("wowowow", optionchain_df.loc[optionchain_df["c_contractSymbol"] == contract_symbol]["Call_LastPrice"])
    # Get the last price for the contract
    if CorP == "C":
        contract_price = optionchain_df.loc[
            optionchain_df["c_contractSymbol"] == contract_symbol
        ]["Call_LastPrice"].values[0]
    else:
        contract_price = optionchain_df.loc[
            optionchain_df["p_contractSymbol"] == contract_symbol
        ]["Put_LastPrice"].values[0]

    # Determine the direction for the notification message
    upordown = "up" if CorP == "C" else "down"

    # Get the current time formatted for the notification message
    current_time = datetime.now()

    # Full date and time format
    formatted_time = current_time.strftime("%y%m%d %H:%M EST")

    # Only time format
    formatted_time_HMonly = current_time.strftime("%H:%M")
    return (
        upordown,
        CorP,
        contractStrike,
        contract_price,
        IB_option_date,
        formatted_time,
        formatted_time_HMonly,
    )


# Main execution
# if __name__ == "__main__":
#     # Replace the placeholder paths with the actual paths to your CSV files
#     asyncio.run(actions(
#         'path_to_optionchain.csv',
#         'path_to_dailyminutes.csv',
#         'path_to_processeddata.csv',
#         'ticker_symbol',
#         'current_price'
#     ))
