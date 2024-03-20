import asyncio
import re
import IB.ibAPI
from Strategy_Testing.Trained_Models import (
    pytorch_trained_minute_models,
)
from UTILITIES.Send_Notifications import send_notifications
from UTILITIES.logger_config import logger

order_manager = IB.ibAPI.IBOrderManager()


# TODO more the model processing to executor processpool or no?  it would fit nicely at edn of calculations?
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
        print(orderRef, "tp/sl: ", take_profit_percent, "/", trail_stop_percent)
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
    try:
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


# TODO make the diff models use diff gamma/delta to find contract.


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
    option_take_profit_percent,
    option_trail_stop_percent,
    current_time,
):
    # Retrieve the contract details
    try:
        result = await get_contract_details(
            optionchain_df, ticker, model_name, current_time
        )
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

            # try:
            #     parent_trade_success = await place_option_order_sync(
            #         CorP,
            #         ticker,
            #         IB_option_date,
            #         contractStrike,
            #         contract_price,
            #         orderRef=ticker + "_" + model_name + "_" + formatted_time_mdHR_MIN_only,
            #         quantity=3,
            #         take_profit_percent=option_take_profit_percent,
            #         trail_stop_percent=option_trail_stop_percent,
            #     )
            # except Exception as trade_e:
            #     logger.exception(
            #         f"An error occurred while creating option order task {trade_e}.",exc_info=True
            #     )
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
            #     await send_notifications.email_me_string(
            #         model_name, current_price, ticker
            #     )
            # except Exception as e:
            #     print(f"Email error {e}.")
            #     logger.exception(f"An error occurred while creating email task {e}")
            if order_manager.ib.isConnected():
                orderRef = (
                    ticker + "_" + model_name + "_" + formatted_time_mdHR_MIN_only
                )
                print("ordermanager is connedted.")
                quantity = 1
                # print(orderRef)
                return (
                    CorP,
                    ticker,
                    IB_option_date,
                    contractStrike,
                    contract_price,
                    orderRef,
                    quantity,
                    option_take_profit_percent,
                    option_trail_stop_percent,
                )
        return None
    except Exception as e:
        raise e


#
# await place_option_order_sync(
#                     CorP,
#                     ticker,
#                     IB_option_date,
#                     contractStrike,
#                     contract_price,
#                     orderRef=ticker + "_" + model_name + "_" + formatted_time_mdHR_MIN_only,
#                     quantity=3,
#                     take_profit_percent=option_take_profit_percent,
#                     trail_stop_percent=option_trail_stop_percent,
#                 )

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
    optionchain_df,
    dailyminutes_df,
    processeddata_df,
    ticker,
    current_price,
    current_time,
):
    # Load your data into dataframes
    # Initialize a variable to keep track of evaluated models
    evaluated_models = set()
    # Initialize a dictionary to track executed models
    executed_models = set()
    potential_orders = []
    unique_orders = set()
    # Iterate over each model in your model list
    for model in get_model_list_for_ticker(ticker):
        try:
            model_name = model.__name__
            model_output = model(
                dailyminutes_df.tail(1)
            )  # error wehen trying to use taill.. either missing data(and used to return whatever row had all features.
            evaluated_models.add(model_name)
            # print(model_output)

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
            # for pair_name, pair_models in model_pairs.items():
            #     if model_name in pair_models:
            #         part_of_pair = True
            #         signal_sums[pair_name] += result
            #
            #         if evaluated_models.issuperset(pair_models):
            #             if signal_sums[pair_name] > 0.5:
            #                 # Execute for pair
            #                 logger.info(
            #                     f"!!!positive pair result? {pair_name}: {signal_sums[pair_name]}"
            #                 )
            #                 # TODO change so that it uses a tp/trail fitted for the pair/combo.
            #                 successfultrade = await handle_model_result(
            #                     pair_name,
            #                     ticker,
            #                     current_price,
            #                     optionchain_df,
            #                     processeddata_df,
            #                     option_take_profit_percent,
            #                     option_trail_stop_percent,current_time
            #                 )
            #                 signal_sums[pair_name] = 0
            #                 executed_models.update(pair_models)
            #             else:
            #                 signal_sums[pair_name] = 0
            #             break  # Exit the loop after handling pair
            # TODO can onlly have 1 positive per contract!!! b/c the cancelling orders etc will interfere...but only if not in orderdumy
            # Execute for individual model if not part of a pair or not executed as part of a pair
            if not part_of_pair or model_name not in executed_models:  # TODO or or and
                if result >= 0.5:  # TODO change this
                    try:
                        order_params = await handle_model_result(
                            model_name,
                            ticker,
                            current_price,
                            optionchain_df,
                            option_take_profit_percent,
                            option_trail_stop_percent,
                            current_time,
                        )
                        if order_params != None:
                            print("Oreder params:", order_params)
                            unique_id = f"{order_params[1]}_{order_params[2]}_{order_params[3]}_{order_params[0]}"  # ticker_IB_option_date_contractStrike_CorP
                            if unique_id not in unique_orders:
                                unique_orders.add(unique_id)
                                potential_orders.append(order_params)

                        # Execute unique orders concurrently
                        # print(potential_orders)
                    except Exception as e:
                        logger.exception(f"Error in handle_model_result. {e}")
        except ValueError as e:
            logger.warning(
                f"{model_name} is likely missing some required feature data."
            )
            continue
        except Exception as e:
            log_error("actions", ticker, model_name, e)
    tasks = [place_option_order_sync(*params) for params in potential_orders]

    await asyncio.gather(*tasks)


# TODO use this log error funciton globally?

# Define a function to send notifications (assuming you have this functionality in the send_notifications module)

##TODO add clause to track price after buy signal.. if it drops x% then rebuy/reaverage.


# Define the model list, this assumes that the model list is predefined
# def get_model_list():
#     return [
#         # Add the actual models here
#         pytorch_trained_minute_models.Buy_3hr_PTminClassSPYA1,
#         pytorch_trained_minute_models.SPY_2hr_50pct_Down_PTNNclass,
#         # pytorch_trained_minute_models.Buy_20min_1pctup_ptclass_B1,
#         # pytorch_trained_minute_models.Buy_20min_05pctup_ptclass_B1,
#         pytorch_trained_minute_models._3hr_40pt_down_FeatSet2_shuf_exc_test_onlyvalloss,
#     ]
#


def get_model_list_for_ticker(ticker):
    # Example mapping of tickers to models
    ticker_to_models = {
        "SPY": [
            pytorch_trained_minute_models.Buy_3hr_PTminClassSPYA1,
            pytorch_trained_minute_models.SPY_2hr_50pct_Down_PTNNclass,
            pytorch_trained_minute_models.SPY_2hr_50pct_Down_PTNNclass_240124,
            pytorch_trained_minute_models._3hr_40pt_down_FeatSet2_shuf_exc_test_onlyvalloss,
            pytorch_trained_minute_models.SPY_ptminclassA1Base_2hr50ptdown_2401290107,
            pytorch_trained_minute_models.SPY_ptminclassA1Base_2hr50ptdown_2401292135,
            pytorch_trained_minute_models.SPY_ptminclassA1Base_2hr50ptdown_2402010049,
        ],
        "MSFT": [
            pytorch_trained_minute_models.MSFT_2hr_50pct_Down_PTNNclass,
            pytorch_trained_minute_models.MSFT_ptminclassA1Base_2hr50ptdown_2401290107,
            pytorch_trained_minute_models.MSFT_ptminclassA1Base_2hr50ptdown_2401292135,
            pytorch_trained_minute_models.MSFT_ptminclassA1Base_2hr50ptdown_2402010051,
        ],
        "TSLA": [
            pytorch_trained_minute_models.TSLA_ptminclassA1Base_2hr50ptdown_2401290106,
            pytorch_trained_minute_models.TSLA_ptminclassA1Base_2hr50ptdown_2401292134,
            pytorch_trained_minute_models.TSLA_ptminclassA1Base_2hr50ptdown_2402010052,
        ],
        # Add other tickers and their models here
    }

    return ticker_to_models.get(
        ticker, []
    )  # Return an empty list if no models are found for the ticker


# TODO make it look for pairs first somehow?  store all orders, and take best?   PROCESSED DATA IS NOT USED
from datetime import datetime


# string 	LiquidHours
#  	The liquid hours of the product. This value will contain the liquid hours (regular trading hours) of the contract on the specified exchange. Format for TWS versions until 969: 20090507:0700-1830,1830-2330;20090508:CLOSED. In TWS versions 965+ there is an option in the Global Configuration API settings to return 1 month of trading hours. In TWS v970 and above, the format includes the date of the closing time to clarify potential ambiguity, e.g. 20180323:0930-20180323:1600;20180326:0930-20180326:1600.orry forgot to mention about the timezone. Yes - the only way to keep sane is to convert all incoming date time objects to pytz.UTC. BTW: you cannot rely on the timezone the IB API gives you. According to them the timezone for the CME is Belize! I hold all those static data in dictionaries in my system.
async def get_contract_details(
    optionchain_df,
    ticker,
    model_name,
    current_time,
    target_delta=0.9,
    gamma_threshold=(0.0, 0.9),
    max_bid_ask_spread_percent=4,
    min_volume=500,
):
    # Determine the type of contract based on the model name

    CorP = (
        "C" if "Buy" in model_name or "Up" in model_name or "up" in model_name else "P"
    )

    # Define liquidity thresholds
    min_volume = min_volume

    # Extract delta and gamma values, calculate bid-ask spread as a percentage of the contract price
    if CorP == "C":
        optionchain_df["c_delta"] = optionchain_df["c_greeks"].apply(
            lambda x: x.get("delta") if isinstance(x, dict) else None
        )
        optionchain_df["c_gamma"] = optionchain_df["c_greeks"].apply(
            lambda x: x.get("gamma") if isinstance(x, dict) else None
        )
        price_column = "Call_LastPrice"
        delta_column = "c_delta"
        gamma_column = "c_gamma"
        volume_column = "Call_Volume"
        bid_column = "c_bid"
        ask_column = "c_ask"
    else:
        optionchain_df["p_delta"] = optionchain_df["p_greeks"].apply(
            lambda x: x.get("delta") if isinstance(x, dict) else None
        )
        optionchain_df["p_gamma"] = optionchain_df["p_greeks"].apply(
            lambda x: x.get("gamma") if isinstance(x, dict) else None
        )
        price_column = "Put_LastPrice"
        delta_column = "p_delta"
        gamma_column = "p_gamma"
        volume_column = "Put_Volume"
        bid_column = "p_bid"
        ask_column = "p_ask"

    # Calculate bid-ask spread percentage and apply liquidity filter
    optionchain_df["bid_ask_spread_percent"] = (
        (optionchain_df[ask_column] - optionchain_df[bid_column])
        / optionchain_df[price_column]
    ) * 100
    liquidity_filter = (optionchain_df[volume_column] >= min_volume) & (
        optionchain_df["bid_ask_spread_percent"] <= max_bid_ask_spread_percent
    )

    # Apply gamma filter
    gamma_filter = (optionchain_df[gamma_column] >= gamma_threshold[0]) & (
        optionchain_df[gamma_column] <= gamma_threshold[1]
    )

    # Apply combined filters
    relevant_df = optionchain_df[liquidity_filter & gamma_filter].dropna(
        subset=[delta_column, gamma_column]
    )
    # Filter out contracts with the same expiration date as the current date
    current_date = current_time.strftime("%y%m%d")
    relevant_df = relevant_df[relevant_df["ExpDate"] != current_date]

    # Ensure the DataFrame is not empty
    if relevant_df.empty:
        logger.info(f"{ticker} contractdetails relevant_df empty", exc_info=True)
        return None

    # Find the contract with delta closest to the target delta
    adjusted_target_delta = target_delta if CorP == "C" else -target_delta
    relevant_df["delta_diff"] = (
        relevant_df[delta_column] - adjusted_target_delta
    ).abs()
    closest_delta_row = relevant_df.iloc[relevant_df["delta_diff"].argsort()[:1]]

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
    contract_symbol = (
        f"{ticker}{date_object.strftime('%y%m%d')}{CorP}{formatted_contract_strike:08d}"
    )

    # Determine the direction for the notification message
    upordown = "up" if CorP == "C" else "down"
    formatted_time = current_time.strftime("%y%m%d %H:%M EST")
    formatted_time_mdHMonly = current_time.strftime("%m%d_%H:%M")
    # formatted_time_mdHMonly = current_time
    # Get the current time formatted for the notification message
    # print (
    #     upordown,
    #     CorP,
    #     contractStrike,
    #     contract_price,
    #     IB_option_date,
    #     formatted_time,
    #     formatted_time_mdHMonly,
    #
    # )
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
