import asyncio
import re
from datetime import datetime
import pandas as pd
import IB.ibAPI
from Strategy_Testing.Trained_Models import trained_minute_models, pytorch_trained_minute_models
from UTILITIES.Send_Notifications import send_notifications
from UTILITIES.logger_config import logger


# Utility function to handle errors
def log_error(location, ticker, model_name, exception):
    logger.error(f"An error occurred in {location}. {ticker}, {model_name}: {exception}", exc_info=True)


# Order placement functions
async def place_option_order_sync(CorP, ticker, exp, strike, contract_current_price, orderRef,
                                  quantity, take_profit_percent, trail_stop_percent):
    try:
        await IB.ibAPI.placeOptionBracketOrder(
            CorP, ticker, exp, strike, contract_current_price, quantity,
            orderRef, take_profit_percent, trail_stop_percent
        )
    except Exception as e:
        log_error("place_option_order_sync", ticker, orderRef, e)


async def place_buy_order_sync(ticker, current_price, orderRef, quantity, take_profit_percent,
                               trail_stop_percent):
    try:
        await IB.ibAPI.placeBuyBracketOrder(
            ticker, current_price, quantity, orderRef, take_profit_percent, trail_stop_percent
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
        seconds = int(num_part) * (60 if unit_part == 'min' else 3600)
        return f"{num_part}{unit_part}", seconds
    else:
        logger.error(f"Invalid model function name: {model_name}")
        return None, None


# Main function to handle model actions
async def actions(optionchain_df, dailyminutes_df, processeddata_df, ticker, current_price):
    # Load your data into dataframes

    # Iterate over each model in your model list
    for model in get_model_list():
        model_name = model.__name__
        model_output = model(dailyminutes_df)
        # print(model_output)
        try:
            # TODO make each model return signal, so they can have individual thressholds for buy/sell.
            if isinstance(model_output, tuple):
                model_output_df, stock_take_profit_percent, stock_trail_stop_percent,option_take_profit_percent,option_trail_stop_percent = model_output
            else:
                model_output_df = model_output  # Assuming `model_output` is a DataFrame or similar
                stock_take_profit_percent = None
                stock_trail_stop_percent = None
                option_take_profit_percent = None
                option_trail_stop_percent = None
            result = model_output_df.iloc[-1]
            print(model_name,result)
            # If the model result is positive (greater than 0.5 in your case), handle the positive result
            if result > 0.5\
                    :
                # Retrieve the contract details
                upordown, CorP, contractStrike, contract_price, IB_option_date, formatted_time = get_contract_details(
                    optionchain_df, processeddata_df, ticker, model_name
                )

                callorput = 'call' if CorP == 'C' else 'put'
                # print(f'Positive result for {ticker} {model_name}')
                timetill_expectedprofit, seconds_till_expectedprofit = check_interval_match(model_name)
                send_notifications.send_tweet_w_countdown_followup(
                    ticker,
                    current_price,
                    upordown,
                    f"${ticker} ${current_price}. {timetill_expectedprofit} to make money on a {callorput} #{model_name} {formatted_time}",
                    seconds_till_expectedprofit, model_name
                )

        #TODO uncomment optionorder.
                # await place_option_order_sync(
                #     CorP, ticker, IB_option_date, contractStrike, contract_price, model_name,
                #     quantity=19, take_profit_percent=take_profit_percent, trail_stop_percent=trail_stop_percent
                # )

                # Place the buy order if applicable (this part depends on your specific trading strategy)
                await place_buy_order_sync(
                    ticker, current_price, model_name, quantity=4,
                    take_profit_percent=stock_take_profit_percent, trail_stop_percent=stock_trail_stop_percent
                )

        except Exception as e:
            log_error("actions", ticker, model_name, e)


# Define a function to send notifications (assuming you have this functionality in the send_notifications module)
async def send_notification(ticker, current_price, upordown, model_name, formatted_time):
    message = f"${ticker} ${current_price}. Time to make money on a {upordown} #{model_name} {formatted_time}"

    # Assuming you have an async function to send notifications
    await send_notifications.async_send_tweet(message)

    # For a synchronous function you can use run_in_executor to run it in an async manner
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, send_notifications.sync_send_email, message)
##TODO add clause to track price after buy signal.. if it drops x% then rebuy/reaverage.

# Define the model list, this assumes that the model list is predefined
def get_model_list():
    return [
        # Add the actual models here
        # trained_minute_models.Buy_3hr_15minA2baseSPYA1,
        # trained_minute_models.Sell_3hr_15minA2baseSPYA1,
        pytorch_trained_minute_models.Buy_20min_1pctup_ptclass_B1,
        pytorch_trained_minute_models.Buy_20min_05pctup_ptclass_B1,
        # pytorch_trained_minute_models.Sell_20min_05pctdown_ptclass_S1,
        # pytorch_trained_minute_models.Buy_1hr_ptmin/classSPYA1,
        pytorch_trained_minute_models.Buy_3hr_PTminClassSPYA1,
        # pytorch_trained_minute_models.Buy_2hr_ptminclassSPYA2,
        # pytorch_trained_minute_models.Buy_2hr_ptminclassSPYA1,
        ]  


# # Function to handle a positive model result
# async def handle_positive_result(model_name, ticker, current_price, optionchain_df, processeddata_df, custom_takeprofit,
#                                  custom_trailingstop):
#     upordown, CorP, contractStrike, contract_price, IB_option_date, formatted_time = get_contract_details(
#         optionchain_df, processeddata_df, ticker, model_name
#     )
#
#     # Send a notification about the positive result
#     await send_notification(ticker, current_price, upordown, model_name, formatted_time)
#
#     # Place the option order
#     await place_option_order_sync(
#         CorP, ticker, IB_option_date, contractStrike, contract_price, model_name,
#         quantity=10, take_profit_percent=custom_takeprofit, trail_stop_percent=custom_trailingstop
#     )
#
#     # If you also want to place a buy order
#     await place_buy_order_sync(
#         ticker, current_price, model_name, quantity=10,
#         take_profit_percent=custom_takeprofit, trail_stop_percent=custom_trailingstop
#     )


# Function to retrieve contract details
def get_contract_details(optionchain_df, processeddata_df, ticker, model_name):
    # Extract the closest expiration date and strikes list
    closest_exp_date = processeddata_df['ExpDate'].iloc[0]
    closest_strikes_list = processeddata_df["Closest Strike Above/Below(below to above,4 each) list"].iloc[0]

    # Format the expiration date for IB
    date_object = datetime.strptime(str(closest_exp_date), "%y%m%d")
    print(date_object)
    formatted_contract_date = date_object.strftime("%y%m%d")

    IB_option_date = date_object.strftime("%Y%m%d")
    print(IB_option_date)
    # Determine the type of contract based on the model name
    CorP = "C" if "Buy" in model_name else "P"

    # Calculate the contract strike and price
    contractStrike = closest_strikes_list[1] if CorP == "C" else closest_strikes_list[-2]
    # has for mat 410.5
    formatted_contract_strike = contractStrike * 1000
    print(contractStrike)
    contract_symbol = f"{ticker}{formatted_contract_date}{CorP}{int(formatted_contract_strike):08d}"
    print(contract_symbol)
    print("wowowow", optionchain_df.loc[optionchain_df["c_contractSymbol"] == contract_symbol]["Call_LastPrice"])
    # Get the last price for the contract
    if CorP == "C":
        contract_price = \
            optionchain_df.loc[optionchain_df["c_contractSymbol"] == contract_symbol]["Call_LastPrice"].values[0]
    else:
        contract_price = \
            optionchain_df.loc[optionchain_df["p_contractSymbol"] == contract_symbol]["Put_LastPrice"].values[0]

    # Determine the direction for the notification message
    upordown = "up" if CorP == "C" else "down"

    # Get the current time formatted for the notification message
    formatted_time = datetime.now().strftime("%y%m%d %H:%M EST")

    return upordown, CorP, contractStrike, contract_price, IB_option_date, formatted_time


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
