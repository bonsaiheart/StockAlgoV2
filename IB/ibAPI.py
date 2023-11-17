import asyncio
import datetime
import logging
import os
import json
from random import random

from ib_insync import *
from mpmath import rand

from UTILITIES.logger_config import logger

# Initialization and global variables
project_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(project_dir, "errorlog")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.getLogger('ib_insync').setLevel(logging.WARNING)
logging.getLogger('ib_insync').addHandler(logger.handlers[0])
# TODO note that when i dont use the order handler += at the end, thats when orders seem to get mismatched/not transmitted etc.
ib = IB()


# util.getLoop()
# parentOrderIdFile = "IB/parent_order_ids.txt"
# parentOrders = {}

# open_orders = ib.openOrders()
# # print(ib.openOrders())
# # process open orders
# def log_order_details(order):
#     with open("IB/order_log.txt", "a") as f:
#         f.write(f"{order}\n")
#
# log_order_details(open_orders)
# TODO order handling for "cannot both sides of ordr" error

def connection_stats():
    print(ib.isConnected())


# ib.disconnect()
async def ib_connect():
    if not ib.isConnected():
        print("~~~ Connecting ~~~")
        # randomclientID = random.randint(0, 999)#TODO change bac kclientid
        try:
            await ib.connectAsync("192.168.1.119", 7497, clientId=2, timeout=45)
            # open_orders = ib.openOrders()
            # print('openorders',ib.reqOpenOrders())
            # print("ALL OPEN ORDERS:",ib.reqAllOpenOrders())

            # print("posistions:" ,ib.positions())
        except (Exception, asyncio.exceptions.TimeoutError) as e:
            logging.getLogger().error("Connection error: %s", e)
            print("~~Connection error:", e)
    else:
        print("~~~IB already connected.~~~")


def ib_disconnect():
    try:
        ib.disconnect()
    except (Exception, asyncio.exceptions.TimeoutError) as e:
        logging.error("Connection error: %s", e)


async def get_execution_price(parent_order_id):
    for trade in ib.trades():
        if trade.order.orderId == parent_order_id and trade.orderStatus.status == 'Filled':
            return trade.orderStatus.avgFillPrice
    return None


async def find_child_orders_with_details(contract, action):
    child_orders = []
    # Filter open orders to find child orders for the given contract
    for order in ib.openOrders():

        trade = await getTrade(order)
        print("action",action)
        # print(order.action)
        # print(contract)
        # print(trade.contract)
        # print(order.parentId , order.action,trade.contract)
        if order.action == 'SELL' and trade.contract == contract:
            # print("childorders trade",trade)
            child_orders.append(order)

    print('childorders trade obj:', child_orders)
    return child_orders


async def getTrade(order):
    trade = next((trade for trade in ib.trades() if trade.order is order), None)

    return trade


# With trade.isDone() can the be checked if the order is stil running or not.


def handle_exception(loop, context):
    msg = context.get("exception", context["message"])
    print('handleexception: ', msg)
    logging.error(f"Caught exception: {msg}")
    logging.info("Shutting down...")
    asyncio.create_task(loop.shutdown())


# async def cancel_order(order):
#     trade = await getTrade(order)
#     if trade.orderStatus.status == 'Cancelled':
#         print("Order already cancelled")
#         return
#
#     # Set up an asyncio Event to wait for the order to be cancelled
#     order_cancelled = asyncio.Event()
#
#     # Define a callback function for order status updates
#     def on_order_status(trade_update, order_status):
#         if trade_update.order.orderId == order.orderId and order_status.status == 'Cancelled':
#             order_cancelled.set()
#
#     # Subscribe to the order status event
#     ib.orderStatusEvent += on_order_status
#
#     # Send cancel request
#     ib.cancelOrder(order)
#
#     # Wait for the order status to change to 'Cancelled'
#     await order_cancelled.wait()
#
#     # Unsubscribe from the event to avoid memory leaks
#     ib.orderStatusEvent -= on_order_status
#
#     print("Order cancelled")
# async def cancel_order(order):
#     print('attempting to cancel',order)
#     trade = await getTrade(order)
#     if trade.orderStatus.status != 'Cancelled':
#         print("TRADE ORDERSTATUSEVENT",trade.orderStatus)
#         # ib.cancelOrder(order)
#         ib.cancelOrder(order)
#         # ib.waitOnUpdate()
#
#     while trade.orderStatus.status != 'Cancelled' :
#         print(trade.orderStatus.status)

# Define a callback function for the cancelOrderEvent
async def cancel_order(order):
    print('Attempting to cancel', order)
    trade = await getTrade(order)

    # Create an asyncio Event to wait for the order to be cancelled
    order_cancelled = asyncio.Event()

    # Define a callback function for the cancelOrderEvent
    def make_on_cancel_order_event(order_id):
        def on_cancel_order_event(trade: Trade):
            if trade.order.orderId == order_id:
                print(f"Order cancellation confirmed for trade: {trade}")
                order_cancelled.set()
        return on_cancel_order_event

    # Subscribe to the cancelOrderEvent
    on_cancel_order_event = make_on_cancel_order_event(order.orderId)
    ib.cancelOrderEvent += on_cancel_order_event
    ib.cancelOrder(order)

    # while not trade.isDone():
    #     print(trade.orderStatus)
    #     await order_cancelled.wait()  # Wait for the cancellation to be confirmed
    #     # print("Order cancelled")

    # Unsubscribe from the event to avoid memory leaks
    ib.cancelOrderEvent -= on_cancel_order_event


async def get_execution_price(parent_order_id):
    for trade in ib.trades():
        if trade.order.orderId == parent_order_id and trade.orderStatus.status == 'Filled':
            return trade.orderStatus.avgFillPrice
    return None


async def cancel_and_replace_orders(contract,action, CorP, ticker, exp, strike, contract_current_price,
                                    quantity, orderRef, take_profit_percent, trailstop_amount_percent):
    child_orders_objs = await find_child_orders_with_details(contract, action)

    # Store details of trailing stop and take profit orders
    order_details = {}
    for order in child_orders_objs:
        ocaGroup = order.ocaGroup
        # print(ocaGroup)
        trade = await getTrade(order)
        # print("TRAFDSISAFIASDFASDIFAD S",trade)
        # print(order.orderType,order)
        if order.parentId not in order_details:
            order_details[order.parentId] = {}
        if order.orderType == "TRAIL":
            order_details[order.parentId]["stopLoss"] = {
                "type": "TRAIL",
                "trailAmount": order.auxPrice,
                "triggerPrice": order.trailStopPrice,
                "ocaGroup": order.ocaGroup,
                "orderRef": order.orderRef,
                "parentID":order.parentId
            }
        elif order.orderType == "LMT":
            print(order_details)
            order_details[order.parentId]["takeProfit"] = {
                "type": "LMT",
                "limitPrice": order.lmtPrice,
                "ocaGroup": order.ocaGroup,
                "orderRef": order.orderRef,
                "parentID":order.parentId
            }
        # elif order.orderType =
        # print('cancelling orders', order)
        await cancel_order(order)
    # print("ORDER DEATAILS DICT", order_details)
    print('Placing order')
    parenttrade = await placeOptionBracketOrder(
        CorP,
        ticker,
        exp,
        strike,
        contract_current_price,
        quantity=quantity,
        orderRef=orderRef,
        take_profit_percent=take_profit_percent,
        trailstop_amount_percent=trailstop_amount_percent,
        check_opposite_orders=False)
    while not parenttrade.isDone():
        # print(parenttrade.isDone())
        ib.waitUntil()
    # trade.orderStatus.ActiveStates
    print('replacing child orders.',order_details)

    await replace_child_orders(order_details, contract,
                               quantity)


async def replace_child_orders(order_details, contract,
                               quantity):
    try:
        for parent_id, child_order_details in order_details.items():
            # print(order_details.items())
            ticker_contract = contract
#TODO chnage qtty back
            quantity = 1  # Replace with the desired order quantity

            # This will be our main or "parent" order
            parent = Order()
            # parent.orderId = parent_id
            parent.orderId = ib.client.getReqId()

            takeProfit = Order()
            takeProfit.action = "SELL"
            takeProfit.orderType = "LMT"
            takeProfit.totalQuantity = quantity
            # takeProfit.parentId = parent.orderId
            takeProfit.outsideRth = True
            takeProfit.transmit = True
            takeProfit.tif = "GTC"
            # Use stored take profit details if available
            if order_details and order_details[parent_id]["takeProfit"]["type"] == "LMT":
                takeProfit.lmtPrice = order_details[parent_id]["takeProfit"]["limitPrice"]
                takeProfit.ocaGroup = order_details[parent_id]["takeProfit"]["ocaGroup"]
                # takeProfit.orderRef = order_details[parent_id]["takeProfit"]["orderRef"]
                takeProfit.orderRef = 'replaced takeprof'

            takeProfit.orderId = ib.client.getReqId()

            stopLoss = Order()
            stopLoss.action = "SELL"
            stopLoss.orderType = "TRAIL"
            stopLoss.TrailingUnit = 1
            stopLoss.totalQuantity = quantity
            # stopLoss.parentId = parent.orderId
            stopLoss.outsideRth = True
            stopLoss.transmit = True
            stopLoss.tif = "GTC"
            # Use stored trailing stop details if available
            if order_details and order_details[parent_id]["stopLoss"]["type"] == "TRAIL":
                stopLoss.auxPrice = order_details[parent_id]["stopLoss"]["trailAmount"]
                stopLoss.trailStopPrice = order_details[parent_id]["stopLoss"]["triggerPrice"]
                stopLoss.ocaGroup = order_details[parent_id]["stopLoss"]["ocaGroup"]
                stopLoss.orderRef = order_details[parent_id]["stopLoss"]["orderRef"]
                stopLoss.orderRef = 'replaced stoploss'

            stopLoss.orderId = ib.client.getReqId()

            bracketOrder = [takeProfit, stopLoss]

            ##TODO change ref back
            print(f"~~~~Replacing cancelled child Order: {parent.orderRef} ~~~~~")
            for o in bracketOrder:
                o.ocaType = 2
                o.clientId = 2
                ib.placeOrder(ticker_contract, o)

            print(f"~~~~Children  Replaced: {parent.orderRef} ~~~~~")
            trade = await getTrade(takeProfit)
            while not trade.isDone():
                # print(trade.orderStatus)
                ib.sleep(0)

    except (Exception, asyncio.exceptions.TimeoutError) as e:
        logger.exception(f"An error occurred while replace children.{ticker_contract},: {e}")


# Event handler for order status updates
async def orderStatusHandler(trade: Trade):
    global parentOrders

    order_id = trade.order.orderId  # Accessing the orderId from the order attribute of the Trade object
    # print("status handler order id",order_id)
    # Find the parent order for the updated child order
    # for parentOrderId, orderDetails in parentOrders.items():
    #     if 'childOrderIds' in orderDetails and order_id in orderDetails['childOrderIds']:
    #         # Update the status of the child order
    #         childOrderStatus = {
    #             "filled": trade.filled(),
    #             "remaining": trade.remaining(),
    #             "status": trade.orderStatus.status
    #         }
    # print("order details :",order_details)
    # orders_to_cancel = [order for order in child_orders_objs]
    # await cancel_orders(child_orders_objs)
    # for order in child_orders_objs:
    #     trade = await getTrade(order)
    #     print("TRADE cancelled?:  ",trade)
    #     # ib.sleep(20)
    #     # print("TRADE cancelled now?:  ", trade)

    # Replace orders with updated details
    # await placeOptionBracketOrder(CorP, ticker, exp, strike, contract_current_price,
    #                               quantity, orderRef, take_profit_percent, trailstop_amount_percent,
    #                               check_opposite_orders=False, order_details=order_details)


# async def reinstate_cancelled_orders(orders_to_reinstate):
#     for order_id, orderDetails in orders_to_reinstate.items():
#         print("reinstating orders: ",orderDetails)
#         await placeOptionBracketOrderFromDetails(orderDetails)
#         # elif orderDetails["contract"].__class__ == Stock:
#         #     await placeStockBracketOrderFromDetails(orderDetails)
#


async def placeOptionBracketOrder(
        CorP,
        ticker,
        exp,
        strike,
        contract_current_price,
        quantity=1,
        orderRef=None,
        take_profit_percent=.15,
        trailstop_amount_percent=.2,
        check_opposite_orders=True):
    if take_profit_percent == None:
        take_profit_percent = .15
    if trailstop_amount_percent == None:
        trailstop_amount_percent = .2
    gtddelta = (datetime.datetime.now() + datetime.timedelta(seconds=30)).strftime("%Y%m%d %H:%M:%S")
    print("~~~Placing order~~~:")
    ticker_contract = Option(ticker, exp, strike, CorP, "SMART")

    if check_opposite_orders:
        try:
            print("Checking opposite orders.")
            action = "SELL"  # Adjust based on your logic
            await cancel_and_replace_orders(ticker_contract,action, CorP, ticker, exp, strike, contract_current_price,
                                            quantity, orderRef, take_profit_percent, trailstop_amount_percent)
            # ib.sleep(5)
        except (Exception, asyncio.exceptions.TimeoutError) as e:
            logger.exception(f"An error occurred while optionbracketorder.{ticker},: {e}")
    else:
        try:

            contract_current_price = round(contract_current_price, 2)
            #TODO changed quantity
            quantity = 1  # Replace with the desired order quantity
            limit_price = contract_current_price  # Replace with your desired limit price
            # print(take_profit_percent, trailstop_amount_percent, "CUSTOMSSSS")
            take_profit_price = round(
                contract_current_price + (contract_current_price * take_profit_percent), 2
            )  # Replace with your desired take profit price

            stop_loss_price = contract_current_price * 0.9  # Replace with your desired stop-loss price
            trailAmount = round(
                contract_current_price * trailstop_amount_percent, 2
            )  # Replace with your desired trailing stop percentage
            triggerPrice = limit_price

            # This will be our main or "parent" order
            parent = Order()
            parent.orderId = ib.client.getReqId()
            parent.action = "BUY"
            parent.orderType = "LMT"
            parent.totalQuantity = quantity
            parent.lmtPrice = limit_price
            parent.transmit = False
            parent.outsideRth = True
            ###this stuff makes it cancel whole order in 45 sec.  If parent fills, children turn to GTC
            parent.tif = "GTD"
            parent.goodTillDate = gtddelta

            takeProfit = Order()
            takeProfit.orderId = ib.client.getReqId()
            takeProfit.action = "SELL"
            takeProfit.orderType = "LMT"
            takeProfit.totalQuantity = quantity
            takeProfit.parentId = parent.orderId
            takeProfit.outsideRth = True
            takeProfit.transmit = False
            takeProfit.tif = "GTC"
            takeProfit.lmtPrice = round(contract_current_price + (contract_current_price * take_profit_percent), 2)

            stopLoss = Order()
            stopLoss.orderId = ib.client.getReqId()
            stopLoss.action = "SELL"
            stopLoss.orderType = "TRAIL"
            stopLoss.TrailingUnit = 1
            stopLoss.totalQuantity = quantity
            stopLoss.parentId = parent.orderId
            stopLoss.outsideRth = True
            stopLoss.transmit = True
            stopLoss.tif = "GTC"
            # Use stored trailing stop details if available

            trailAmount = round(contract_current_price * trailstop_amount_percent, 2)
            stopLoss.auxPrice = trailAmount
            stopLoss.trailStopPrice = contract_current_price - trailAmount

            bracketOrder = [parent, takeProfit, stopLoss]

            ##TODO change ref back
            # print(f"~~~~Placing Order: {parent.orderRef} ~~~~~")
            for o in bracketOrder:
                o.ocaType = 2
                o.clientId = 2
                if orderRef is not None:
                    o.orderRef = orderRef
                ib.placeOrder(ticker_contract, o)
            ##changed this 7.25
            # await ib.sleep(0)
            #     ib.sleep()

            print(f"~~~~Order Placed: {parent.orderRef} ~~~~~")#maybe just revert to when the canelling was working. maybe it was just overwhelmed?

            trade = await getTrade(parent)
            while not trade.isDone():
                # print(trade.orderStatus)
                await asyncio.sleep(0)
            return trade
        except (Exception, asyncio.exceptions.TimeoutError) as e:
            logger.exception(f"An error occurred while optionbracketorder.{ticker},: {e}")

            # ib.disconnect()


# async def placeBuyBracketOrder(ticker, current_price,
#                                quantity=1,
#                                orderRef=None,
#                                take_profit_percent=.0003,
#                                trail_stop_percent=.0002):
#     print(f"~~~~~Placing {ticker} BuyBracket order~~~~~")
#     try:
#         if take_profit_percent == None:
#             take_profit_percent = .003
#         if trail_stop_percent == None:
#             trail_stop_percent = .002
#         gtddelta = (datetime.datetime.now() + datetime.timedelta(seconds=180)).strftime("%Y%m%d %H:%M:%S")
#         # print("orders : ",ib.openOrders())
#         ticker_symbol = ticker
#         ticker_contract = Stock(ticker_symbol, "SMART", "USD")
#         # ib.qualifyContracts(ticker_contract)
#
#         current_price = current_price
#         quantity = quantity
#         limit_price = current_price
#         take_profit_price = round(current_price + (current_price * take_profit_percent), 2)
#         stop_loss_price = current_price * 0.9
#         trailAmount = round(current_price * trail_stop_percent, 2)
#         triggerPrice = limit_price
#
#         parent = Order()
#         parent.orderId = ib.client.getReqId()
#         parent.action = "BUY"
#         parent.orderType = "LMT"
#         parent.totalQuantity = quantity
#         parent.lmtPrice = limit_price
#         parent.transmit = False
#         parent.outsideRth = True
#         ###this stuff makes it cancel whole order in 45 sec.  If parent fills, children turn to GTC
#         parent.goodTillDate = gtddelta
#         parent.tif = "GTD"
#
#         takeProfit = Order()
#         takeProfit.orderId = ib.client.getReqId()
#         takeProfit.action = "SELL" if parent.action == "BUY" else "BUY"
#         takeProfit.outsideRth = True
#         takeProfit.orderType = "LMT"
#         takeProfit.totalQuantity = quantity
#         takeProfit.lmtPrice = take_profit_price
#         takeProfit.parentId = parent.orderId
#         takeProfit.transmit = False
#         takeProfit.tif = "GTC"
#
#         stopLoss = Order()
#         stopLoss.orderId = ib.client.getReqId()
#         stopLoss.action = "SELL" if parent.action == "BUY" else "BUY"
#         stopLoss.orderType = "TRAIL"
#         stopLoss.TrailingUnit = 1
#         stopLoss.outsideRth = True
#         stopLoss.auxPrice = trailAmount
#         stopLoss.trailStopPrice = limit_price - trailAmount
#         stopLoss.totalQuantity = quantity
#         stopLoss.parentId = parent.orderId
#         stopLoss.transmit = True
#         stopLoss.tif = "GTC"
#
#         bracketOrder = [parent, takeProfit, stopLoss]
#         orderDetailsDict = {
#             "ticker": ticker,
#             "contract_current_price": current_price,
#             "quantity": quantity,
#             "orderRef": orderRef,
#             "take_profit_percent": take_profit_percent,
#             "trailstop_amount_percent": trail_stop_percent,
#             "childOrderIds": [takeProfit.orderId, stopLoss.orderId]
#         }
#
#         ##TODO change ref back
#         for o in bracketOrder:
#             if orderRef is not None:
#                 o.orderRef = orderRef
#             ib.placeOrder(ticker_contract, o)
#             ##changed this 7.25
#             # ib.sleep(0)
#         print(f"~~~~Placing order: {parent.orderRef} ~~~~~")
#     except (Exception) as e:
#         logger.exception(f"An error occurred while placeBuyBracketOrder.{ticker},: {e}")

# Load previously stored parent order IDs and child order IDs
# parentOrders = retrieveOrderIdFromFile(parentOrderIdFile)

# Register the event handler for order status
# ib.orderStatusEvent += orderStatusHandler

###TODO diff client id for diff stat. and add options.
"""

My old code placing a bracket order looked like this:

order = ib.bracketOrder('BUY', amount, limit, takeprofit, stoploss, outsideRth=True, tif='GTC')
for ord in order:
    ib.placeOrder(contract, ord
    )
This will place a bracket order with TIF (time in force) set to GTC (good till cancelled) for all 3 orders, the parent and the 2 children.

When I changed it to TIF=GTD and specified a time, this, of course, applied to the whole bracket order. So if it fills in the desired time, the takeprofit and stoploss will disappear after the GTD time expires. Not good.

Then, someone sent me a little help and now this code works for me:

bracket = ib.bracketOrder('BUY', amount, limit, takeprofit,stoploss, outsideRth=True)

gtddelta = (datetime.now() + timedelta(seconds=45)).strftime("%Y%m%d %H:%M:%S")
bracket.parent.tif = 'GTD'
bracket.parent.goodTillDate = gtddelta

for order in bracket:
    ib.placeOrder(contract, order)
This sets a 45 second GTD for the parent order. If it does not get filled, the whole bracket order will be cancelled. If it does get filled, the takeprofit and the stoploss order remain, with TIF=GTC.

Please note the changes to the old code. It is important to call the fist variable 'bracket' to be able to define the 'bracket.parent' later (look into the definiton).


If you want to cancel the outstanding order you can use  ib.reqGlobalCancel() (note: this will cancel all the open orders)

If you have an open position with this order you should use the ib.positions()and retrieve the relevant position to be closed


******Just add a trigger price and adjusted parameters to your stop order, like this code for example. Your stop will become a trail stop once the trigger price is touched

stop = Order(action = reverseAction, totalQuantity = 1, orderType = "STP",
                         parentId = parent.orderId, auxPrice = stop_price,
                         tif ="GTC", transmit = True, orderId = self.ib.client.getReqId(),
                         triggerPrice = trigger_price,
                         adjustedOrderType = "TRAIL",  
                         adjustedTrailingAmount = trail_amount,
                         adjustedStopPrice = adjusted_stop_price )  
"""
# def getTrade(order):
#     trade = next((trade for trade in ib.trades() if trade.order is order), None)
# With trade.isDone() can the be checked if the order is stil running or not.
#
# Another way to check is with order in ib.openOrders().
