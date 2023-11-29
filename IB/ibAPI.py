import asyncio
import datetime
import logging
import os
# import random
from ib_insync import *
from UTILITIES.logger_config import logger
#TODO set up logger
project_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where the script is
log_dir = os.path.join(project_dir, "errorlog")  # Builds the path to the errorlog directory

# Create the directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.getLogger('ib_insync').setLevel(logging.WARNING)
# Explicitly add the handler for ib_insync
logging.getLogger('ib_insync').addHandler(logger.handlers[0])

log_file = os.path.join(log_dir, "error_ib.log")  # Builds the path to the log file

# Set up logging
# logging.basicConfig(
#     filename=log_file,
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M",
# )
# Initialize IB object
ib = IB()

# File paths for storing order IDs
parentOrderIdFile = "IB/parent_order_ids.txt"

# Global variables
parentOrders = {}

#TODO order handling for "cannot both sides of ordr" error

# ...
# @app.task
# def close_orders():
#     global parentOrdersf
#     for parentOrderId, childOrders in parentOrders.items():
#         for childOrderId in childOrders:
#             ib.cancelOrder(childOrderId)
#         parentOrder = ib.reqParentOrders(parentOrderId)
#         if parentOrder and parentOrder[0].filled:
#             # Get the contract and action from the parent order
#             contract = parentOrder[0].contract
#             action = "SELL" if parentOrder[0].action == "BUY" else "BUY"
#
#             # Create a closing order
#             closeOrder = MarketOrder(action, parentOrder[0].filledQuantity)
#
#             # Place the closing order
#             ib.placeOrder(contract, closeOrder)
#
#             # Remove the parent order from the dictionary
#             del parentOrders[parentOrderId]


def connection_stats():
    print(ib.isConnected())

def handle_exception(loop, context):
    msg = context.get("exception", context["message"])
    logging.error(f"Caught exception: {msg}")
    logging.info("Shutting down...")
    asyncio.create_task(loop.shutdown())
loop = asyncio.get_event_loop()
loop.set_exception_handler(handle_exception)


# ib.disconnect()
async def ib_connect():
    if not ib.isConnected():
        print("~~~ Connecting ~~~")
        # randomclientID = random.randint(0, 999)#TODO change bac kclientid
        try:#TODO change back to client 1
            await ib.connectAsync("192.168.1.119", 7497, clientId=2,timeout=45)
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


# asyncio.get_event_loop().close()
def saveOrderIdToFile(file_path, order_ids):
    with open(file_path, "w") as f:
        for parentOrderId, childOrders in order_ids.items():
            f.write(f'{parentOrderId}: {",".join(str(childOrderId) for childOrderId in childOrders)}\n')


def retrieveOrderIdFromFile(file_path):
    order_ids = {}
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                parentOrderId, childOrderIds = line.strip().split(": ")
                order_ids[int(parentOrderId)] = {
                    int(childOrderId) for childOrderId in childOrderIds.split(",") if childOrderId
                }
    return order_ids


def orderStatusHandler(orderStatus: OrderStatus):
    global parentOrders

    # print("printorderstatus.filled:", orderStatus.filled)
    # if orderStatus.filled == "filled":
    #     parentOrderId = orderStatus.orderStatus.parentId
    #     childOrderId = orderStatus.orderSetatus.orderId
    #     if parentOrderId in parentOrders and childOrderId in parentOrders[parentOrderId]:
    #         parentOrders[parentOrderId].pop(childOrderId, None)


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
):
    if take_profit_percent == None:
        take_profit_percent=.15
    if trailstop_amount_percent==None:
        trailstop_amount_percent=.2
    gtddelta = (datetime.datetime.now() + datetime.timedelta(seconds=180)).strftime("%Y%m%d %H:%M:%S")

    print("~~~Placing order~~~:")
    print("~~~gtddelta:",gtddelta)
    try:

        ticker_contract = Option(ticker, exp, strike, CorP, "SMART")
        await ib.qualifyContractsAsync(ticker_contract)
        print(ticker_contract)
        contract_current_price = round(contract_current_price, 2)
        print(contract_current_price)
        quantity = quantity  # Replace with the desired order quantity
        limit_price = contract_current_price  # Replace with your desired limit price
        print(take_profit_percent,trailstop_amount_percent,"CUSTOMSSSS")
        take_profit_price = round(
            contract_current_price+(contract_current_price * take_profit_percent), 2
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
        takeProfit.lmtPrice = take_profit_price
        takeProfit.parentId = parent.orderId
        takeProfit.outsideRth = True
        takeProfit.transmit = False
        takeProfit.tif = "GTC"

        stopLoss = Order()
        stopLoss.orderId = ib.client.getReqId()
        stopLoss.action = "SELL"
        stopLoss.orderType = "TRAIL"
        stopLoss.TrailingUnit = 1
        stopLoss.auxPrice = trailAmount
        stopLoss.trailStopPrice = limit_price - trailAmount
        stopLoss.totalQuantity = quantity
        stopLoss.parentId = parent.orderId
        stopLoss.outsideRth = True
        stopLoss.transmit = True
        stopLoss.tif = "GTC"

        bracketOrder = [parent, takeProfit, stopLoss]
        parentOrderId = parent.orderId
        childOrderIds = [takeProfit.orderId, stopLoss.orderId]
        parentOrders[parentOrderId] = childOrderIds  # Assign child order IDs to parent order ID key

        ##TODO change ref back
        print(f"~~~~Placing Order: {parent.orderRef} ~~~~~")
        for o in bracketOrder:
            if orderRef is not None:
                o.orderRef = orderRef
            # ib.sleep(1)
            ib.placeOrder(ticker_contract, o)
##changed this 7.25
            # await ib.sleep(0)
        # ib.sleep(0)
        # saveOrderIdToFile(parentOrderIdFile, parentOrders)
        print(f"~~~~Order Placed: {parent.orderRef} ~~~~~")

    except (Exception, asyncio.exceptions.TimeoutError) as e:
        logger.exception(f"An error occurred while optionbracketorder.{ticker},: {e}")

        # ib.disconnect()


async def placeBuyBracketOrder(ticker, current_price,
    quantity=1,
    orderRef=None,
    take_profit_percent=.0003,
    trail_stop_percent=.0002):
    trail_stop_percent = trail_stop_percent / 100
    take_profit_percent =take_profit_percent/100
    print(f"~~~~~Placing {ticker} BuyBracket order~~~~~")
    try:
        if take_profit_percent==None:
            take_profit_percent=.003
        if trail_stop_percent==None:
            trail_stop_percent=.002
        gtddelta = (datetime.datetime.now() + datetime.timedelta(seconds=180)).strftime("%Y%m%d %H:%M:%S")

        ticker_symbol = ticker
        ticker_contract = Stock(ticker_symbol, "SMART", "USD")
        # ib.qualifyContracts(ticker_contract)

        current_price = current_price
        quantity = quantity
        limit_price = current_price
        take_profit_price = round(current_price+(current_price * take_profit_percent), 2)
        stop_loss_price = current_price * 0.9
        trailAmount = round(current_price * trail_stop_percent, 2)
        triggerPrice = limit_price

        parent = Order()
        parent.orderId = ib.client.getReqId()
        parent.action = "BUY"
        parent.orderType = "LMT"
        parent.totalQuantity = quantity
        parent.lmtPrice = limit_price
        parent.transmit = False
        parent.outsideRth = True
        ###this stuff makes it cancel whole order in 45 sec.  If parent fills, children turn to GTC
        parent.goodTillDate = gtddelta
        parent.tif = "GTD"

        takeProfit = Order()
        takeProfit.orderId = ib.client.getReqId()
        takeProfit.action = "SELL" if parent.action == "BUY" else "BUY"
        takeProfit.outsideRth = True
        takeProfit.orderType = "LMT"
        takeProfit.totalQuantity = quantity
        takeProfit.lmtPrice = take_profit_price
        takeProfit.parentId = parent.orderId
        takeProfit.transmit = False
        takeProfit.tif = "GTC"

        stopLoss = Order()
        stopLoss.orderId = ib.client.getReqId()
        stopLoss.action = "SELL" if parent.action == "BUY" else "BUY"
        stopLoss.orderType = "TRAIL"
        stopLoss.TrailingUnit = 1
        stopLoss.outsideRth = True
        stopLoss.trailingPercent = trailAmount
        # stopLoss.trailStopPrice = limit_price - trailAmount
        stopLoss.totalQuantity = quantity
        stopLoss.parentId = parent.orderId
        stopLoss.transmit = True
        stopLoss.tif = "GTC"

        bracketOrder = [parent, takeProfit, stopLoss]
        parentOrderId = parent.orderId
        childOrderIds = [takeProfit.orderId, stopLoss.orderId]
        parentOrders[parentOrderId] = childOrderIds  # Assign child order IDs to parent order ID key

        ##TODO change ref back
        for o in bracketOrder:
            if orderRef is not None:
                o.orderRef = orderRef
            ib.placeOrder(ticker_contract, o)
            ##changed this 7.25
            # ib.sleep(0)
        print(f"~~~~Placing order: {parent.orderRef} ~~~~~")
        saveOrderIdToFile(parentOrderIdFile, parentOrders)
    except (Exception) as e:
        logger.exception(f"An error occurred while placeBuyBracketOrder.{ticker},: {e}")



# Register the event handler for order status
ib.orderStatusEvent += orderStatusHandler

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

