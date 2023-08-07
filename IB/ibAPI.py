import asyncio
import datetime
import logging
import os
import random
import signal
from ib_insync import *
import signal
from Task_Queue.task_queue_cellery_bossman import app as app

project_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory where the script is
log_dir = os.path.join(project_dir, "errorlog")  # Builds the path to the errorlog directory

# Create the directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, "error_ib.log")  # Builds the path to the log file

# Set up logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M",
)
# Initialize IB object
ib = IB()

# File paths for storing order IDs
parentOrderIdFile = "IB/parent_order_ids.txt"

# Global variables
gtddelta = (datetime.datetime.now() + datetime.timedelta(seconds=180)).strftime("%Y%m%d %H:%M:%S")
parentOrders = {}
#TODO order handling for "cannot both sides of ordr" error

# ...
# @app.task
# def close_orders():
#     global parentOrders
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
        print("Connecting")
        randomclientID = random.randint(0, 999)
        try:
            await ib.connectAsync("192.168.1.119", 7497, clientId=1)
        except (Exception, asyncio.exceptions.TimeoutError) as e:
            logging.getLogger().error("Connection error: %s", e)
            print("Connection error:", e)
    else:
        print("IB already connected.")


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

def place_option_order_sync(CorP, ticker, exp, strike, contract_current_price, quantity, orderRef):
    try:
        ib.placeOptionBracketOrder(corP=CorP,
                                   ticker=ticker,
                                   exp=exp,
                                   strike=strike,
                                   contract_current_price=contract_current_price,
                                   quantity=quantity,
                                   orderRef=orderRef)
    except Exception as e:
        print(f'Error occurred while placing order: {e}')

def orderStatusHandler(orderStatus: OrderStatus):
    global parentOrders

    print("printorderstatus.filled:", orderStatus.filled)
    if orderStatus.status == "filled":
        parentOrderId = orderStatus.orderStatus.parentId
        childOrderId = orderStatus.orderStatus.orderId
        if parentOrderId in parentOrders and childOrderId in parentOrders[parentOrderId]:
            parentOrders[parentOrderId].pop(childOrderId, None)


def placeOptionBracketOrder(
    CorP,
    ticker,
    exp,
    strike,
    contract_current_price,
    quantity,
    orderRef=None,
    custom_takeprofit=None,
    custom_trailamount=None,
):
    print("Placing order:")

    try:
        # print(ticker, exp, strike, contract_current_price)
        # print(type(ticker))
        # print(type(exp))
        # print(type(strike))
        # print(type(contract_current_price))
        ## needed to remove 'USD' for option
        ticker_contract = Option(ticker, exp, strike, CorP, "SMART")
        # ib.qualifyContracts(ticker_contract)
        contract_current_price = round(contract_current_price, 2)
        quantity = quantity  # Replace with the desired order quantity
        limit_price = contract_current_price  # Replace with your desired limit price
        if custom_takeprofit is not None:
            take_profit_price = round(
                contract_current_price * custom_takeprofit, 2
            )  # Replace with your desired take profit price

        else:
            take_profit_price = round(contract_current_price * 1.05, 2)  # Replace with your desired take profit price
        stop_loss_price = contract_current_price * 0.9  # Replace with your desired stop-loss price
        if custom_trailamount is not None:
            trailAmount = round(
                contract_current_price * custom_trailamount, 2
            )  # Replace with your desired trailing stop percentage

        else:
            trailAmount = round(contract_current_price * 0.3, 2)  # Replace with your desired trailing stop percentage

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

        bracketOrder = [parent, takeProfit, stopLoss]
        parentOrderId = parent.orderId
        parentOrders[parentOrderId] = {"parent": parentOrderId}  # Create an entry for the parent order ID

        childOrderId = [takeProfit.orderId, stopLoss.orderId]
        parentOrders[parentOrderId] = childOrderId  # Assign child order IDs to parent order ID key
##TODO change ref back
        for o in bracketOrder:
            if orderRef is not None:
                o.orderRef = orderRef
            print(ib.placeOrder(ticker_contract, o))
            ib.sleep(0)

            print(o.orderId)
##changed this 7.25
            ib.sleep(0)
        print("ORDERPLACED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        saveOrderIdToFile(parentOrderIdFile, parentOrders)

    except (Exception, asyncio.exceptions.TimeoutError) as e:
        logging.exception("PlaceCallBracketOrder error.")
        logging.getLogger().error("Placeoptionbracket error: %s", e)

        # ib.disconnect()


def placeBuyBracketOrder(ticker, current_price,
    quantity=1,
    orderRef=None,
    custom_takeprofit=None,
    custom_trailamount=None):
    print(f"Placing {ticker} BuyBracket order")
    try:
        ticker_symbol = ticker
        ticker_contract = Stock(ticker_symbol, "SMART", "USD")
        # ib.qualifyContracts(ticker_contract)

        current_price = current_price
        quantity = quantity
        limit_price = current_price
        take_profit_price = round(current_price * 1.003, 2)
        stop_loss_price = current_price * 0.9
        trailAmount = round(current_price * 0.002, 2)
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

        stopLoss = Order()
        stopLoss.orderId = ib.client.getReqId()
        stopLoss.action = "SELL" if parent.action == "BUY" else "BUY"
        stopLoss.orderType = "TRAIL"
        stopLoss.TrailingUnit = 1
        stopLoss.outsideRth = True
        stopLoss.auxPrice = trailAmount
        stopLoss.trailStopPrice = limit_price - trailAmount
        stopLoss.totalQuantity = quantity
        stopLoss.parentId = parent.orderId
        stopLoss.transmit = True

        bracketOrder = [parent, takeProfit, stopLoss]
        parentOrderId = parent.orderId
        parentOrders[parentOrderId] = {"parent": parentOrderId}  # Create an entry for the parent order ID

        childOrderId = [takeProfit.orderId, stopLoss.orderId]
        parentOrders[parentOrderId] = childOrderId  # Assign child order IDs to parent order ID key
        ##TODO change ref back
        for o in bracketOrder:
            if orderRef is not None:
                o.orderRef = orderRef
            print(ib.placeOrder(ticker_contract, o))
            ##changed this 7.25
            ib.sleep(0)
        print("ORDERPLACED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        saveOrderIdToFile(parentOrderIdFile, parentOrders)

    except (Exception, asyncio.exceptions.TimeoutError) as e:
        logging.exception("PlaceBuyOrder error.")
        logging.getLogger().error("PlaceBuyOrder error: %s", e)


def placeSellBracketOrder(ticker, current_price):
    try:
        ticker_symbol = ticker
        ticker_contract = Stock(ticker_symbol, "SMART", "USD")
        ib.qualifyContracts(ticker_contract)

        current_price = current_price
        quantity = 5
        limit_price = current_price
        take_profit_price = round(current_price * 1.003, 2)
        stop_loss_price = current_price * 0.9
        trailAmount = round(current_price * 0.002, 2)
        triggerPrice = limit_price

        parent = Order()
        parent.orderId = ib.client.getReqId()
        parent.action = "SELL"
        parent.orderType = "LMT"
        parent.totalQuantity = quantity
        parent.outsideRth = True
        parent.tif = "GTD"
        parent.goodTillDate = gtddelta

        parent.lmtPrice = limit_price
        parent.transmit = False

        takeProfit = Order()
        takeProfit.orderId = ib.client.getReqId()
        takeProfit.outsideRth = True
        takeProfit.action = "BUY"
        takeProfit.orderType = "LMT"
        takeProfit.totalQuantity = quantity
        takeProfit.lmtPrice = take_profit_price
        takeProfit.parentId = parent.orderId
        takeProfit.transmit = False

        stopLoss = Order()
        stopLoss.orderId = ib.client.getReqId()
        stopLoss.action = "BUY"
        stopLoss.orderType = "TRAIL"
        stopLoss.outsideRth = True
        stopLoss.TrailingUnit = 1
        stopLoss.auxPrice = trailAmount
        stopLoss.trailStopPrice = limit_price - trailAmount
        stopLoss.totalQuantity = quantity
        stopLoss.parentId = parent.orderId
        stopLoss.transmit = True

        bracketOrder = [parent, takeProfit, stopLoss]

        parentOrderId = ib.placeOrder(ticker_contract, parent)
        parentOrders[parentOrderId] = {}  # Create an empty dictionary for child orders of this parent order
        saveOrderIdToFile(parentOrderIdFile, parentOrders)

        for o in bracketOrder:
            childOrderId = ib.placeOrder(ticker_contract, o)
            parentOrders[parentOrderId][childOrderId] = o

        ib.sleep(0)

    except (Exception, asyncio.exceptions.TimeoutError) as e:
        logging.error("SellBracketOrder error: %s", e)


def placeCallBracketOrder(
    ticker, exp, strike, current_price, quantity, orderRef=None, custom_takeprofit=None, custom_trailamount=None
):
    try:
        ticker_symbol = ticker
        # print(ticker, exp, strike, current_price)
        # print(type(ticker))
        # print(type(exp))
        # print(type(strike))
        # print(type(current_price))
        ## needed to remove 'USD' for option
        ticker_contract = Option(ticker, exp, strike, "C", "SMART")
        ib.qualifyContracts(ticker_contract)
        current_price = round(current_price, 2)
        quantity = quantity
        limit_price = current_price
        if custom_takeprofit is not None:
            take_profit_price = round(current_price * custom_takeprofit, 2)
        else:
            take_profit_price = round(current_price * 1.15, 2)
        stop_loss_price = current_price * 0.9
        if custom_trailamount is not None:
            trailAmount = round(current_price * custom_trailamount, 2)
        else:
            trailAmount = round(current_price * 0.05, 2)

        triggerPrice = limit_price

        parent = Order()
        parent.orderId = ib.client.getReqId()
        parent.action = "BUY"
        parent.orderType = "LMT"
        parent.totalQuantity = quantity
        parent.lmtPrice = limit_price
        parent.transmit = False
        parent.outsideRth = True
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

        stopLoss = Order()
        stopLoss.orderId = ib.client.getReqId()
        stopLoss.action = "SELL"
        stopLoss.orderType = "TRAIL"
        stopLoss.outsideRth = True
        stopLoss.TrailingUnit = 1
        stopLoss.auxPrice = trailAmount
        stopLoss.trailStopPrice = limit_price - trailAmount
        stopLoss.totalQuantity = quantity
        stopLoss.parentId = parent.orderId
        stopLoss.transmit = True

        bracketOrder = [parent, takeProfit, stopLoss]

        parentOrderId = ib.placeOrder(ticker_contract, parent)
        parentOrders[parentOrderId] = {}  # Create an empty dictionary for child orders of this parent order
        saveOrderIdToFile(parentOrderIdFile, parentOrders)

        for o in bracketOrder:
            if orderRef != None:
                o.orderRef = orderRef
            childOrderId = ib.placeOrder(ticker_contract, o)
            parentOrders[parentOrderId][childOrderId] = o

        ib.sleep(0)

    except (Exception, asyncio.exceptions.TimeoutError) as e:
        logging.error("PlaceCallBracketOrder error: %s", e)


# Load previously stored parent order IDs and child order IDs
parentOrders = retrieveOrderIdFromFile(parentOrderIdFile)

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

