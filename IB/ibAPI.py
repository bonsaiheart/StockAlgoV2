import asyncio
import datetime
import logging
import os
# import random
from ib_insync import *
from mpmath import rand

from UTILITIES.logger_config import logger

# Initialization and global variables
project_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(project_dir, "errorlog")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.getLogger('ib_insync').setLevel(logging.WARNING)
# Explicitly add the handler for ib_insync
logging.getLogger('ib_insync').addHandler(logger.handlers[0])
# TODO note that when i dont use the order handler += at the end, thats when orders seem to get mismatched/not transmitted etc.
# ib_insync.util.getLoop()
ib = IB()

# def handle_exception(loop, context):
#     msg = context.get("exception", context["message"])
#     logging.error(f"Caught exception: {msg}")
#     logging.info("Shutting down...")

# TODO order handling for "cannot both sides of ordr" error
def ib_reset_and_close_pos():
    if not ib.isConnected():
        print("~~~ Connecting ~~~")
        # randomclientID = random.randint(0, 999)#TODO change bac kclientid
        try:

            ib.connect("192.168.1.119", 7497, clientId=5, timeout=45)
            # uncomment to not clear
            print("connected?")
            reset_all()
            logger.info("Reset all positions/closed open orders.")
        except (Exception, asyncio.exceptions.TimeoutError) as e:
            logging.getLogger().error("Connection error or error reset posistions.: %s", e)
            print("~~Connection/closepositions error:", e)
def reset_all():
    ib.reqGlobalCancel()

    positions = ib.positions()
    print(positions)
    for position in positions:
        contract = position.contract
        # contract = ib.qualifyContracts(contract)[0]
        # print(contract)
        size = position.position

        # Determine the action ('BUY' to close a short position, 'SELL' to close a long position)
        action = 'BUY' if size < 0 else 'SELL'

        # Create a market order to close the position
        close_order = MarketOrder(action, abs(size))
        contract.exchange = 'SMART'  # Specify the exchange
        # Send the order
        print(contract)
        ib.placeOrder(contract, close_order)
    logger.info("Reset all positions/closed open orders.")


# ib_reset_and_close_pos()


def connection_stats():
    print(ib.isConnected())

# def handle_exception(loop, context):
#     msg = context.get("exception", context["message"])
#     logging.error(f"Caught exception: {msg}")
#     logging.info("Shutting down...")
#     asyncio.create_task(loop.shutdown())
# loop = asyncio.get_event_loop()
# loop.set_exception_handler(handle_exception)


# ib.disconnect()
async def ib_connect():
    if not ib.isConnected():
        print("~~~ Connecting ~~~")
        # randomclientID = random.randint(0, 999)#TODO change bac kclientid
        # ib_insync.util.getLoop()

        try:
            await ib.connectAsync("192.168.1.119", 7497, clientId=0, timeout=45)

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
    child_orders_list = []
    print('find cchild or')
    # Filter open orders to find child orders for the given contract
    for trade in ib.openTrades():

        # trade = await getTrade(order)
        # print("action",action)
        # print(order.parentId , order.action,trade.contract)
        if trade.order.action == 'SELL' and trade.contract == contract:
            # print("childorders trades",trade)
            child_orders_list.append(trade.order)

    # print('childorders trade obj:', child_orders)
    return child_orders_list


async def getTrade(order):
    trade = next((trade for trade in ib.trades() if trade.order is order), None)

    return trade


# With trade.isDone() can the be checked if the order is stil running or not.

#
# def handle_exception(loop, context):
#     msg = context.get("exception", context["message"])
#     print('handleexception: ', msg)
#     logging.error(f"Caught exception: {msg}")
#     logging.info("Shutting down...")
#     asyncio.create_task(loop.shutdown())




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
    await order_cancelled.wait()  # Wait for the cancellation to be confirmed
    #     # print("Order cancelled")

    # Unsubscribe from the event to avoid memory leaks
    ib.cancelOrderEvent -= on_cancel_order_event




async def cancel_and_replace_orders(contract,action, CorP, ticker, exp, strike, contract_current_price,
                                    quantity, orderRef, take_profit_percent, trailstop_amount_percent):
    child_orders_objs_list = await find_child_orders_with_details(contract, action)
    print('1')

    if not child_orders_objs_list:
        await placeOptionBracketOrder(
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
    else:
    # Store details of trailing stop and take profit orders
        order_details = {}
        for order in child_orders_objs_list:
            ocaGroup = order.ocaGroup
            # print(ocaGroup)
            print(f"ocaGroup value: '{ocaGroup}'")

            trade = await getTrade(order)
            # print("TRAFDSISAFIASDFASDIFAD S",trade)
            # print(order.orderType,order)
            if order.ocaGroup not in order_details:
                order_details[order.ocaGroup] = {}
            if order.orderType == "TRAIL":
                order_details[order.ocaGroup]["stopLoss"] = {
                    "type": "TRAIL",
                    "trailAmount": order.auxPrice,
                    "triggerPrice": order.trailStopPrice,
                    "ocaGroup": order.ocaGroup,
                    "orderRef": order.orderRef,
                    # "parentID":order.parentId
                }
            elif order.orderType == "LMT":
                # print(order_details)
                order_details[order.ocaGroup]["takeProfit"] = {
                    "type": "LMT",
                    "limitPrice": order.lmtPrice,
                    "ocaGroup": order.ocaGroup,
                    "orderRef": order.orderRef,
                    # "parentID":order.parentId
                }
            # elif order.orderType =
            print('cancelling order:', order)
            await cancel_order(order)
            print(order,"cancelled")
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
            print(parenttrade.orderStatus)
            print('sleeping 1 sec')
            await asyncio.sleep(1)
            print('waiting for parent to fill before replacing children.')
            # print(parenttrade.isDone())

        # trade.orderStatus.ActiveStates
        print('replacing child orders.',order_details)

        await replace_child_orders(order_details, contract,
                                   quantity)


async def replace_child_orders(order_details, contract,
                               quantity):
    try:
        for ocaGroup, child_order_details in order_details.items():
            # print(order_details.items())
            ticker_contract = contract
#TODO chnage qtty back
            quantity = 1  # Replace with the desired order quantity
            print(ocaGroup)
            # This will be our main or "parent" order
            # parent = Order()
            # parent.orderId = parent_id
            # parent.orderId = ib.client.getReqId()
            # new_ocaGroup = (int(ocaGroup)+100)//10
            # new_ocaGroup = ["takeProfit"]["ocaGroup"]
            # ocaGroup = new_ocaGroup
            takeProfit = Order()

            takeProfit.orderId = ib.client.getReqId()

            takeProfit.action = "SELL"
            takeProfit.orderType = "LMT"
            takeProfit.totalQuantity = quantity
            # takeProfit.parentId = parent.orderId
            takeProfit.outsideRth = True
            takeProfit.transmit = True
            takeProfit.tif = "GTC"
            # Use stored take profit details if available
            if order_details and order_details[ocaGroup]["takeProfit"]["type"] == "LMT":
                takeProfit.lmtPrice = order_details[ocaGroup]["takeProfit"]["limitPrice"]
                takeProfit.ocaGroup = takeProfit.orderId

                # takeProfit.orderRef = order_details[parent_id]["takeProfit"]["orderRef"]
                takeProfit.orderRef = 'replaced takeprof'


            stopLoss = Order()
            stopLoss.action = "SELL"
            stopLoss.orderType = "TRAIL"
            stopLoss.TrailingUnit = 1
            stopLoss.totalQuantity = quantity
            # stopLoss.parentId = parent.orderId
            stopLoss.orderId = ib.client.getReqId()
            stopLoss.outsideRth = True
            stopLoss.transmit = True
            stopLoss.tif = "GTC"
            # Use stored trailing stop details if available
            if order_details and order_details[ocaGroup]["stopLoss"]["type"] == "TRAIL":
                stopLoss.auxPrice = order_details[ocaGroup]["stopLoss"]["trailAmount"]
                stopLoss.trailStopPrice = order_details[ocaGroup]["stopLoss"]["triggerPrice"]
                stopLoss.ocaGroup = takeProfit.orderId
                stopLoss.orderRef = order_details[ocaGroup]["stopLoss"]["orderRef"]
                stopLoss.orderRef = 'replaced stoploss'

            stopLoss.orderId = ib.client.getReqId()
            bracketOrder = [takeProfit, stopLoss]

            ib.oneCancelsAll(orders=bracketOrder, ocaGroup=takeProfit.orderId, ocaType=2)


            ##TODO change ref back
            print(f"~~~~Replacing cancelled for ocaGroup: {ocaGroup} ~~~~~")
            for o in bracketOrder:
                # o.ocaType = 2
                # o.clientId = 2
                # o.ocaGroup = new_ocaGroup
                ib.placeOrder(ticker_contract, o)
            trade = await getTrade(takeProfit)
            trade2 = await getTrade(stopLoss)

            while trade.isDone() and trade2.isDone():
                await asyncio.sleep(1)

            # print(await getTrade(takeProfit))
    except (Exception, asyncio.exceptions.TimeoutError) as e:
        logger.exception(f"An error occurred while replace child orders.{ticker_contract},: {e}")

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
    gtddelta = (datetime.datetime.now() + datetime.timedelta(seconds=180)).strftime("%Y%m%d %H:%M:%S")

    print("~~~Placing order~~~:")
    print("~~~gtddelta:", gtddelta)
    ticker_contract = Option(ticker, exp, strike, CorP, "SMART")
    qualified_contract = await ib.qualifyContractsAsync(ticker_contract)

    if await can_place_new_order(ib, qualified_contract):
        if check_opposite_orders:
            try:
                print("Checking opposite orders.")
                action = "SELL"  # Adjust based on your logic
                await cancel_and_replace_orders(ticker_contract, action, CorP, ticker, exp, strike,
                                                contract_current_price,
                                                quantity, orderRef, take_profit_percent, trailstop_amount_percent)
                # ib.sleep(5)
            except (Exception, asyncio.exceptions.TimeoutError) as e:
                logger.exception(f"An error occurred while optionbracketorder.{ticker},: {e}")
        else:
            try:
                contract_current_price = round(contract_current_price, 2)
                quantity = quantity  # Replace with the desired order quantity
                limit_price = contract_current_price  # Replace with your desired limit price
                take_profit_price = round(
                    contract_current_price + (contract_current_price * take_profit_percent), 2
                )  # Replace with your desired take profit price

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
                takeProfit.ocaGroup = parent.orderId
                takeProfit.transmit = False
                takeProfit.tif = "GTC"

                stopLoss = Order()
                stopLoss.orderId = ib.client.getReqId()
                stopLoss.action = "SELL"
                stopLoss.orderType = "TRAIL"
                stopLoss.TrailingUnit = 1
                # stopLoss.auxPrice = trailAmount
                # stopLoss.trailStopPrice = limit_price - trailAmount
                stopLoss.trailingPercent = trailAmount
                stopLoss.totalQuantity = quantity
                stopLoss.parentId = parent.orderId
                stopLoss.ocaGroup = parent.orderId
                stopLoss.outsideRth = True
                stopLoss.transmit = True
                stopLoss.tif = "GTC"

                bracketOrder = [parent, takeProfit, stopLoss]

                ##TODO change ref back
                print(f"~~~~Placing Order: {parent.orderRef} ~~~~~")
                for o in bracketOrder:
                    o.ocaType = 2
                    if orderRef is not None:

                        o.orderRef = orderRef
                    # ib.sleep(1)
                    ib.placeOrder(ticker_contract, o)
                ##changed this 7.25
                # await ib.sleep(0)
                # ib.sleep(0)
                # saveOrderIdToFile(parentOrderIdFile, parentOrders)
                print(f"~~~~Order Placed: {parent.orderRef} ~~~~~")
                trade = await getTrade(parent)
                while not trade.isDone():
                    # print(trade.orderStatus)
                    await asyncio.sleep(1)
                return trade
            except (Exception, asyncio.exceptions.TimeoutError) as e:
                logger.exception(f"An error occurred while optionbracketorder.{ticker},: {e}")
    else:
        logger.warning(f"Too many open orders (7+){ticker_contract} {orderRef} .  Skipping order placement.")


# ib.disconnect()
async def can_place_new_order(ib, contract, threshold=7):
    # Fetch open orders
    open_trades =  ib.openTrades()
    print(open_trades)

    # print(contract[0].conId)
    # Count orders for the specified contract
    count = sum(1 for trade in open_trades if trade.contract.conId == contract[0].conId)
    print(f"{count} open orders for {contract}.")
    # Return True if below threshold, False otherwise
    return count < threshold

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
        # awaitib.qualifyContractsAsync(ticker_contract)

        current_price = current_price
        quantity = quantity
        limit_price = current_price
        take_profit_price = round(current_price+(current_price * take_profit_percent), 2)
        trailAmount = round(current_price * trail_stop_percent, 2)

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

        ##TODO change ref back
        for o in bracketOrder:
            if orderRef is not None:
                o.orderRef = orderRef
            ib.placeOrder(ticker_contract, o)
            ##changed this 7.25
            # ib.sleep(0)
        print(f"~~~~Placing order: {parent.orderRef} ~~~~~")
    except (Exception) as e:
        logger.exception(f"An error occurred while placeBuyBracketOrder.{ticker},: {e}")



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
        gtddelta = (datetime.datetime.now() + datetime.timedelta(seconds=180)).strftime("%Y%m%d %H:%M:%S")

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
        # parentOrders[parentOrderId] = {}  # Create an empty dictionary for child orders of this parent order
        # saveOrderIdToFile(parentOrderIdFile, parentOrders)

        for o in bracketOrder:
            if orderRef != None:
                o.orderRef = orderRef
            # childOrderId = ib.placeOrder(ticker_contract, o)
            # parentOrders[parentOrderId][childOrderId] = o

        # ib.sleep(0)

    except (Exception, asyncio.exceptions.TimeoutError) as e:
        logging.error("PlaceCallBracketOrder error: %s", e)

#
# # Load previously stored parent order IDs and child order IDs
# parentOrders = retrieveOrderIdFromFile(parentOrderIdFile)
#
# # Register the event handler for order status
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

