import asyncio
import datetime
import logging
import os
import re

# import random
from ib_insync import *
from UTILITIES.logger_config import logger
#TODO set up logger
# Initialization and global variables
project_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(project_dir, "errorlog")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.getLogger('ib_insync').setLevel(logging.WARNING)
# Explicitly add the handler for ib_insync
# logging.getLogger('ib_insync').addHandler(logger.handlers[0])

ib = IB()


def ib_reset_and_close_pos():
    # if ib.isConnected():
    #     ib.disconnect()
    # if not ib.isConnected():
    #     print("~~~ Connecting ~~~")
    #     # randomclientID = random.randint(0, 999)#TODO change bac kclientid
    #     try:
    #
    #         ib.connect("192.168.1.119", 7497, clientId=0, timeout=45)
    #         print("connected.")
    #
    #
    #     except (Exception, asyncio.exceptions.TimeoutError) as e:
    #         logging.getLogger().error("Connection error or error reset posistions.: %s", e)
    #         print("~~Connection/closepositions error:", e)
    reset_all()
    logger.info("Reset all positions/closed open orders.")
def reset_all():
    ib.reqGlobalCancel()

    positions = ib.positions()
    # print(positions)
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
        # print(contract)
        ib.placeOrder(contract, close_order)
    logger.info("Reset all positions/closed open orders.")


async def ib_connect():
    if not ib.isConnected():
        print("~~~ Connecting ~~~")
        try:
            await ib.connectAsync("192.168.1.119", 7497, clientId=0, timeout=45)
        except Exception as e:
            logging.getLogger().error("Connection error: %s", e)
            # print("~~Connection error:", e)
    else:
        print("~~~IB already connected. Cannot connect.~~~")


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

def increment_order_ref(order_ref):
    """ Increments the numeric part of the order reference. """
    match = re.search(r'([\s\S]*?)(\d*Rep\.)(\d+)$', order_ref)
    if match:
        prefix = match.group(1)
        rep_part = match.group(2)
        number = int(match.group(3)) + 1
        return f"{prefix}{rep_part}{number}"
    else:
        # If "Rep." doesn't exist, add it and start with 1
        return f"{order_ref}_Rep.1"
async def find_child_orders_with_details(contract, action):
    child_orders_list = []
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




# Define a callback function for the cancelOrderEvent
async def cancel_order(order):
    # print('Attempting to cancel', order)

    trade = await getTrade(order)
    # Create an asyncio Event to wait for the order to be cancelled
    order_cancelled = asyncio.Event()

    # Define a callback function for the cancelOrderEvent
    def make_on_cancel_order_event(order_id):
        def on_cancel_order_event(trade: Trade):
            if trade.order.orderId == order_id:
                # print(f"Order cancellation confirmed for trade: {trade}")
                order_cancelled.set()
        return on_cancel_order_event

    # Subscribe to the cancelOrderEvent
    on_cancel_order_event = make_on_cancel_order_event(order.orderId)
    ib.cancelOrderEvent += on_cancel_order_event
    ib.cancelOrder(order)

    await order_cancelled.wait()  # Wait for the cancellation to be confirmed
    # Unsubscribe from the event to avoid memory leaks
    ib.cancelOrderEvent -= on_cancel_order_event




async def cancel_and_replace_orders(contract,action, CorP, ticker, exp, strike, contract_current_price,
                                    quantity, orderRef, take_profit_percent, trailstop_amount_percent):
    child_orders_objs_list = await find_child_orders_with_details(contract, action)
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
        order_details = {}
        for order in child_orders_objs_list:
            new_order_ref = increment_order_ref(order.orderRef)
            ocaGroup = order.ocaGroup
            trade = await getTrade(order)

            if order.ocaGroup not in order_details:
                order_details[order.ocaGroup] = {}
            if order.orderType == "TRAIL":
                print(order,"aux:",order.auxPrice,"trailstopprice:",order.trailStopPrice)

                order_details[order.ocaGroup]["stopLoss"] = {
                    "type": "TRAIL",
                    "trailingPercent": order.trailingPercent,
                    "auxPrice": order.auxPrice,
                    "trailStopPrice": order.trailStopPrice,
                    "ocaGroup": order.ocaGroup,
                    "parentID":order.parentId,
                    "percentOffset": order.percentOffset,
                    "orderRef": new_order_ref,
                }
            elif order.orderType == "LMT":
                order_details[order.ocaGroup]["takeProfit"] = {
                    "type": "LMT",
                    "limitPrice": order.lmtPrice,
                    "ocaGroup": order.ocaGroup,
                    "parentID": order.parentId,
                    "orderRef": new_order_ref,
                }
            await cancel_order(order)
        # print("ORDER DEATAILS DICT", order_details)
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
            await asyncio.sleep(0)
            # print('waiting for parent to fill before replacing children.')
        # trade.orderStatus.ActiveStates
        await replace_child_orders(order_details, contract,
                                   quantity)


async def replace_child_orders(order_details, contract,
                               quantity):
    try:
        for ocaGroup, child_order_details in order_details.items():
            ticker_contract = contract
            print("dict",order_details[ocaGroup]["stopLoss"])
            quantity = quantity  # Replace with the desired order quantity
            takeProfit = Order()
            takeProfit.orderId = ib.client.getReqId()
            takeProfit.action = "SELL"
            takeProfit.orderType = "LMT"
            takeProfit.totalQuantity = quantity
            takeProfit.outsideRth = True
            takeProfit.transmit = True
            takeProfit.tif = "GTC"
            # Use stored take profit details if available
            if order_details and order_details[ocaGroup]["takeProfit"]["type"] == "LMT":
                takeProfit.lmtPrice = order_details[ocaGroup]["takeProfit"]["limitPrice"]
                takeProfit.ocaGroup = takeProfit.orderId
                # takeProfit.orderRef = order_details[ocaGroup]["takeProfit"]["orderRef"]
                # takeProfit.parentId = order_details[ocaGroup]["takeProfit"]["parentID"]

                takeProfit.orderRef = order_details[ocaGroup]["takeProfit"]["orderRef"]
                # takeProfit.orderRef = ''


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
                stopLoss.trailStopPrice = order_details[ocaGroup]["stopLoss"]["trailStopPrice"]
                stopLoss.trailingPercent = order_details[ocaGroup]["stopLoss"]["trailingPercent"]
                # arbitrary_value = order_details[ocaGroup]["stopLoss"]["trailStopPrice"]
                # trailing_amount = arbitrary_value * stopLoss.trailingPercent
                # For a long position, initial stop price is set below the arbitrary price
                # initial_stop_price = arbitrary_value + trailing_amount
                # stopLoss.trailStopPrice = initial_stop_price  # initial stop price

                # stopLoss.percentOffset = order_details[ocaGroup]["stopLoss"]["percentOffset"]
                #TODO monday make sure this is correct.  it should be same floor as the og order.
                # stopLoss.auxPrice = round(order_details[ocaGroup]["stopLoss"]["trailStopPrice"],2) * (1 - order_details[ocaGroup]["stopLoss"]["trailingPercent"])
                # stopLoss.auxPrice = order_details[ocaGroup]["stopLoss"]["auxPrice"]
                # stopLoss.trailStopPrice = order_details[ocaGroup]["stopLoss"]["trailStopPrice"]  # Use the stored trailStopPrice

                # print("child replaced stoploss aux:",stopLoss.auxPrice)
                # print("child replaced stoploss trailstop:",stopLoss.trailStopPrice)

                stopLoss.ocaGroup = takeProfit.orderId
                # stopLoss.parentId = order_details[ocaGroup]["stopLoss"]["parentID"]
                stopLoss.orderRef = order_details[ocaGroup]["stopLoss"]["orderRef"]
            bracketOrder = [takeProfit, stopLoss]

            # ib.oneCancelsAll(orders=bracketOrder, ocaGroup=ocaGroup, ocaType=2)

            for o in bracketOrder:
                o.ocaType = 2
                ib.placeOrder(ticker_contract, o)
            trade = await getTrade(takeProfit)
            trade2 = await getTrade(stopLoss)
            print("TRADE2 STOPLOSS:" ,trade2)
            while  trade.isDone() and  trade2.isDone():
                await asyncio.sleep(0)
            print("TRADE2 STOPLOSS:" ,trade2)

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
        take_profit_percent=3,
        trailstop_amount_percent=3,
        check_opposite_orders=True):
    if take_profit_percent == None:
        take_profit_percent = 3
    if trailstop_amount_percent == None:
        trailstop_amount_percent =3

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
                gtddelta = (datetime.datetime.now() + datetime.timedelta(seconds=10)).strftime("%Y%m%d %H:%M:%S")

                contract_current_price = round(contract_current_price, 2)
                quantity = quantity  # Replace with the desired order quantity
                limit_price = contract_current_price  # Replace with your desired limit price
                take_profit_price = round(
                    contract_current_price + (contract_current_price * (take_profit_percent/100)), 2
                )  # Replace with your desired take profit price

                auxPrice = round(
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
                # stopLoss.trailStopPrice = limit_price - trailAmount
                # stopLoss.trailStopPrice = .5
                #I  think the big issue HERE was that percent was a whole number like 3, so it was doin some weird thinkg.
                stopLoss.trailStopPrice = round(contract_current_price *((100-trailstop_amount_percent)/100),2)
                # stopLoss.trailingPercent = 99
                stopLoss.trailingPercent = trailstop_amount_percent

                # stopLoss.trailingPercent = trailAmount
                stopLoss.totalQuantity = quantity
                stopLoss.parentId = parent.orderId
                stopLoss.ocaGroup = parent.orderId
                stopLoss.outsideRth = True
                stopLoss.transmit = True
                stopLoss.tif = "GTC"

                bracketOrder = [parent, takeProfit, stopLoss]

                ##TODO change ref back
                for o in bracketOrder:
                    o.ocaType = 2
                    if orderRef is not None:

                        o.orderRef = orderRef
                    # ib.sleep(1)
                    ib.placeOrder(ticker_contract, o)
                print(f"~~~~Placed OptionOrder.  Parent orderRef: {parent.orderRef} ~~~~~")

                ##changed this 7.25
                # await ib.sleep(0)
                # ib.sleep(0)
                # saveOrderIdToFile(parentOrderIdFile, parentOrders)
                # print(f"~~~~Order Placed: {parent.orderRef} ~~~~~")
                trade = await getTrade(parent)
                while not trade.isDone():
                    # print(trade.orderStatus)
                    await asyncio.sleep(0)
                return trade
            except (Exception, asyncio.exceptions.TimeoutError) as e:
                logger.exception(f"An error occurred while optionbracketorder.{ticker},: {e}")
    else:
        logger.warning(f"Too many open orders (7+){ticker_contract} {orderRef} .  Skipping order placement.")


# ib.disconnect()
async def can_place_new_order(ib, contract, threshold=7):
    # Fetch open orders
    open_trades =  ib.openTrades()
    # print(open_trades)

    # print(contract[0].conId)
    # Count orders for the specified contract
    count = sum(1 for trade in open_trades if trade.contract.conId == contract[0].conId)
    # print(f"{count} open orders for {contract}.")
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
        stopLoss.trailingPercent = trail_stop_percent
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
print(__name__)
if __name__ == "__main__":
    print("ib name is main")
    ib_reset_and_close_pos()
