import asyncio
import datetime
import logging
import os
import re

# import random
from ib_insync import *

from UTILITIES.logger_config import logger

# TODO set up logger
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

    if not ib.isConnected():
        print("~~~ Connecting ~~~")
        # randomclientID = random.randint(0, 999)#TODO change bac kclientid
        try:
#119=toiuchscreen dell.  109 is studio
            ib.connect("192.168.1.109", 7497, clientId=0, timeout=45)
            print("connected.")


        except (Exception, asyncio.exceptions.TimeoutError) as e:
            logging.getLogger().error("Connection error or error reset posistions.: %s", e)
            print("~~Connection/closepositions error:", e)
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
            await ib.connectAsync("192.168.1.109", 7497, clientId=0, timeout=45)
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

# TODO swapped these cancels_ordres because sometimes it seems order is getting filled after trying to cancel, adn i have mismatching positions/open trades.
async def cancel_oca_group_orders(oca_group_orders):
    cancellation_events = []
    for order in oca_group_orders:
        order_cancelled = asyncio.Event()
        cancellation_events.append((order, order_cancelled))

        def make_on_cancel_order_event(order_id, event):
            def on_cancel_order_event(trade: Trade):
                if trade.order.orderId == order_id:
                    event.set()
            return on_cancel_order_event

        on_cancel_order_event = make_on_cancel_order_event(order.orderId, order_cancelled)
        ib.cancelOrderEvent += on_cancel_order_event
        ib.cancelOrder(order)

    # Wait for all cancellations to complete
    await asyncio.gather(*(event.wait() for _, event in cancellation_events))
    for order, event in cancellation_events:
        ib.cancelOrderEvent -= make_on_cancel_order_event(order.orderId, event)

        # New approach to calculate remaining quantities for each OCA group
    oca_group_remaining_qty = {}
    fills = ib.fills()
    for order in oca_group_orders:
        order_fills = [fill for fill in fills if fill.execution.orderId == order.orderId]
        filled_qty = sum(fill.execution.shares for fill in order_fills)
        remaining_qty = order.totalQuantity - filled_qty
        print("filled qty:",filled_qty,"remainig qty:", remaining_qty)

        oca_group = order.ocaGroup
        oca_group_remaining_qty[oca_group] = remaining_qty

    return oca_group_remaining_qty


async def cancel_and_replace_orders(contract, action, CorP, ticker, exp, strike, contract_current_price, quantity,
                                    orderRef, take_profit_percent, trailstop_amount_percent):
    child_orders_objs_list = await find_child_orders_with_details(contract, action)
    if not child_orders_objs_list:
        await placeOptionBracketOrder(CorP, ticker, exp, strike, contract_current_price, quantity, orderRef,
                                      take_profit_percent, trailstop_amount_percent, check_opposite_orders=False)

    else:
        oca_group_remaining_qty = await cancel_oca_group_orders(child_orders_objs_list)
        order_details = {}
        for order in child_orders_objs_list:
            oca_group = order.ocaGroup
            remaining_qty = oca_group_remaining_qty.get(oca_group, 0)
            # # new_order_ref = increment_order_ref(order.orderRef)
            # ocaGroup = order.ocaGroup
            # trade = await getTrade(order)
            # await cancel_order(order)

            if order.ocaGroup not in order_details:
                order_details[order.ocaGroup] = {}
            if order.orderType == "TRAIL":
                # print(order,"aux:",order.auxPrice,"trailstopprice:",order.trailStopPrice)

                order_details[order.ocaGroup]["stopLoss"] = {
                    "type": "TRAIL",
                    "trailingPercent": order.trailingPercent,
                    "auxPrice": order.auxPrice,
                    "trailStopPrice": order.trailStopPrice,
                    # "ocaGroup": order.ocaGroup,
                    "parentID": order.parentId,
                    "percentOffset": order.percentOffset,
                    "orderRef": order.orderRef,
                    "remainingQty": remaining_qty

                }
            elif order.orderType == "LMT":
                order_details[order.ocaGroup]["takeProfit"] = {
                    "type": "LMT",
                    "limitPrice": order.lmtPrice,
                    # "ocaGroup": order.ocaGroup,
                    "parentID": order.parentId,
                    "orderRef": order.orderRef,
                    "remainingQty": remaining_qty

                }
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
        while True:
            if parenttrade.isDone() or parenttrade.orderStatus.status == "Inactive":
                break
            await asyncio.sleep(0)

        # trade.orderStatus.ActiveStates
        await replace_child_orders(order_details, contract)


async def replace_child_orders(order_details, contract):
    try:
        for ocaGroup, child_order_details in order_details.items():
            quantity = order_details[ocaGroup]["takeProfit"]["remainingQty"]

            ticker_contract = contract
            qualified_contract = await ib.qualifyContractsAsync(ticker_contract)

            # print("dict",order_details[ocaGroup]["stopLoss"])
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
                stopLoss.ocaGroup = takeProfit.orderId
                stopLoss.orderRef = order_details[ocaGroup]["stopLoss"]["orderRef"]
            bracketOrder = [takeProfit, stopLoss]

            for o in bracketOrder:
                o.ocaType = 2
                ib.placeOrder(qualified_contract[0], o)

            while True:
                stoplossTrade = await getTrade(stopLoss)
                takeprofitTrade = await getTrade(takeProfit)
                if stoplossTrade is not None and takeprofitTrade is not None:
                    # You might want to add additional checks here, depending on your logic
                    break
                await asyncio.sleep(0)  # Properly yield control to the event loop

            # print("TRADE1 Take Profit:" ,trade)
            # # while  trade.isDone() and  trade2.isDone():
            # #     await asyncio.sleep(0)
            # print("TRADE2 STOPLOSS:" ,trade2)

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
        check_opposite_orders=True,parent_tif="GTD"):
    if take_profit_percent == None:
        take_profit_percent = 3
    if trailstop_amount_percent == None:
        trailstop_amount_percent = 3

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
                    contract_current_price + (contract_current_price * (take_profit_percent / 100)), 2
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
                parent.orderRef = orderRef

                parent.transmit = False
                parent.outsideRth = True
                ###this stuff makes it cancel whole order in 45 sec.  If parent fills, children turn to GTC
                parent.tif = parent_tif
                parent.goodTillDate = gtddelta

                takeProfit = Order()
                takeProfit.orderId = ib.client.getReqId()
                takeProfit.action = "SELL"
                takeProfit.orderType = "LMT"
                takeProfit.totalQuantity = quantity
                takeProfit.lmtPrice = take_profit_price
                takeProfit.parentId = parent.orderId
                takeProfit.orderRef = orderRef + "_takeprof"

                takeProfit.outsideRth = True
                takeProfit.ocaGroup = parent.orderId
                takeProfit.transmit = False
                takeProfit.tif = "GTC"

                stopLoss = Order()
                stopLoss.orderId = ib.client.getReqId()
                stopLoss.action = "SELL"
                stopLoss.orderType = "TRAIL"
                stopLoss.TrailingUnit = 1
                stopLoss.orderRef = orderRef + "_trail"

                # stopLoss.trailStopPrice = limit_price - trailAmount
                # stopLoss.trailStopPrice = .5
                # I  think the big issue HERE was that percent was a whole number like 3, so it was doin some weird thinkg.
                stopLoss.trailStopPrice = round(contract_current_price * ((100 - trailstop_amount_percent) / 100), 2)
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

                    # ib.sleep(1)
                    ib.placeOrder(ticker_contract, o)
                print(f"~~~~Placed OptionOrder.  Parent orderRef: {parent.orderRef} ~~~~~")

                ##changed this 7.25
                # await ib.sleep(0)
                # ib.sleep(0)
                # saveOrderIdToFile(parentOrderIdFile, parentOrders)
                # print(f"~~~~Order Placed: {parent.orderRef} ~~~~~")
                while True:
                    parenttrade = await getTrade(parent)
                    if parenttrade is not None:
                        # You might want to add additional checks here, depending on your logic
                        break
                    await asyncio.sleep(0)  # Properly yield control to the event loop

                # while not trade.isDone():
                #     # print(trade.orderStatus)
                #     await asyncio.sleep(0)
                return parenttrade
            except (Exception, asyncio.exceptions.TimeoutError) as e:
                logger.exception(f"An error occurred while optionbracketorder.{ticker},: {e}")
    else:
        logger.warning(f"Too many open orders (18+){ticker_contract} {orderRef} .  Skipping order placement.")


# ib.disconnect()
async def can_place_new_order(ib, contract, threshold=18):
    # Fetch open orders
    open_trades = ib.openTrades()
    # print(open_trades)
    # print("OpenTradesfor contract:",open_trades)
    # print(contract[0].conId)
    # Count orders for the specified contract
    count = sum(1 for trade in open_trades if trade.contract.conId == contract[0].conId)
    print(f"{count} open orders for {contract[0].localSymbol}.")
    # Return True if below threshold, False otherwise
    return count < threshold


async def placeBuyBracketOrder(ticker, current_price,
                               quantity=1,
                               orderRef=None,
                               take_profit_percent=.0003,
                               trail_stop_percent=.0002):
    trail_stop_percent = trail_stop_percent / 100
    take_profit_percent = take_profit_percent / 100
    print(f"~~~~~Placing {ticker} BuyBracket order~~~~~")
    try:
        if take_profit_percent == None:
            take_profit_percent = .003
        if trail_stop_percent == None:
            trail_stop_percent = .002
        gtddelta = (datetime.datetime.now() + datetime.timedelta(seconds=180)).strftime("%Y%m%d %H:%M:%S")

        ticker_symbol = ticker
        ticker_contract = Stock(ticker_symbol, "SMART", "USD")
        # awaitib.qualifyContractsAsync(ticker_contract)

        current_price = current_price
        quantity = quantity
        limit_price = current_price
        take_profit_price = round(current_price + (current_price * take_profit_percent), 2)
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
