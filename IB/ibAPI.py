import asyncio
import datetime
import logging
import os

# import random
from ib_insync import *

from UTILITIES.logger_config import logger

project_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(project_dir, "errorlog")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.getLogger("ib_insync").setLevel(logging.WARNING)
# Explicitly add the handler for ib_insync
# logging.getLogger('ib_insync').addHandler(logger.handlers[0])

ib = IB()


class IBOrderManager:
    def __init__(self):
        self.ib = ib
        self.pending_orders = {}
        self.order_events = {}
        self.is_subscribed_onorderstatuschange = False  # Add a flag to track the subscription status

    # def on_order_status_change(self, trade):
    #     order_id = trade.order.orderId
    #     order_status = trade.orderStatus
    #     # When creating an event for a new order
    #
    #     # Handle active status
    #     if order_status.status == 'Submitted' and order_id in self.order_events:
    #         self.order_events[order_id]['active'].set()
    #         # Optionally delete the 'active' event here if no longer needed
    #         del self.order_events[order_id]['active']
    #
    #     # Handle final states
    #     elif order_status.status in ['Filled', 'Cancelled', 'Inactive']:
    #         if order_id in self.order_events:
    #             self.order_events[order_id]['done'].set()
    #             # Delete the entire entry after handling the final state
    #             del self.order_events[order_id]

    def on_order_status_change(self, trade):
        order_id = trade.order.orderId
        order_status = trade.orderStatus

        if order_id in self.order_events:
            if order_status.status == "Submitted":
                self.order_events[order_id]["active"].set()
                asyncio.create_task(self.delayed_event_deletion(order_id))


            elif order_status.status in ["Filled", "Cancelled", "Inactive"]:
                self.order_events[order_id]["done"].set()
                asyncio.create_task(self.delayed_event_deletion(order_id))

                # Unsubscribe if no more interested orders are present
                # if not self.order_events:
                #     self.ib.orderStatusEvent -= self.on_order_status_change
            if not self.order_events:
                    if self.is_subscribed:
                        self.ib.orderStatusEvent -= self.on_order_status_change
                        self.is_subscribed_onorderstatuschange = False  # Reset the subscription flag

    async def delayed_event_deletion(self, order_id, delay=60):
        await asyncio.sleep(delay)
        if order_id in self.order_events:
            del self.order_events[order_id]

    def ib_reset_and_close_pos(self):
        if not self.ib.isConnected():
            print("~~~ Connecting ~~~")
            # randomclientID = random.randint(0, 999)#TODO change bac kclientid
            try:
                self.ib.connect("192.168.1.119", 7497, clientId=5, timeout=45)
                print("connected.")

            except (Exception, asyncio.exceptions.TimeoutError) as e:
                logging.getLogger().error(
                    "Connection error or error reset posistions.: %s", e
                )
                print("~~Connection/closepositions error:", e)
        self.reset_all()
        logger.info("Reset all positions/closed open orders.")

    def reset_all(self):
        self.ib.reqGlobalCancel()

        positions = self.ib.positions()
        # print(positions)
        for position in positions:
            contract = position.contract
            # contract = ib.qualifyContracts(contract)[0]
            # print(contract)
            size = position.position

            # Determine the action ('BUY' to close a short position, 'SELL' to close a long position)
            action = "BUY" if size < 0 else "SELL"

            # Create a market order to close the position
            close_order = MarketOrder(action, abs(size))
            contract.exchange = "SMART"  # Specify the exchange
            # Send the order
            # print(contract)
            self.ib.placeOrder(contract, close_order)
        logger.info("Reset all positions/closed open orders.")

    async def ib_connect(self):
        if not self.ib.isConnected():
            print("~~~ Connecting ~~~")
            try:
                await self.ib.connectAsync(
                    "192.168.1.109", 7497, clientId=0, timeout=45
                )
            except Exception as e:
                logging.getLogger().error("Connection error: %s", e)
                # print("~~Connection error:", e)
        else:
            print("~~~IB already connected. Cannot connect.~~~")

    def ib_disconnect(self):
        try:
            self.ib.disconnect()
        except (Exception, asyncio.exceptions.TimeoutError) as e:
            logging.error("Connection error: %s", e)

    async def get_execution_price(self, parent_order_id):
        for trade in self.ib.trades():
            if (
                trade.order.orderId == parent_order_id
                and trade.orderStatus.status == "Filled"
            ):
                return trade.orderStatus.avgFillPrice
        return None

    # def increment_order_ref(self,order_ref):
    #     """ Increments the numeric part of the order reference. """
    #     match = re.search(r'([\s\S]*?)(\d*Rep\.)(\d+)$', order_ref)
    #     if match:
    #         prefix = match.group(1)
    #         rep_part = match.group(2)
    #         number = int(match.group(3)) + 1
    #         return f"{prefix}{rep_part}{number}"
    #     else:
    #         # If "Rep." doesn't exist, add it and start with 1
    #         return f"{order_ref}_Rep.1"

    async def find_child_orders_with_details(self, contract, action):
        child_orders_list = []
        # Filter open orders to find child orders for the given contract
        for trade in self.ib.openTrades():
            # trade = await getTrade(order)
            # print("action",action)
            # print(order.parentId , order.action,trade.contract)
            # print(trade.contract,contract)
            if trade.order.action == "SELL" and trade.contract == contract:
                # print("childorders trades",trade)
                child_orders_list.append(trade.order)

        # print('childorders trade obj:', child_orders)
        return child_orders_list

    async def getTrade(self, order):
        trade = next(
            (trade for trade in self.ib.trades() if trade.order is order), None
        )

        return trade

    # TODO swapped these cancels_ordres because sometimes it seems order is getting filled after trying to cancel, adn i have mismatching positions/open trades.
    async def cancel_oca_group_orders(self, oca_group_orders):
        cancellation_events = []
        event_listeners = []

        for order in oca_group_orders:
            order_cancelled = asyncio.Event()
            cancellation_events.append((order, order_cancelled))

            def make_on_cancel_order_event(order_id, event):
                def on_cancel_order_event(trade: Trade):
                    if trade.order.orderId == order_id:
                        event.set()

                return on_cancel_order_event

            on_cancel_order_event = make_on_cancel_order_event(
                order.orderId, order_cancelled
            )
            self.ib.cancelOrderEvent += on_cancel_order_event
            event_listeners.append(on_cancel_order_event)
            self.ib.cancelOrder(order)

        # Wait for all cancellations to complete
        await asyncio.gather(*(event.wait() for _, event in cancellation_events))

        # Detach event listeners
        for listener in event_listeners:
            self.ib.cancelOrderEvent -= listener
            print("listeneres: ",listener)
        # Rest of your method...

        # New approach to calculate remaining quantities for each OCA group
        oca_group_remaining_qty = {}
        fills = ib.fills()
        for order in oca_group_orders:
            order_fills = [
                fill for fill in fills if fill.execution.orderId == order.orderId
            ]
            filled_qty = sum(fill.execution.shares for fill in order_fills)
            remaining_qty = order.totalQuantity - filled_qty
            # print("filled qty:", filled_qty, "remainig qty:", remaining_qty)

            oca_group = order.ocaGroup
            oca_group_remaining_qty[oca_group] = remaining_qty

        return oca_group_remaining_qty

    async def cancel_and_replace_orders(
        self,
        contract,
        action,
        CorP,
        ticker,
        exp,
        strike,
        contract_current_price,
        quantity,
        orderRef,
        take_profit_percent,
        trailstop_amount_percent,
    ):
        child_orders_objs_list = await self.find_child_orders_with_details(
            contract, action
        )
        if not child_orders_objs_list:
            print("No children for ", ticker)
            await self.placeOptionBracketOrder(
                CorP,
                ticker,
                exp,
                strike,
                contract_current_price,
                quantity,
                orderRef,
                take_profit_percent,
                trailstop_amount_percent,
                check_opposite_orders=False,
            )

        else:
            oca_group_remaining_qty = await self.cancel_oca_group_orders(
                child_orders_objs_list
            )
            order_details = {}
            for order in child_orders_objs_list:
                oca_group = order.ocaGroup
                remaining_qty = oca_group_remaining_qty.get(oca_group, 0)
                # # new_order_ref = increment_order_ref(order.orderRef)

                if order.ocaGroup not in order_details:
                    order_details[order.ocaGroup] = {}
                if order.orderType == "TRAIL":
                    # print(order,"aux:",order.auxPrice,"trailstopprice:",order.trailStopPrice)

                    order_details[order.ocaGroup]["stopLoss"] = {
                        "type": "TRAIL",
                        "trailingPercent": order.trailingPercent,
                        "auxPrice": order.auxPrice,
                        "trailStopPrice": order.trailStopPrice,
                        "parentID": order.parentId,
                        "percentOffset": order.percentOffset,
                        "orderRef": order.orderRef,
                        "remainingQty": remaining_qty,
                    }
                elif order.orderType == "LMT":
                    order_details[order.ocaGroup]["takeProfit"] = {
                        "type": "LMT",
                        "limitPrice": order.lmtPrice,
                        "parentID": order.parentId,
                        "orderRef": order.orderRef,
                        "remainingQty": remaining_qty,
                    }
            # print("ORDER DEATAILS DICT", order_details)
            await self.placeOptionBracketOrder(
                CorP,
                ticker,
                exp,
                strike,
                contract_current_price,
                quantity=quantity,
                orderRef=orderRef,
                take_profit_percent=take_profit_percent,
                trailstop_amount_percent=trailstop_amount_percent,
                check_opposite_orders=False,
            )

            # trade.orderStatus.ActiveStates
            await self.replace_child_orders(order_details, contract)

    async def replace_child_orders(self, order_details, contract):
        try:
            for ocaGroup, child_order_details in order_details.items():
                quantity = order_details[ocaGroup]["takeProfit"]["remainingQty"]
                ticker_contract = contract
                qualified_contract = await self.ib.qualifyContractsAsync(
                    ticker_contract
                )
                takeProfit = Order()
                takeProfit.orderId = self.ib.client.getReqId()
                takeProfit.action = "SELL"
                takeProfit.orderType = "LMT"
                takeProfit.totalQuantity = quantity
                takeProfit.outsideRth = True
                takeProfit.transmit = True
                takeProfit.tif = "GTC"
                # Use stored take profit details if available
                if (
                    order_details
                    and order_details[ocaGroup]["takeProfit"]["type"] == "LMT"
                ):
                    takeProfit.lmtPrice = order_details[ocaGroup]["takeProfit"][
                        "limitPrice"
                    ]
                    takeProfit.ocaGroup = takeProfit.orderId
                    takeProfit.orderRef = order_details[ocaGroup]["takeProfit"][
                        "orderRef"
                    ]

                stopLoss = Order()
                stopLoss.action = "SELL"
                stopLoss.orderType = "TRAIL"
                stopLoss.TrailingUnit = 1
                stopLoss.totalQuantity = quantity
                stopLoss.orderId = ib.client.getReqId()
                stopLoss.outsideRth = True
                stopLoss.transmit = True
                stopLoss.tif = "GTC"
                # Use stored trailing stop details if available
                if (
                    order_details
                    and order_details[ocaGroup]["stopLoss"]["type"] == "TRAIL"
                ):
                    stopLoss.trailStopPrice = order_details[ocaGroup]["stopLoss"][
                        "trailStopPrice"
                    ]
                    stopLoss.trailingPercent = order_details[ocaGroup]["stopLoss"][
                        "trailingPercent"
                    ]
                    stopLoss.ocaGroup = takeProfit.orderId
                    stopLoss.orderRef = order_details[ocaGroup]["stopLoss"]["orderRef"]
                bracketOrder = [takeProfit, stopLoss]
                order_ids = []

                for o in bracketOrder:
                    o.ocaType = 2
                    trade = ib.placeOrder(qualified_contract[0], o)

                    self.order_events[trade.order.orderId] = {
                        "active": asyncio.Event(),
                        "done": asyncio.Event(),
                    }
                    order_ids.append(trade.order.orderId)
        # TODO i guess this makes it stall out?
            await asyncio.gather(*(self.order_events[orderId]['active'].wait() for orderId in order_ids))

        except (Exception, asyncio.exceptions.TimeoutError) as e:
            logger.exception(
                f"An error occurred while replace child orders.{ticker_contract},: {e}"
            )

    async def placeOptionBracketOrder(
        self,
        CorP,
        ticker,
        exp,
        strike,
        contract_current_price,
        quantity=1,
        orderRef=None,
        take_profit_percent=3,
        trailstop_amount_percent=3,
        check_opposite_orders=True,
        parent_tif="GTD",
    ):
        if take_profit_percent == None:
            take_profit_percent = 3
        if trailstop_amount_percent == None:
            trailstop_amount_percent = 3
        if not self.is_subscribed_onorderstatuschange:
            self.ib.orderStatusEvent += self.on_order_status_change
            self.is_subscribed_onorderstatuschange = True  # Set the flag to indicate subscription

        ticker_contract = Option(ticker, exp, strike, CorP, "SMART")
        qualified_contract = await self.ib.qualifyContractsAsync(ticker_contract)
        print("placeoptionbracketorder")
        if await self.can_place_new_order(qualified_contract[0]):
            if check_opposite_orders:
                try:
                    print("Checking opposite orders.")
                    action = "SELL"  # Adjust based on your logic
                    await self.cancel_and_replace_orders(
                        qualified_contract[0],
                        action,
                        CorP,
                        ticker,
                        exp,
                        strike,
                        contract_current_price,
                        quantity,
                        orderRef,
                        take_profit_percent,
                        trailstop_amount_percent,
                    )
                    # ib.sleep(5)
                except (Exception, asyncio.exceptions.TimeoutError) as e:
                    logger.exception(
                        f"An error occurred while optionbracketorder.{ticker},: {e}"
                    )
            else:
                try:
                    gtddelta = (
                        datetime.datetime.now() + datetime.timedelta(seconds=10)
                    ).strftime("%Y%m%d %H:%M:%S")

                    contract_current_price = round(contract_current_price, 2)
                    quantity = quantity  # Replace with the desired order quantity
                    limit_price = (
                        contract_current_price  # Replace with your desired limit price
                    )
                    take_profit_price = round(
                        contract_current_price
                        + (contract_current_price * (take_profit_percent / 100)),
                        2,
                    )  # Replace with your desired take profit price
                    auxPrice = round(
                        contract_current_price * trailstop_amount_percent, 2
                    )  # Replace with your desired trailing stop percentage
                    triggerPrice = limit_price

                    # This will be our main or "parent" order
                    parent = Order()
                    parent.orderId = self.ib.client.getReqId()

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
                    takeProfit.orderId = self.ib.client.getReqId()
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
                    stopLoss.orderId = self.ib.client.getReqId()
                    stopLoss.action = "SELL"
                    stopLoss.orderType = "TRAIL"
                    stopLoss.TrailingUnit = 1
                    stopLoss.orderRef = orderRef + "_trail"

                    # stopLoss.trailStopPrice = limit_price - trailAmount
                    # stopLoss.trailStopPrice = .5
                    # I  think the big issue HERE was that percent was a whole number like 3, so it was doin some weird thinkg.
                    stopLoss.trailStopPrice = round(
                        contract_current_price
                        * ((100 - trailstop_amount_percent) / 100),
                        2,
                    )
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

                    # self.ib.orderStatusEvent += self.on_order_status_change

                    # Store the parent trade
                    parent_trade = parent.orderId  # IDK DELETE THIS
                    # order_ids = []

                    for o in bracketOrder:
                        o.ocaType = 2
                        trade = ib.placeOrder(qualified_contract[0], o)
                        self.order_events[trade.order.orderId] = {
                            "active": asyncio.Event(),
                            "done": asyncio.Event(),
                        }
                        # order_ids.append(trade.order.orderId)

                    await self.order_events[parent.orderId]["done"].wait()

                    # await asyncio.gather(*(self.order_events[orderId]['done'].wait() for orderId in order_ids))

                    # Return the parent trade
                    return parent_trade

                except (Exception, asyncio.exceptions.TimeoutError) as e:
                    logger.exception(
                        f"An error occurred while optionbracketorder.{ticker},: {e}"
                    )
        else:
            logger.warning(
                f"Too many open orders (15 on either side of contract is max){ticker_contract} {orderRef} .  Skipping order placement."
            )

    # ib.disconnect()
    async def can_place_new_order(self, contract, threshold=13):#15 open orders on either side is MAX.
        # Fetch open orders
        open_trades = self.ib.openTrades()
        # Count orders for the specified contract
        count = sum(
            1 for trade in open_trades if trade.contract.conId == contract.conId
        )
        print(f"{count} open orders for {contract.localSymbol}.")
        # Return True if below threshold, False otherwise
        return count < threshold


"""
Please note the changes to the old code. It is important to call the fist variable 'bracket' to be able to define the 'bracket.parent' later (look into the definiton).
f you have an open position with this order you should use the ib.positions()and retrieve the relevant position to be closed
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
    ib = IB()
    order_manager = IBOrderManager(ib)
    asyncio.run(order_manager.ib_connect())
    order_manager.ib_reset_and_close_pos()
    # ... [Any other operations you want to perform] ...
    order_manager.ib_disconnect()
