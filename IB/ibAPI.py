import asyncio
import datetime
import logging
import os
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
        self.is_subscribed_onorderstatuschange = (
            False  # Add a flag to track the subscription status
        )

    # def ib_enable_log(self,level=logging.ERROR):#try using this
    #     """Enables ib insync logging"""
    #     util.logToConsole(level)
    def on_order_status_change(self, trade):
        order_id = trade.order.orderId
        order_status = trade.orderStatus

        if order_id in self.order_events:
            if (
                order_status.status == "Submitted"
                or order_status.status == "PreSubmitted"
            ):
                # print("ORDERSTATUS submitted: ", order_status.status)
                self.order_events[order_id]["active"].set()
                # asyncio.create_task(self.delayed_event_deletion(order_id))
                # del self.order_events[order_id]

            elif order_status.status in [
                "Filled",
                "Cancelled",
                "Inactive",
                "ApiCancelled",
            ]:
                self.order_events[order_id]["done"].set()
                # asyncio.create_task(
                #     self.delayed_event_deletion(order_id)
                # )  # TODO i think these are causeing the destroyed taks  error?
                # del self.order_events[order_id]

            if not self.order_events:
                if self.is_subscribed_onorderstatuschange == True:
                    self.ib.orderStatusEvent -= self.on_order_status_change
                    self.is_subscribed_onorderstatuschange = (
                        False  # Reset the subscription flag
                    )

    # TODO 24/01/12 I curernylt think this is the bit tat is causing a freeze when running all the tasks frm main.

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

    async def find_child_orders_with_details(self, contract, action, open_trades):
        child_orders_list = []
        # Use the provided list of open trades
        for trade in open_trades:
            if trade.order.action == action and trade.contract == contract:
                if trade.orderStatus.status in ["Submitted", "PreSubmitted"]:
                    child_orders_list.append(trade.order)
        return child_orders_list

        return child_orders_list

    async def getTrade(self, order):
        trade = next(
            (trade for trade in self.ib.trades() if trade.order is order), None
        )

        return trade

    async def cancel_oca_group_orders(self, oca_group_orders, ticker):
        cancellation_events = []
        event_listeners = []
        processed_groups = set()  # Set to keep track of processed OCA groups

        for order in oca_group_orders:
            oca_group = order.ocaGroup
            if oca_group in processed_groups:
                continue
            trade = await self.getTrade(order)
            if trade and trade.orderStatus.status in [
                "Cancelled",
                # "ApiCancelled",
                "Filled",
                "Inactive",
            ]:
                if trade.orderStatus.status == "Filled":
                    logger.error("THERES FILLED orders IN OCAGROUP TOCANCeL?")
                logger.info(
                    f"Skipping cancellation for order {order.orderId} as it's already {trade.orderStatus.status}"
                )
                continue
            # Process this group
            processed_groups.add(oca_group)
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

            try:
                self.ib.cancelOrder(order)
            except Exception as e:
                logger.error("error in ib.cancelORder for oca", e)
                print(f"Error cancelling order {order.orderId}: {e}")
                if trade.order.orderId == order.orderId and trade.isDone:
                    # event.set()
                    order_cancelled.set()  # Set the event to avoid waiting indefinitely
                continue

        # Wait for all cancellations to finsh
        try:
            await asyncio.wait(
                [
                    event.wait() for _, event in cancellation_events
                ],  # timeout=60  removed timeout.
            )
            logger.info(f"All children cancelled for {ticker}")
        except Exception as e:
            logger.error(f"error cancelling these childrenfor {ticker}, {e}")
            raise e
        listener_sum = 0
        for listener in event_listeners:
            listener_sum += 1
            self.ib.cancelOrderEvent -= listener
        # Rest of your method...
        # print("listener sum: ", listener_sum)

        # 12/20 New approach to calculate remaining quantities for each OCA group
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
        open_trades,
    ):
        child_orders_objs_list = await self.find_child_orders_with_details(
            contract, action, open_trades
        )
        if not child_orders_objs_list:
            # print("No children for ", ticker)
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
                child_orders_objs_list, ticker
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
            logger.info(f"All child order replaced.for {ticker}")

    async def replace_child_orders(self, order_details, contract):
        try:
            order_ids = []

            for ocaGroup, child_order_details in order_details.items():
                # print(order_details[ocaGroup])
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

                for o in bracketOrder:
                    o.ocaType = 2
                    trade = ib.placeOrder(qualified_contract[0], o)

                    self.order_events[trade.order.orderId] = {
                        "active": asyncio.Event(),
                        "done": asyncio.Event(),
                    }
                    order_ids.append(trade.order.orderId)
            # TODO i guess this gathe part makes it stall out?  Note:  I believe the issue may be that "active" didn't include "presubmitted" i.e. for trailing stop.

            # await asyncio.gather(
            #     *(self.order_events[orderId]["active"].wait() for orderId in order_ids),
            #     return_exceptions=True,
            # )
            await asyncio.gather(
                self.order_events[takeProfit.orderId]["active"].wait(),
                self.order_events[stopLoss.orderId]["active"].wait(),
                return_exceptions=True,
            )

            # Delete the entries after the wait calls
            self.order_events.pop(takeProfit.orderId, None)
            self.order_events.pop(stopLoss.orderId, None)

            # await asyncio.gather(*(self.order_events[orderId]['done'].wait() for orderId in order_ids))

            # print("waitforchildren", waitforchildren)
        except (Exception, asyncio.exceptions.TimeoutError) as e:
            logger.exception(
                f"An error occurred while replace child orders.{ticker_contract},: {e}\n order_details_dict for ocaGroup:{ocaGroup}: {order_details[ocaGroup]}"
            )

    # TODO will still cancel children from a leftorver bracket order that has parent still active.. leaving just parent.
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

        ticker_contract = Option(ticker, exp, strike, CorP, "SMART")
        qualified_contract = await self.ib.qualifyContractsAsync(ticker_contract)
        can_place_order, open_trades = await self.can_place_new_order(
            qualified_contract[0]
        )
        if can_place_order:
            if not self.is_subscribed_onorderstatuschange:
                self.ib.orderStatusEvent += self.on_order_status_change
                self.is_subscribed_onorderstatuschange = (
                    True  # Set the flag to indicate subscription
                )
            if check_opposite_orders:
                try:
                    # print("Checking opposite orders.")
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
                        open_trades,
                    )
                    # ib.sleep(5)
                except (Exception, asyncio.exceptions.TimeoutError) as e:
                    logger.exception(
                        f"An error occurred while optionbracketorder.{ticker},: {e}"
                    )
            else:
                try:
                    gtddelta = (
                        datetime.datetime.now() + datetime.timedelta(seconds=15)
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
                    )
                    auxPrice = round(
                        contract_current_price * trailstop_amount_percent, 2
                    )
                    triggerPrice = limit_price

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
                    parent.tif = parent_tif  # TODO 'GTC'
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

                    stopLoss.trailStopPrice = round(
                        contract_current_price
                        * ((100 - trailstop_amount_percent) / 100),
                        2,
                    )

                    stopLoss.trailingPercent = trailstop_amount_percent
                    stopLoss.totalQuantity = quantity
                    stopLoss.parentId = parent.orderId
                    stopLoss.ocaGroup = parent.orderId
                    stopLoss.outsideRth = True
                    stopLoss.transmit = True
                    stopLoss.tif = "GTC"

                    bracketOrder = [parent, takeProfit, stopLoss]

                    # self.ib.orderStatusEvent += self.on_order_status_change

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

                    await asyncio.gather(
                        self.order_events[parent.orderId]["done"].wait(),
                        self.order_events[takeProfit.orderId]["active"].wait(),
                        self.order_events[stopLoss.orderId]["active"].wait(),
                        return_exceptions=True,
                    )

                    # Delete the entries after the wait calls
                    self.order_events.pop(parent.orderId, None)
                    self.order_events.pop(takeProfit.orderId, None)
                    self.order_events.pop(stopLoss.orderId, None)

                    # await asyncio.gather(*(self.order_events[orderId]['done'].wait() for orderId in order_ids))

                    # Return the parent trade
                    return parent_trade

                except (Exception, asyncio.exceptions.TimeoutError) as e:
                    logger.exception(
                        f"An error occurred while optionbracketorder.{ticker},: {e}"
                    )
        else:
            # logger.warning(
            #     f"Too many open orders (15 on either side of contract is max){ticker_contract} {orderRef} .  Skipping order placement."
            # )
            pass

    # ib.disconnect()
    # print(f"{count} open orders for {contract.localSymbol}.")

    async def can_place_new_order(self, contract, threshold=6):
        # Fetch open orders
        open_trades = self.ib.openTrades()
        # Filter and count orders for the specified contract
        contract_open_trades = [
            trade for trade in open_trades if trade.contract.conId == contract.conId
        ]
        count = len(contract_open_trades)
        # Return a tuple of the boolean check and the filtered open trades
        return count <= threshold, contract_open_trades


# TODO'p[
"""yeah, I have seen this with bracket orders as well. What I have started doing is I store the order objects in cache in memory after placing the order, and then using the trade filled and CancelEvents to manage the positions myself. Right now, I am also seeing issues with ib.positions() for some of my accounts.

trade = ib.placeOrder(self.contract, o)
trade.filledEvent += self.order_status
trade.cancelledEvent += self.order_status"""
