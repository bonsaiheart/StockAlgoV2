from ib_insync import *
# util.startLoop()  # uncomment this line when in a notebook

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# contract = Stock('AAPL', 'SMART', 'USD')  # Replace with the appropriate contract details for your instrument
#
# limitBuyOrder = LimitOrder('BUY', quantity, limitPrice)
# limitSellOrder = LimitOrder('SELL', quantity, limitPrice)
# trailStopOrder = ('SELL', trailingStopPrice, trailPercent)
#
# parentOrder = ib.placeOrder(
#     contract,
#     limitBuyOrder,
#     ocaGroup='bracketOrder',  # Specify a group name to link the OTOCO orders
#     transmit=False  # Set transmit to False to submit the parent order without transmitting it
# )
# ocoOrder = [limitSellOrder, trailStopOrder]
# ocoOrders = ib.placeOrder(
#     contract,
#     ocoOrder,
#     parentId=parentOrder.order.orderId,  # Link the OCO orders to the parent order
#     transmit=True  # Set transmit to True to submit the OCO orders immediately
# )
# ib.sleep(1)  # Wait briefly to allow orders to be processed
# print(parentOrder, ocoOrders)
# ib.disconnect()



def BracketOrder(parentOrderId, childOrderId, action, quantity, limitPrice, trailAmount):

    #This will be our main or "parent" order
    parent = Order()
    parent.orderId = parentOrderId
    parent.action = action
    parent.orderType = "LMT"
    parent.totalQuantity = quantity
    parent.lmtPrice = limitPrice
    parent.transmit = False

    takeProfit = Order()
    takeProfit.orderId = childOrderId
    takeProfit.action = "SELL" if action == "BUY" else "BUY"
    takeProfit.orderType = "TRAIL"
    takeProfit.auxPrice = trailAmount
    takeProfit.trailStopPrice = limitPrice - trailAmount
    takeProfit.totalQuantity = quantity
    takeProfit.parentId = parentOrderId
    takeProfit.transmit = False

    stopLoss = Order()
    stopLoss.orderId = childOrderId
    stopLoss.action = "SELL" if action == "BUY" else "BUY"
    stopLoss.orderType = "TRAIL"
    stopLoss.auxPrice = trailAmount
    stopLoss.trailStopPrice = limitPrice - trailAmount
    stopLoss.totalQuantity = quantity
    stopLoss.parentId = parentOrderId
    stopLoss.transmit = True

    bracketOrder = [parent, stopLoss]
    return bracketOrder

# bracket = BracketOrder(ib.client.getReqId(), ib.client.getReqId(), "BUY", quantity, limitPrice, trailAmount)
bracket = BracketOrder(ib.client.getReqId(), ib.client.getReqId(), "BUY", '1', 5, .11)
for o in bracket:
    ib.placeOrder(contract, o)