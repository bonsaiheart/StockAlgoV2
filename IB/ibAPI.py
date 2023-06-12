import time

from ib_insync import *
# util.startLoop()  # uncomment this line when in a notebook

ib = IB()
###PAPER THRU TWS
ib.connect('127.0.0.1', 7497, clientId=1)

###REAL ACCOUNT THRU TWS
z# contract = Stock('AAPL', 'SMART', 'USD')  # Replace with the appropriate contract details for your instrument
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
def buy(ticker,cont):

ticker_symbol = "AAPL"

ticker_contract = Stock(ticker_symbol,'SMART','USD')

# print(ib.qualifyContracts(ticker_contract))
# data = ib.reqMktData(ticker_contract)
# print(data.marketPrice())

# contract = Stock('SPY', 'SMART', 'USD')
# contract1 = ib.qualifyContracts(contract)
# a = ib.reqMktData(*contract1)
# print(contract1,*contract1)
# while a.last != a.last:
#
#     ib.sleep(0.01)
# ib.cancelMktData(*contract1)
# print(a.last)

    #place bracket order


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
    takeProfit.orderType = "LMT"
    takeProfit.lmtPrice = limitPrice
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
# bracket = BracketOrder(ib.client.getReqId(), ib.client.getReqId(), "BUY", '1', 181.5, .11)
# for o in bracket:
#     ib.placeOrder(ticker_contract, o)