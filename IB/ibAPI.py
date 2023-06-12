from ib_insync import *

# Connect to IB TWS or Gateway
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  ###7467 for papaer
# Define order details


def place_long_bracket_order(ticker):
    symbol = ticker  # Replace with the desired instrument symbol
    ticker_contract = Stock(symbol,'SMART','USD')
    print(ib.qualifyContracts(ticker_contract))
    data = ib.reqMktData(ticker_contract)
    qualifiedContract = ib.qualifyContracts(ticker_contract)
    a = ib.reqMktData(*qualifiedContract)
    print(data.marketPrice())
    print(*qualifiedContract)
    while a.last != a.last:

        ib.sleep(0.01)

    ib.cancelMktData(*qualifiedContract)
    current_price = a.last
    print(current_price)

    quantity = 1  # Replace with the desired order quantity
    limit_price = 100.0  # Replace with your desired limit price
    take_profit_price = 110.0  # Replace with your desired take profit price
    stop_loss_price = 90.0  # Replace with your desired stop-loss price
    trailing_percent = 2.0  # Replace with your desired trailing stop percentage


    # Define order details


    # Create contract object
    contract = Stock(symbol, 'SMART', 'USD')

    # Create order objects
    limit_order = LimitOrder('BUY', quantity, limit_price)

    take_profit_order = LimitOrder('SELL', quantity, take_profit_price)
    stop_loss_order = StopOrder('SELL', quantity, stop_loss_price)

    # Place the limit order
    LimitOrder.transmit = False
    limit_trade = ib.placeOrder(contract, limit_order)
    limit_order_id = limit_trade.order.orderId

    # Place the take profit order as a child of the limit order
    take_profit_order.parentId = limit_order_id
    take_profit_order.transmit = True
    ib.placeOrder(contract, take_profit_order)

    # Place the stop loss order as a child of the limit order
    # stop_loss_order.parentId = limit_order_id
    # stop_loss_trade = ib.placeOrder(contract, stop_loss_order)
    # stop_loss_order_id = stop_loss_trade.order.orderId

    # Modify the stop loss order to trailing stop loss
    trailing_stop_price = stop_loss_price * (1 - trailing_percent / 100)
    trailing_stop_order = Order()
    trailing_stop_order.action = 'SELL'
    trailing_stop_order.orderType = 'TRAIL'
    trailing_stop_order.totalQuantity = quantity
    trailing_stop_order.trailingPercent = trailing_percent
    trailing_stop_order.parentId = limit_order_id
    trailing_stop_order.transmit = True

    ib.placeOrder(contract, trailing_stop_order)
    # Update the stop loss order with adjustable trailing stop
    stop_loss_order.trailingStopPrice = trailing_stop_price

    # Disconnect from IB TWS or Gateway
ib.disconnect()
"""
#####...A late answer to this question... I had the same problem and found a solution, with a little help, but I still wanna share, because it took me weeks to fix this.

My old code placing a bracket order looked like this:

order = ib.bracketOrder('BUY', amount, limit, takeprofit, stoploss, outsideRth=True, tif='GTC')
for ord in order:
    ib.placeOrder(contract, ord)
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

Share
Improve this answer
"""