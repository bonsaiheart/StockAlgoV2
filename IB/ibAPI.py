import asyncio
import datetime
import logging
from ib_insync import *
# util.startLoop()  # uncomment this line when in a notebook

ib = IB()
logging.basicConfig(filename='error_ib.log', level=logging.ERROR)

if ib.isConnected()==False:
    print('Connecting')
    try:
        ib.connect('127.0.0.1', 7497, clientId=1)
    except (ConnectionRefusedError,asyncio.exceptions.TimeoutError) as e:
        logging.error('Connection error: %s', e)
        print('Connection error: %s', e)
        pass
# limitBuyOrder = LimitOrder('BUY', quantity, limit_price)
# limitSellOrder = LimitOrder('SELL', quantity, limit_price)
# trailStopOrder = ('SELL', stop_loss_price, trailing_percent)
gtddelta = (datetime.datetime.now() + datetime.timedelta(seconds=45)).strftime("%Y%m%d %H:%M:%S")

##removed quantity
def placeBuyBracketOrder(ticker, current_price,action):
    ticker_symbol = ticker
    ticker_contract = Stock(ticker_symbol, 'SMART', 'USD')
    ib.qualifyContracts(ticker_contract)

    current_price = current_price
    quantity = 1  # Replace with the desired order quantity
    limit_price = current_price  # Replace with your desired limit price
    take_profit_price = round(current_price * 1.005,2) # Replace with your desired take profit price

    print(take_profit_price)
    stop_loss_price = current_price * .9  # Replace with your desired stop-loss price
    trailAmount = round(current_price * .995,2)  # Replace with your desired trailing stop percentage
    triggerPrice = limit_price

    #This will be our main or "parent" order
    parent = Order()
    parent.orderId = ib.client.getReqId()
    parent.action = action
    parent.orderType = "LMT"
    parent.totalQuantity = quantity
    parent.lmtPrice = limit_price
    parent.transmit = False

    ###this stuff makes it cancel whole order in 45 sec.  If parent fills, children turn to GTC
    parent.tif = 'GTD'
    parent.goodTillDate = gtddelta

    takeProfit = Order()
    takeProfit.orderId = ib.client.getReqId()
    takeProfit.action = "SELL" if action == "BUY" else "BUY"
    takeProfit.orderType = "LMT"
    takeProfit.totalQuantity = quantity
    takeProfit.lmtPrice = take_profit_price
    takeProfit.parentId = parent.orderId

    takeProfit.transmit = False





    stopLoss = Order()
    stopLoss.orderId = ib.client.getReqId()
    stopLoss.action = "SELL" if action == "BUY" else "BUY"
    stopLoss.orderType = "TRAIL"
    stopLoss.TrailingUnit=1
    stopLoss.auxPrice = trailAmount
    stopLoss.trailStopPrice = limit_price - trailAmount
    stopLoss.totalQuantity = quantity
    stopLoss.parentId = parent.orderId
    stopLoss.transmit = True

    bracketOrder = [parent, takeProfit,stopLoss]
    # return bracketOrder
    for o in bracketOrder:
        print(ib.placeOrder(ticker_contract,o))

def placeSellBracketOrder(ticker, current_price):
    ticker_symbol = ticker
    ticker_contract = Stock(ticker_symbol, 'SMART', 'USD')
    ib.qualifyContracts(ticker_contract)

    current_price = current_price
    quantity = 1  # Replace with the desired order quantity
    limit_price = current_price  # Replace with your desired limit price
    take_profit_price = round(current_price * .995,2) # Replace with your desired take profit price

    print(take_profit_price)
    stop_loss_price = current_price * .9  # Replace with your desired stop-loss price
    trailAmount = round(current_price * 1.005,2)  # Replace with your desired trailing stop percentage
    triggerPrice = limit_price

    #This will be our main or "parent" order
    parent = Order()
    parent.orderId = ib.client.getReqId()
    parent.action = "SELL"
    parent.orderType = "LMT"
    parent.totalQuantity = quantity
    parent.lmtPrice = limit_price
    parent.transmit = False

    ###this stuff makes it cancel whole order in 45 sec.  If parent fills, children turn to GTC
    parent.tif = 'GTD'
    parent.goodTillDate = gtddelta

    takeProfit = Order()
    takeProfit.orderId = ib.client.getReqId()
    takeProfit.action =  "BUY"
    takeProfit.orderType = "LMT"
    takeProfit.totalQuantity = quantity
    takeProfit.lmtPrice = take_profit_price
    takeProfit.parentId = parent.orderId

    takeProfit.transmit = False





    stopLoss = Order()
    stopLoss.orderId = ib.client.getReqId()
    stopLoss.action = "BUY"
    stopLoss.orderType = "TRAIL"
    stopLoss.TrailingUnit=1
    stopLoss.auxPrice = trailAmount
    stopLoss.trailStopPrice = limit_price - trailAmount
    stopLoss.totalQuantity = quantity
    stopLoss.parentId = parent.orderId
    stopLoss.transmit = True

    bracketOrder = [parent, takeProfit,stopLoss]
    # return bracketOrder
    for o in bracketOrder:
        print(ib.placeOrder(ticker_contract,o))

def placeCallBracketOrder(ticker,exp,strike, current_price):
    ticker_symbol = ticker
    ticker_contract = Option("spy","2023",430,"C",'SMART', 'USD')
    current_price = current_price
    quantity = 1  # Replace with the desired order quantity
    limit_price = current_price  # Replace with your desired limit price
    take_profit_price = round(current_price * 1.1,2) # Replace with your desired take profit price

    print(take_profit_price)
    stop_loss_price = current_price * .9  # Replace with your desired stop-loss price
    trailAmount = round(current_price * .9,2)  # Replace with your desired trailing stop percentage
    triggerPrice = limit_price

    #This will be our main or "parent" order
    parent = Order()
    parent.orderId = ib.client.getReqId()
    parent.action = "BUY"
    parent.orderType = "LMT"
    parent.totalQuantity = quantity
    parent.lmtPrice = limit_price
    parent.transmit = False

    ###this stuff makes it cancel whole order in 45 sec.  If parent fills, children turn to GTC
    parent.tif = 'GTD'
    parent.goodTillDate = gtddelta

    takeProfit = Order()
    takeProfit.orderId = ib.client.getReqId()
    takeProfit.action =  "SELL"
    takeProfit.orderType = "LMT"
    takeProfit.totalQuantity = quantity
    takeProfit.lmtPrice = take_profit_price
    takeProfit.parentId = parent.orderId

    takeProfit.transmit = False





    stopLoss = Order()
    stopLoss.orderId = ib.client.getReqId()
    stopLoss.action = "SELL"
    stopLoss.orderType = "TRAIL"
    stopLoss.TrailingUnit=1
    stopLoss.auxPrice = trailAmount
    stopLoss.trailStopPrice = limit_price - trailAmount
    stopLoss.totalQuantity = quantity
    stopLoss.parentId = parent.orderId
    stopLoss.transmit = True

    bracketOrder = [parent, takeProfit,stopLoss]
    # return bracketOrder
    for o in bracketOrder:
        print(ib.placeOrder(ticker_contract,o))
# outsideRth=True
def placePutBracketOrder(ticker,exp,strike, current_price):
    ticker_symbol = ticker
    ticker_contract = Option(ticker_symbol,exp,strike,"P",'SMART', 'USD')

    current_price = current_price
    quantity = 1  # Replace with the desired order quantity
    limit_price = current_price  # Replace with your desired limit price
    take_profit_price = round(current_price * 1.1,2) # Replace with your desired take profit price

    print(take_profit_price)
    stop_loss_price = current_price * .9  # Replace with your desired stop-loss price
    trailAmount = round(current_price * .9,2)  # Replace with your desired trailing stop percentage
    triggerPrice = limit_price

    #This will be our main or "parent" order
    parent = Order()
    parent.orderId = ib.client.getReqId()
    parent.action = "BUY"
    parent.orderType = "LMT"
    parent.totalQuantity = quantity
    parent.lmtPrice = limit_price
    parent.transmit = False

    ###this stuff makes it cancel whole order in 45 sec.  If parent fills, children turn to GTC
    parent.tif = 'GTD'
    parent.goodTillDate = gtddelta

    takeProfit = Order()
    takeProfit.orderId = ib.client.getReqId()
    takeProfit.action =  "SELL"
    takeProfit.orderType = "LMT"
    takeProfit.totalQuantity = quantity
    takeProfit.lmtPrice = take_profit_price
    takeProfit.parentId = parent.orderId

    takeProfit.transmit = False





    stopLoss = Order()
    stopLoss.orderId = ib.client.getReqId()
    stopLoss.action = "SELL"
    stopLoss.orderType = "TRAIL"
    stopLoss.TrailingUnit=1
    stopLoss.auxPrice = trailAmount
    stopLoss.trailStopPrice = limit_price - trailAmount
    stopLoss.totalQuantity = quantity
    stopLoss.parentId = parent.orderId
    stopLoss.transmit = True

    bracketOrder = [parent, takeProfit,stopLoss]
    # return bracketOrder
    for o in bracketOrder:
        print(ib.placeOrder(ticker_contract,o))

###TODO diff client id for diff stat. and add options.
"""

I have had success by setting the "clientId" to a different number for each strategy. When you call IB.connect you can specify clientId as one of the parameters:

e.g.

strategy_A = IB()
strategy_A.connect(self.ib.connect(host='127.0.0.1', port=7497, clientId=101)
 
strategy_B = IB()
strategy_B.connect(self.ib.connect(host='127.0.0.1', port=7497, clientId=102)
Whenever a connection does any trading, that clientId will be associated with that trade. Specifically, it can be found in the "execution" field. I think there are numerous ways to retrieve your trades. The one I use is IB.fills(). The fill object looks something like this:

Fill(contract=Future(conId=123456789, symbol='MNQ', lastTradeDateOrContractMonth='20210618', right='?', multiplier='2', exchange='GLOBEX',
currency='USD', localSymbol='MNQM1', tradingClass='MNQ'), execution=Execution(execId='0000a01b.01c1d01.01.01', time=datetime.datetime(2021,
 5, 21, 1, 45, 45, tzinfo=datetime.timezone.utc), acctNumber='U123456', exchange='GLOBEX', side='BOT', shares=1.0, price=13490.5, permId=11111111111, clientId=101, orderId=166, liquidation=0, cumQty=4.0, avgPrice=13490.5, orderRef='', evRule='', evMultiplier=0.0, modelCode='', la
stLiquidity=1), commissionReport=CommissionReport(execId='0000a01b.01c1d01.01.01', commission=0.52, currency='USD', realizedPNL=28.96, yiel
d_=0.0, yieldRedemptionDate=0), time=datetime.datetime(2021, 5, 21, 1, 45, 45, tzinfo=datetime.timezone.utc)),
You can see the associated clientId embedded within the execution object and analyze your trades accordingly to each clientId. It doesn't matter whether you log into TWS/GW as Live account or Paper account, there is no difference to the approach.

Good luck!
#####...A late answer to this question... I had the same problem and found a solution, with a little help, but I still wanna share, because it took me weeks to fix this.

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

Share
Improve this answer
*********

If you want to cancel the outstanding order you can use  ib.reqGlobalCancel() (note: this will cancel all the open orders)

If you have an open position with this order you should use the ib.positions()and retrieve the relevant position to be closed
*********


******Just add a trigger price and adjusted parameters to your stop order, like this code for example. Your stop will become a trail stop once the trigger price is touched

stop = Order(action = reverseAction, totalQuantity = 1, orderType = "STP",
                         parentId = parent.orderId, auxPrice = stop_price,
                         tif ="GTC", transmit = True, orderId = self.ib.client.getReqId(),
                         triggerPrice = trigger_price,
                         adjustedOrderType = "TRAIL",  
                         adjustedTrailingAmount = trail_amount,
                         adjustedStopPrice = adjusted_stop_price )  
"""

