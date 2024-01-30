import asyncio

from IB import *
# from IB import ibAPI
import IB.ibAPI as ib
from UTILITIES.logger_config import logger


async def getTrade(order):
    trade = next((trade for trade in ib.trades() if trade.order is order), None)

    return trade



# Explicitly add the handler for ib_insync
# logging.getLogger('ib_insync').addHandler(logger.handlers[0])
# ib =ib_insync.util.getLoop()
# ib_insync.util.
# Connect to the IB API with a unique client ID



def reset_all():
    ib.ib.connect(
        "192.168.1.109", 7497, clientId=1, timeout=45
    )
    # print("Connected.")
    #
    # await IB.ibAPI.ib.reqGlobalCancel()
    #
    # # Assuming ib.positions() is synchronous or you have handled its asynchronous nature elsewhere
    # positions = IB.ibAPI.positions()
    # # for open_trade in positions:
    # contract = IB.ibAPI.Stock('TSLA','SMART', 'USD')
    # order = IB.ibAPI.MarketOrder('BUY', 1)
    # order.outsideRth=True
    # order.tif = 'GTC'
    # print(order)
    #
    # trade = IB.ibAPI.placeOrder(contract, order)
    # print(trade)
        # print(open_trade)
        # direction = 'BUY' if open_trade.position < 0 else 'SELL'
        # quantity = 1
        # print(quantity)
        # order = MarketOrder(direction, quantity)
        # order.tif = 'GTC'
        # order.outsideRth=True
        # trade = ib.placeOrder(open_trade.contract, order)
        # print('trade placed')
        # print(trade)




if __name__ == "__main__":
    print("ib name is main")
    reset_all()
