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


async def reset_all():
    # ib.ib.connect(
    #     "192.168.1.109", 7497, clientId=1, timeout=45
    # )
    await ib.ib.connectAsync("localhost", 4002, clientId=0, timeout=45)

    print("Connected.")

    ib.ib.reqGlobalCancel()

    # Assuming ib.positions() is synchronous or you have handled its asynchronous nature elsewhere
    positions = ib.ib.positions()
    for position in positions:
        contract = position.contract
        if position.position > 0:  # Number of active Long positions
            action = "Sell"  # to offset the long positions
        elif position.position < 0:  # Number of active Short positions
            action = "Buy"  # to offset the short positions
        else:
            assert False
        totalQuantity = abs(position.position)
        order = ib.MarketOrder(action=action, totalQuantity=totalQuantity)
        trade = ib.ib.placeOrder(contract, order)
        print(f"Flatten Position: {action} {totalQuantity} {contract.localSymbol}")
        assert trade in ib.ib.trades(), "trade not listed in ib.trades"
    ib.ib.disconnect()


if __name__ == "__main__":
    print("ib name is main")
    asyncio.run(reset_all())
