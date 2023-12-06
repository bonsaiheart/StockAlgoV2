import asyncio

import ib_insync.util

from ibAPI import *
from ibAPI import ib  # Import the ib instance from ibAPI.py

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
# ib =ib_insync.util.getLoop()
# ib_insync.util.
# Connect to the IB API with a unique client ID
try:
    ib.connect("192.168.1.119", 7497, clientId=1, timeout=45)
    print("Connected.")
except (Exception, asyncio.exceptions.TimeoutError) as e:
    logging.getLogger().error("Connection error or error reset positions: %s", e)
    print("Connection/close positions error:", e)


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




print(__name__)
if __name__ == "__main__":
    print("ib name is main")
    ib_reset_and_close_pos()
