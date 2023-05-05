
from webull import webull, paper_webull
import json
import pandas as pd
import PrivateData.webull_info

email = PrivateData.webull_info.email
deviceID = PrivateData.webull_info.did
trade_token = PrivateData.webull_info.trade_token
password = PrivateData.webull_info.password

webull = paper_webull()
webull._did = deviceID


def login():
    print(webull.get_mfa(email))
    print("logging in")
    webull.login(email, password , 'AnythingYouWant', 'mfa code', 'Security Question ID', 'Security Question Answer')
    webull.get_account_id()
    webull.get_trade_token(trade_token)
    print(webull.refresh_login())
    print(webull.get_account())
def buy(tickerid, price, quantity):
    # stoplmt = float(price) * .97
    sellprice = float(price) * 1.0001


    webull.refresh_login()
    print(webull.place_order(tickerid, None, f'{price}', "BUY", "MKT", "DAY", quantity))


    webull.refresh_login()
    print(webull.place_order(tickerid, None, f'{sellprice}', "SELL", "LMT", "DAY", quantity))




# print(webull.get_options(stock="SPY",))
# # df = pd.DataFrame(webull.get_options("SPY"))
# # df.to_csv("webulldf.csv")
# option_quote = webull.get_option_quote(stock='AAPL', optionId='1038472680')
# print(webull.place_order("CMPS",None, "1.00","BUY", "LMT","GTC","1"))
# print(option_quote)
#
# # option_chain = webull.get_options(stock='AAPL')
# # print(type(option_chain[0]))
# # print(option_chain[0])
# # option_chain = webull.get_options(stock='AAPL')
# # result = webull.place_order_option(optionId='913256135', lmtPrice='1', action='BUY', orderType='LMT', enforce='GTC', quant=1)
# # print(result)
# # print(option_quote)
