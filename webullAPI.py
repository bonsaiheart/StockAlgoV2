
from webull import webull, paper_webull
import json
import pandas as pd
import PrivateData.webull_info

email = PrivateData.webull_info.email
deviceID = PrivateData.webull_info.did
trade_token = PrivateData.webull_info.trade_token
password = PrivateData.webull_info.password

webull = webull()
webull._did = deviceID


webull.get_mfa(email)
webull.get_security(email)
webull.login(email, password , 'AnythingYouWant', 'mfa code', 'Security Question ID', 'Security Question Answer')
webull.get_account_id()
webull.get_trade_token(trade_token)
# print(webull.refresh_login())
webull.refresh_login()
webull.get_options(stock="SPY",)
df = pd.DataFrame(webull.get_options("SPY"))
df.to_csv("webulldf.csv")
# option_quote = webull.get_option_quote(stock='AAPL', optionId='1017921255')
print(webull.place_order("CMPS",None, "1.00","BUY", "LMT","GTC","1"))

# option_chain = webull.get_options(stock='AAPL')
# print(type(option_chain[0]))
# print(option_chain[0])

# print(option_quote)
print(webull.get_account())