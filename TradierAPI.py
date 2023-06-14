import requests
import send_notifications as send_notifications
import PrivateData.tradier_info

paper_acc = PrivateData.tradier_info.paper_acc
paper_auth =PrivateData.tradier_info.paper_auth
real_acc = PrivateData.tradier_info.real_acc
real_auth = PrivateData.tradier_info.real_auth
# response = requests.post('https://sandbox.tradier.com/v1/accounts/VA24599882/orders',
#     data={'class': 'oto', 'duration': 'gtc', 'type[0]': 'limit', 'price[0]': '160.55', 'option_symbol[0]': 'SPY230512C00415000', 'side[0]': 'buy_to_open', 'quantity[0]': '1', 'type[1]': 'market', 'option_symbol[1]': 'SPY230512C00415000', 'side[1]': 'sell_to_close', 'quantity[1]': '1'},
#     headers={'Authorization': 'Bearer ee5myzHJ8pAoJR9vdHunAJsdQMFJ', 'Accept': 'application/json'}
# )
# json_response = response.json()
#
# print(response.status_code)
# print(json_response)
###6/02.23
###CHANGED TO MARKET ORDER , AND STOP instead SL and duration to day.
def buy(order):
    note, price, quantity, contract, stopcoefficient, sellcoefficient = order
    price = "{:.2f}".format(float(price))
    print(order)
    print("price",price)
    print("stopcoefficient",stopcoefficient)
    response = requests.post(
        f"https://sandbox.tradier.com/v1/accounts/{paper_acc}/orders",
        data={
            "class": "otoco",
            "duration": "day",
            "type[0]": "limit",
            "price[0]": price,
            "option_symbol[0]": contract,
            "side[0]": "buy_to_open",
            "quantity[0]": quantity,
            "type[1]": "stop",
            # "price[1]": str(round(float(price) * float(stopcoefficient), 2)),
            "stop[1]": str(round(float(price) * float(stopcoefficient), 2)),
            "option_symbol[1]": contract,
            "side[1]": "sell_to_close",
            "quantity[1]": quantity,
            "type[2]": "limit",
            "price[2]": str(round(float(price) * float(sellcoefficient), 2)),
            "option_symbol[2]": contract,
            "side[2]": "sell_to_close",
            "quantity[2]": quantity,
        },
        headers={"Authorization": f"Bearer {paper_auth}", "Accept": "application/json"},
    )

    print(response.request.body)


    json_response = response.json()
    print(response.status_code)
    print(json_response)
    send_notifications.email_me_string(order,response.status_code,json_response)

def get_cost_basis():
    print("getting cost basis...")
    response = requests.get(f'https://sandbox.tradier.com/v1/accounts/{paper_acc}/gainloss',
                            params={'page': '1', 'limit': '100', 'sortBy': 'closeDate', 'sort': 'desc',
                                    'start': '2023-05-19', 'end': '2023-05-19', 'symbol': 'SPY'},
                            headers={'Authorization': f'Bearer {paper_auth}', 'Accept': 'application/json'}
                            )

    data = response.json()
    closed_positions = data['gainloss']['closed_position']

    cost_sum = sum(position['cost'] for position in closed_positions)
    gain_loss_sum = sum(position['gain_loss'] for position in closed_positions)
    gain_loss_percent_avg = sum(position['gain_loss_percent'] for position in closed_positions) / len(closed_positions)

    print("Sum of cost:", cost_sum)
    print("Sum of gain_loss:", gain_loss_sum)
    print("Average gain_loss_percent:", gain_loss_percent_avg)

def get_acc_hist():
    print("Getting acc. history...")
    response = requests.get(f'https://api.tradier.com/v1/accounts/{paper_acc}/history',
        params={'page': '3', 'limit': '100', 'type': 'trade, option, ach, wire, dividend, fee, tax, journal, check, transfer, adjustment, interest', 'start': 'yyyy-mm-dd', 'end': 'yyyy-mm-dd', 'symbol': 'SPY', 'exactMatch': 'true'},
        headers={'Authorization': f'Bearer {paper_auth}', 'Accept': 'application/json'}                                                                                                                                                                                                                     )
    json_response = response.json()
    print(response.status_code)
    print(json_response)
# get_acc_hist()
# response = requests.get('https://api.tradier.com/v1/user/history',
#     params={},
#     headers={'Authorization': 'Bearer <TOKEN>', 'Accept': 'application/json'}
# )
# json_response = response.json()
# print(response.status_code)
# print(json_response)