import time
import tweepy
from datetime import datetime, timedelta
import requests
from PrivateData import tradier_info
import PrivateData.twitter_info
from Task_Queue.task_queue_cellery_bossman import app as app


def reply_tweet_1_hour_later(message,tweet_id):

    bearer_token = PrivateData.twitter_info.bearer_token
    consumer_key = PrivateData.twitter_info.consumer_key
    consumer_secret = PrivateData.twitter_info.consumer_secret
    access_token = PrivateData.twitter_info.access_token
    access_token_secret = PrivateData.twitter_info.access_token_secret
    api_key = PrivateData.twitter_info.api_key
    api_key_secret = PrivateData.twitter_info.api_key_secret

    client = tweepy.Client(bearer_token=bearer_token, consumer_key=consumer_key, consumer_secret=consumer_secret,
                           access_token=access_token, access_token_secret=access_token_secret)

    response = client.create_tweet(text=message,in_reply_to_tweet_id=tweet_id)

    tweet_id = response.data['id']

    print('Tweet ID:', tweet_id)
    # wait_60_minutes_and_send_tweet("test3")
# #

@app.task
def wait_60_minutes_and_send_tweet(ticker,current_price,tweet_id,upordown):
    old_price = float(current_price)

    # wait_time = timedelta(minutes=60)
    # end_time = datetime.now() + wait_time
    # while datetime.now() < end_time:
    #     pass


    headers = {
        'Authorization': f'Bearer {tradier_info.real_auth}',
        'Accept': 'application/json'
    }

    # Calculate the start and end times
    current_time = datetime.now()
    start_time = current_time - timedelta(hours=1)

    params = {
        'symbol': f'{ticker.upper()}',
        'interval': '1min',
        'start': start_time.strftime('%Y-%m-%d %H:%M'),
        'end': current_time.strftime('%Y-%m-%d %H:%M'),
        'session_filter': 'all'
    }

    response = requests.get('https://api.tradier.com/v1/markets/timesales', params=params, headers=headers)

    time_high_low_dict = {}
    print(response.content)
    json_response = response.json()
    print(json_response)
    highs = []
    lows = []

    for item in json_response['series']['data']:
        high = item['high']
        low = item['low']
        time = item['time']
        time_high_low_dict[time] = {'high': high, 'low': low}

    # Calculate the highest and lowest prices
    highs = [values['high'] for values in time_high_low_dict.values()]
    lows = [values['low'] for values in time_high_low_dict.values()]
    highest_price = max(highs)
    lowest_price = min(lows)
    highest_price_time = next(key for key, value in time_high_low_dict.items() if value['high'] == highest_price)
    lowest_price_time = next(key for key, value in time_high_low_dict.items() if value['low'] == lowest_price)

    if upordown == "up":
        reply_tweet_1_hour_later(
            f"As of {highest_price_time} EST, ${ticker} was ${highest_price}. Up ${round(highest_price - old_price,3)}, or %{round(((highest_price-old_price ) / old_price) * 100,3)} from entry.",
            tweet_id)
    if upordown == "down":
        reply_tweet_1_hour_later(
            f"As of {lowest_price_time} EST, ${ticker} was ${lowest_price}. Down ${round(old_price - lowest_price,3)}, or %{round(((old_price - lowest_price) / lowest_price) * 100,3)} from entry.",
            tweet_id)
###TODO run celery -A your_module_name worker --loglevel=info -D
@app.task
def wait_300_minutes_and_send_tweet(ticker,current_price,tweet_id,upordown):
    old_price = float(current_price)

    # wait_time = timedelta(minutes=60)
    # end_time = datetime.now() + wait_time
    # while datetime.now() < end_time:
    #     pass


    headers = {
        'Authorization': f'Bearer {tradier_info.real_auth}',
        'Accept': 'application/json'
    }

    # Calculate the start and end times
    current_time = datetime.now()
    start_time = current_time - timedelta(hours=5)

    params = {
        'symbol': f'{ticker.upper()}',
        'interval': '1min',
        'start': start_time.strftime('%Y-%m-%d %H:%M'),
        'end': current_time.strftime('%Y-%m-%d %H:%M'),
        'session_filter': 'all'
    }

    response = requests.get('https://api.tradier.com/v1/markets/timesales', params=params, headers=headers)

    time_high_low_dict = {}
    print(response.content)
    json_response = response.json()
    print(json_response)
    highs = []
    lows = []

    for item in json_response['series']['data']:
        high = item['high']
        low = item['low']
        time = item['time']
        time_high_low_dict[time] = {'high': high, 'low': low}

    # Calculate the highest and lowest prices
    highs = [values['high'] for values in time_high_low_dict.values()]
    lows = [values['low'] for values in time_high_low_dict.values()]
    highest_price = max(highs)
    lowest_price = min(lows)
    highest_price_time = next(key for key, value in time_high_low_dict.items() if value['high'] == highest_price)
    lowest_price_time = next(key for key, value in time_high_low_dict.items() if value['low'] == lowest_price)

    if upordown == "up":
        reply_tweet_1_hour_later(
            f"As of {highest_price_time} EST, ${ticker} was ${highest_price}. Up ${round(highest_price - old_price,3)}, or %{round(((highest_price-old_price ) / old_price) * 100,3)} from entry.",
            tweet_id)
    if upordown == "down":
        reply_tweet_1_hour_later(
            f"As of {lowest_price_time} EST, ${ticker} was ${lowest_price}. Down ${round(old_price - lowest_price,3)}, or %{round(((old_price - lowest_price) / lowest_price) * 100,3)} from entry.",
            tweet_id)
#daemon mode backgroud.
# Original code
# wait_60_minutes_and_send_tweet('AAPL', '100', '123456789', 'up')
#daemon mode backgroud.
# Original code
# result = wait_60_minutes_and_send_tweet('AAPL', '100', '123456789', 'up')
