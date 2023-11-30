import asyncio

import Task_Queue.task_queue

###Note, its no longer 1 hour, but a countdown passed into function.
def send_to_celery_1_hour(ticker, current_price, tweet_id, upordown, countdownseconds):
    result = Task_Queue.task_queue.wait_60_minutes_and_send_tweet.apply_async(
        args=[ticker, current_price, tweet_id, upordown, countdownseconds], countdown=countdownseconds
    )
    print("result from sendtocelery:",result)
async def followup_tweet_async_cycle(ticker, current_price, tweet_id, upordown, countdownseconds):
    asyncio.create_task(Task_Queue.task_queue.followup_tweet(ticker, current_price, tweet_id, upordown, countdownseconds))