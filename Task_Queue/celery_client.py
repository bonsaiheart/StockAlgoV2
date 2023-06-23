from Task_Queue.task_queue_cellery_bossman import app
import Task_Queue.task_queue

def send_to_celery_1_hour(ticker, current_price, tweet_id, upordown):
    result = Task_Queue.task_queue.wait_60_minutes_and_send_tweet.apply_async(args=[ticker, current_price, tweet_id, upordown],
                                                          countdown=3600)
    print(result)
def send_to_celery_5_hour(ticker, current_price, tweet_id, upordown):
    result = Task_Queue.task_queue.wait_300_minutes_and_send_tweet.apply_async(args=[ticker, current_price, tweet_id, upordown],
                                                          countdown=18000)
    print(result)