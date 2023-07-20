from celery import Celery
import redis

app = Celery("StockAlgoV2_followup_tweet", broker="redis://redis_container:6379/0", backend="redis://redis_container:6379/0")
# redis_client = redis.Redis(host='0.0.0.0', port=6379)
###NOte this goes in a docker container, communicates with redis container.