
from celery import Celery
import redis
print('hello')
app = Celery('StockAlgoV2_followup_tweet', broker="redis://localhost:6379/0",backend="redis://localhost:6379/0")
# redis_client = redis.Redis(host='0.0.0.0', port=6379)
