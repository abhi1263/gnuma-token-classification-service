import os
import redis
from rq import Queue, Worker, Connection

redis_connection = redis.Redis(
    host=os.getenv("REDIS_HOST", "127.0.0.1"),
    port=os.getenv("REDIS_PORT", "6379"),
    password=os.getenv("REDIS_PASSWORD", ""),
)

listen = ['default']

