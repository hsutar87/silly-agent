import os
from dotenv import load_dotenv

def load_env():
    load_dotenv()

def env(key: str, default=None):
    return os.getenv(key, default)
