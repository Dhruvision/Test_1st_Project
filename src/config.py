import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, '..', '.env')
load_dotenv(ENV_PATH)

TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH")
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH")
PREDICTIONS_PATH = os.getenv("PREDICTIONS_PATH")
TEST_SIZE = float(os.getenv("TEST_SIZE"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE"))