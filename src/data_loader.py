import os
from dotenv import load_dotenv
from .config import TRAIN_DATA_PATH, TEST_DATA_PATH
import pandas as pd

def load_train_data():
    """Load training data from the specified path."""
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Training data file not found at {TRAIN_DATA_PATH}")
    df = pd.read_csv(TRAIN_DATA_PATH)
    return df

def load_test_data():
    """Load test data from the specified path."""
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data file not found at {TEST_DATA_PATH}")
    df = pd.read_csv(TEST_DATA_PATH)
    return df