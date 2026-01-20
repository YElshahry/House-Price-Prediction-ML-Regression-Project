import pandas as pd
import os

def load_csv(path='data/housing.csv'):
    """
    Loads the housing data from the specified path.
    """
    if not os.path.exists(path):
        parent_path = os.path.join("..", path)
        if os.path.exists(parent_path):
            return pd.read_csv(parent_path)
        else:
            raise FileNotFoundError(f"Could not find file at {path} or {parent_path}.")

    return pd.read_csv(path)