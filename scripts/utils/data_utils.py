"""Utility functions for DataFrame operations and logging.

Provides reusable functions for column detection and action logging.

@author: Feurking
"""

import pandas as pd
import numpy as np
from datetime import datetime
from functools import wraps


def log_action(action: str) -> callable:
    """Decorator to log actions performed on the data.

        @param action: The action to log
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = datetime.now()
            print(f"[INFO {start_time}] - {action}")
            result = func(self, *args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            if hasattr(self, 'log'):
                self.log.append({"action": action, "duration": duration, "timestamp": start_time})
            return result
        return wrapper
    return decorator

def load_data(file_path: str, limit: int = None) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(file_path, sep='\t', on_bad_lines='skip', nrows=limit, low_memory=False)

def get_numeric_columns(df: pd.DataFrame) -> list:
    """Get numeric columns in the DataFrame."""
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_columns(df: pd.DataFrame) -> list:
    """Get categorical columns in the DataFrame."""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def get_datetime_columns(df: pd.DataFrame) -> list:
    """Get datetime columns in the DataFrame."""
    return df.select_dtypes(include=['datetime']).columns.tolist()

def get_ordinal_columns(df: pd.DataFrame) -> list:
    """Get ordinal columns (categorical with limited unique values)."""
    return [col for col in get_categorical_columns(df) if df[col].nunique() <= 10]

def get_nominal_columns(df: pd.DataFrame) -> list:
    """Get nominal columns (categorical with many unique values)."""
    return [col for col in get_categorical_columns(df) if df[col].nunique() > 10]
