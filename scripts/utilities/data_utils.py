"""
Utility functions for DataFrame operations and logging.

Provides reusable functions for column detection and action logging.

@author: Feurking
"""

from datetime import datetime
from functools import wraps
from typing import Callable

try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Error: {e}. Please make sure to install the required packages.")

def log_action(action: str) -> Callable:
    """
    Decorator to log actions performed on the data.

    @param action: The action to log
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            print(f"[INFO {start_time}] - {action}")
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            if hasattr(wrapper, 'log'):
                wrapper.log.append({"action": action, "duration": duration, "timestamp": start_time})
            return result
        return wrapper
    return decorator

def load_data(file_path: str, limit: int = None) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.

    @param file_path: Path to the CSV file
    @param limit: Maximum number of rows to load (optional)
    @return: A DataFrame containing the loaded data
    """
    return pd.read_csv(file_path, sep=';', encoding='utf-8', on_bad_lines='skip',
                       nrows=limit, low_memory=False)

def get_numeric_columns(df: pd.DataFrame) -> list:
    """
    Get numeric columns in the DataFrame.

    @param df: The DataFrame to analyze
    @return: A list of numeric column names
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_columns(df: pd.DataFrame) -> list:
    """
    Get categorical columns in the DataFrame.

    @param df: The DataFrame to analyze
    @return: A list of categorical column names
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def get_datetime_columns(df: pd.DataFrame) -> list:
    """
    Get datetime columns in the DataFrame.

    @param df: The DataFrame to analyze
    @return: A list of datetime column names
    """
    return df.select_dtypes(include=['datetime']).columns.tolist()

def get_ordinal_columns(df: pd.DataFrame) -> list:
    """
    Get ordinal columns (categorical with limited unique values).

    @param df: The DataFrame to analyze
    @return: A list of ordinal column names
    """
    return [col for col in get_categorical_columns(df) if df[col].nunique() <= 10]

def get_nominal_columns(df: pd.DataFrame) -> list:
    """
    Get nominal columns (categorical with many unique values).

    @param df: The DataFrame to analyze
    @return: A list of nominal column names
    """
    return [col for col in get_categorical_columns(df) if df[col].nunique() > 10]
