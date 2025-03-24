"""
This module contains classes for scaling data using various techniques,
including standard scaling, min-max scaling, robust scaling, and others.
It also includes helper functions for data summarization and logging actions.

Primary Functions & Classes:
    DataScaling: Class to load and summarize data from a CSV file.
    FeatureScaler: Class to scale and process data using various scaling techniques.
    DataSummaryMixin: Mixin class to add summary functionality to a DataFrame.
    generate_summary: Function to generate a summary of a DataFrame.

@author: Feurking
"""

# [Tache2.0][4pts] Faire le scaling des données

# python -m scripts --file_path "C:/Users/Quentin/Desktop/machine learninig/teaching_ml_bis_2025/data/en.openfoodfacts.org.products.csv" --output_path "C:/Users/Quentin/Desktop/Fuck YourSlef PEP8/teaching_ml_bis_2025/results/test.csv" --target_col "price"

import re
from datetime import datetime

try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
except ImportError as e:
    print(f"Error: {e}. Please make sure to install the required packages.")

from ..utilities.data_utils import load_data, log_action, get_numeric_columns, get_categorical_columns, get_datetime_columns


def generate_summary(df, log) -> dict:
    """
    Generates a summary of the DataFrame, including the shape, missing values,
    duplicate rows, and the last 3 actions logged.

    @param df: The DataFrame to summarize.
    @param log: The list of actions to fetch the last 3 from.
    @return: A dictionary containing the shape, missing values count, duplicate count, and last 3 actions.
    """
    return {
        'shape': df.shape,
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'last_3_actions': [entry['action'] for entry in log[-3:]]
    }


class DataSummaryMixin(object):
    """
    Mixin class to add summary functionality to a DataFrame.
    Provides string representation, summary, and length of the DataFrame.
    """
  
    def __repr__(self) -> str:
        """
        Returns a string representation of the DataScaler instance, showing basic info 
        about the DataFrame like shape and column counts.
        """
        return (f"DataScaler(shape={self.df.shape}, "
                f"num_cols={len(get_numeric_columns(self.df))}, "
                f"cat_cols={len(get_categorical_columns(self.df))}, "
                f"date_cols={len(get_datetime_columns(self.df))}, "
                f"actions={len(self.log)})")
  
    def __str__(self) -> str:
        """
        Returns a summary string of the DataScaler instance, including shape, missing values, 
        duplicates, and the last 3 logged actions.
        """
        summary = generate_summary(self.df, self.log)
        return (f"DataScaler Summary:\n"
                f"Shape: {summary['shape']}\n"
                f"Missing Values: {summary['missing_values']}\n"
                f"Duplicates: {summary['duplicates']}\n"
                f"Last 3 actions: {summary['last_3_actions']}")

    def __len__(self) -> int:
        """Returns the number of rows in the DataFrame."""
        return len(self.df)


class DataScaling(DataSummaryMixin):
    """
    Class for scaling and processing data using various scaling techniques.
    Includes functionality to load data from a CSV file and summarize it.
    """
    def __init__(self, df: pd.DataFrame, file_path: str = None) -> None:
        self.df = df.copy()
        self.log = []
        self._file_path = file_path

    @classmethod
    def from_csv(cls, file_path: str, limit: int) -> 'DataScaling':
        """
        Loads a DataFrame from a CSV file with an optional row limit.

        @param file_path (str): Path to the CSV file.
        @param limit (int): Maximum number of rows to load from the CSV file.
        @return: An instance of the DataScaling class initialized with the loaded data.
        """
        df = load_data(file_path, limit)
        return cls(df, file_path)

    @property
    def file_path(self) -> str:
        """Getter for the file path."""
        return self._file_path

    @file_path.setter
    def file_path(self, file_path: str) -> None:
        """
        Setter for the file path. Validates the file path to ensure it's a string
        and ends with '.csv'.
        """
        if not isinstance(file_path, str): 
            raise TypeError("File path must be a string.")
        if not re.match(r'.+\.csv', file_path): 
            raise ValueError("File path must be a CSV file.")
        self._file_path = file_path

    def summarize(self) -> dict:
        """
        Summarizes the DataFrame, including its shape, missing values, and duplicates.

        - Return -> dict:
            A summary of the DataFrame.
        """
        return generate_summary(self.df, self.log)

class FeatureScaler(DataScaling):
    """
    Class that extends DataScaling to add feature scaling capabilities, including
    standard scaling, min-max scaling, and robust scaling.
    """
    def save_scaled_df(self) -> None:
        """
        Saves the scaled DataFrame to the specified file path in CSV format with UTF-8 encoding.
        """
        self.df.to_csv(self.file_path, index=False, encoding='utf-8')

    def display_info(self) -> None:
        """Prints information about the DataFrame including its data types and summary."""
        print(self.df.info())
        print(self.df.dtypes)

    def _scale(self, scaler) -> pd.DataFrame:
        """
        Scales the numeric columns of the DataFrame using the provided scaler.
        
        @param scaler: The scaling method (e.g., StandardScaler, MinMaxScaler, etc.)
        @return: The scaled DataFrame.
        """
        numeric_cols = get_numeric_columns(self.df)
        if numeric_cols:
            self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols].fillna(0))
            self.log.append({'action': scaler.__class__.__name__, 'timestamp': datetime.now()})
        else:
            print("No numeric columns to scale.")
        return self.df
    
    def scale_based_on_distribution(self, option: int = 0) -> pd.DataFrame:
        """
        Chooses and applies the most suitable scaling method based on the distribution of the data.
        
        - Returns -> pd.DataFrame: 
            The DataFrame after applying the chosen scaling method.
        """
        numeric_cols = get_numeric_columns(self.df)
        
        if not numeric_cols: return self.df

        stats = self.df[numeric_cols].agg(['skew', 'kurtosis']).T

        for col, row in stats.iterrows():
            skewness, kurtosis = row['skew'], row['kurtosis']
            print(f"{col}: Skewness = {skewness}, Kurtosis = {kurtosis}")

            if abs(skewness) > 1:
                self._apply_robust_scaling(col) if option == 0 else self._robust_true_scale(numeric_cols)
            elif abs(kurtosis) < 3:
                self._apply_min_max_scaling(col) if option == 0 else self._min_max_true_scale(numeric_cols)
            else: 
                self._apply_standard_scaling(col) if option == 0 else self._standard_true_scale(numeric_cols)

        return self.df

    def _apply_standard_scaling(self, col: str) -> None:
        """
        Applies StandardScaler to a single column.
        
        @param col: The column name to scale.
        """
        print(f"Applying Standard Scaling to {col}")
        self.df[[col]] = StandardScaler().fit_transform(self.df[[col]].fillna(0))
        self.log.append({'action': f"Standard Scaling applied to {col}", 'timestamp': datetime.now()})

    def _apply_min_max_scaling(self, col: str) -> None:
        """
        Applies MinMaxScaler to a single column.
        
        @param col: The column name to scale.
        """
        print(f"Applying Min-Max Scaling to {col}")
        self.df[[col]] = MinMaxScaler().fit_transform(self.df[[col]].fillna(0))
        self.log.append({'action': f"Min-Max Scaling applied to {col}", 'timestamp': datetime.now()})

    def _apply_robust_scaling(self, col: str) -> None:
        """
        Applies RobustScaler to a single column.
        
        @param col: The column name to scale.
        """
        print(f"Applying Robust Scaling to {col}")
        self.df[[col]] = RobustScaler().fit_transform(self.df[[col]].fillna(0))
        self.log.append({'action': f"Robust Scaling applied to {col}", 'timestamp': datetime.now()})

    def _standard_scale(self, numeric_cols: list) -> pd.DataFrame:
        """
        Scales the numeric columns using StandardScaler and logs the action.
        
        @param numeric_cols: List of numeric column names to scale.
        @return: The scaled DataFrame.
        """
        self.df[numeric_cols] = StandardScaler().fit_transform(self.df[numeric_cols].fillna(0))
        self.log.append({'action': 'StandardScaler applied to numeric columns', 'timestamp': datetime.now()})
        return self.df
    
    def _standard_true_scale(self, numeric_cols: list) -> pd.DataFrame:
        """
        Scales the numeric columns using a custom implementation of StandardScaler and logs the action.
        
        @param numeric_cols: List of numeric column names to scale.
        @return: The scaled DataFrame.
        """
        for col in numeric_cols:
            col_mean = self.df[col].mean()
            col_std = self.df[col].std()
            if col_std > 0:
                self.df[col] = (self.df[col] - col_mean) / col_std
            else:
                print(f"Column {col} has no standard deviation. Skipping scaling.")
        self.log.append({'action': 'Custom StandardScaler applied', 'timestamp': datetime.now()})
        return self.df
    
    def _robust_true_scale(self, numeric_cols: list) -> pd.DataFrame:
        """
        Scales the numeric columns using a custom implementation of RobustScaler and logs the action.
        
        @param numeric_cols: List of numeric column names to scale.
        @return: The scaled DataFrame.
        """
        for col in numeric_cols:
            col_median = self.df[col].median()
            col_iqr = self.df[col].quantile(0.75) - self.df[col].quantile(0.25)
            if (col_iqr > 0).any():
                self.df[col] = (self.df[col] - col_median) / col_iqr
            else:
                print(f"Column {col} has no IQR (interquartile range). Skipping scaling.")
        self.log.append({'action': 'Custom RobustScaler applied', 'timestamp': datetime.now()})
        return self.df

    def _min_max_true_scale(self, numeric_cols: list) -> pd.DataFrame:
        """
        Scales the numeric columns using MinMaxScaler to a true scale (0 to 1)
        based on the global min and max values of each column.

        @param numeric_cols: List of numeric column names to scale.
        @return: The scaled DataFrame.
        """
        if numeric_cols:
            for col in numeric_cols:
                col_min = self.df[col].min()
                col_max = self.df[col].max()
                
                if isinstance(col_min, (int, float)) and isinstance(col_max, (int, float)):
                    if (col_max - col_min) > 0:
                        self.df[col] = (self.df[col] - col_min) / (col_max - col_min)
                    else:
                        print(f"Column {col} has no range (min equals max). Skipping scaling.")
                else:
                    print(f"Column {col} returned non-scalar values for min/max. Skipping scaling.")
                    
            self.log.append({'action': 'Min-Max True Scaling applied', 'timestamp': datetime.now()})
        else: print("No numeric columns to scale.")
        
        return self.df


    def _robust_scale(self) -> pd.DataFrame:
        """
        Scales the numeric columns using RobustScaler and logs the action.
        
        - Returns -> pd.DataFrame:
            The scaled DataFrame.
        """
        return self._scale(RobustScaler())

    @log_action("📏 Normalisation des colonnes numériques")
    def normalize(self) -> pd.DataFrame:
        """
        Normalizes the numeric columns of the DataFrame (L2 normalization) and logs the action.
        
        - Returns -> pd.DataFrame:
            The normalized DataFrame.
        """
        numeric_cols = get_numeric_columns(self.df)
        self.df[numeric_cols] = self.df[numeric_cols].apply(lambda x: x / np.linalg.norm(x.fillna(0)), axis=0)
        return self.df
    
    @log_action("📏 Logarithmisation des colonnes numériques")
    def log_transform(self) -> pd.DataFrame:
        """
        Applies a logarithmic transformation to the numeric columns and logs the action.
        
        - Returns -> pd.DataFrame:
            The transformed DataFrame.
        """
        numeric_cols = get_numeric_columns(self.df)
        self.df[numeric_cols] = self.df[numeric_cols].apply(lambda x: np.log1p(x.fillna(0)), axis=0)
        return self.df