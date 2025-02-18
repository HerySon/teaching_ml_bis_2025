"""
This module contains a class for data cleaning operations.

Primary Functions & Classes:
    DataCleaning: Class to clean and preprocess data in a DataFrame.
    
@author: Feurking
"""

import pandas as pd
import numpy as np

import re
import pytest

from datetime import datetime

from .utils.data_utils import load_data, log_action, get_numeric_columns, get_categorical_columns, get_datetime_columns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class DataCleaning:
    @pytest.mark.parametrize("df", [ ("/data/en.openfoodfacts.org.products.csv") ])
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self.log = []

    def __repr__(self) -> str:
        return (f"DataCleaner(shape={self.df.shape}, "
                f"num_cols={len(get_numeric_columns(self.df))}, "
                f"cat_cols={len(get_categorical_columns(self.df))}, "
                f"date_cols={len(get_datetime_columns(self.df))}, "
                f"actions={len(self.log)})")

    def __str__(self) -> str:
        summary = self.summarize()
        return (f"DataCleaner Summary:\n"
                f"Shape: {summary['shape']}\n"
                f"Missing Values: {summary['missing_values']}\n"
                f"Duplicates: {summary['duplicates']}\n"
                f"Last 3 actions: {[entry['action'] for entry in self.log[-3:]]}")

    def __len__(self) -> int:
        return len(self.df)

    @classmethod
    def from_csv(cls, file_path : str, limit : int) -> 'DataCleaning':
        """Create a DataCleaning object from a CSV file.
        
            @param file_path: Path to the CSV file
            @param limit: Maximum number of rows to load
        """
        df = load_data(file_path, limit)
        return cls(df)

    @log_action("🪪 Dropping uninformative columns")
    def drop_uninformative_columns(self) -> None:
        cols_to_drop = [col for col in self.df.columns if self.df[col].nunique() <= 1]
        self.df.drop(columns=cols_to_drop, inplace=True)

    @log_action("🚮 Dropping irrelevant columns")
    def drop_irrelevant_columns(self, irrelevant_cols: list) -> None:
        self.df.drop(columns=[col for col in irrelevant_cols if col in self.df.columns], inplace=True)

    @log_action("👊 Handling missing values")
    def handle_missing_values(self, threshold: float = 0.5, strategies=None) -> None:
        """Handle missing values in the DataFrame using a threshold and strategies."""
        high_missing_cols = self.df.columns[self.df.isnull().mean() > threshold]
        self.df.drop(columns=high_missing_cols, inplace=True)
        strategies = strategies or {}

        for col in self.df.columns[self.df.isnull().any()]:
            if col not in high_missing_cols:
                self.df[col] = self.df[col].fillna(strategies.get(col, self.df[col].mode()[0] if self.df[col].dtype == 'object' else self.df[col].mean()))

    @log_action("🔍 Extracting specific patterns")
    def extract_pattern(self, col_name: str, pattern: str, new_col: str) -> None:
        """Extract patterns from the specified column using regex."""
        if col_name in self.df.columns:
            self.df[new_col] = self.df[col_name].apply(
                lambda x: re.search(pattern, x).group(0) if pd.notnull(x) and re.search(pattern, x) else None
            )

    @log_action("❎ Correcting errors")
    def fix_errors(self, col_name: str, correction_func) -> None:
        if col_name in self.df.columns:
            self.df[col_name] = self.df[col_name].apply(correction_func)

    @log_action("🔍 Extracting specific patterns")
    def extract_pattern(self, col_name: str, pattern: str, new_col: str) -> None:
        """Extract patterns from the specified column using regex."""
        if col_name in self.df.columns:
            self.df[new_col] = self.df[col_name].apply(
                lambda x: re.search(pattern, x).group(0) if pd.notnull(x) and re.search(pattern, x) else None
            )
        
    @log_action("⚠️ Détection et suppression des outliers")
    def remove_outliers(self, method='IQR', factor=1.5) -> None:
        for col in get_numeric_columns(self.df):
            if method == 'IQR':
                Q1, Q3 = self.df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                self.df = self.df[(self.df[col] >= Q1 - factor * IQR) & (self.df[col] <= Q3 + factor * IQR)]

    @log_action("🧹 Nettoyage des espaces blancs")
    def clean_whitespace(self) -> None:
        for col in get_categorical_columns(self.df):
            self.df[col] = self.df[col].str.strip()
            
    @log_action("🗂️ Normalisation des colonnes de date")
    def normalize_date_columns(self, date_format='%Y-%m-%d') -> None:
        for col in get_datetime_columns(self.df):
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce').dt.strftime(date_format)

    @log_action("🆑 Suppression des doublons - duplicata")
    def remove_duplicates(self) -> None:
        self.df.drop_duplicates(inplace=True)

    def summarize(self) -> dict:
        return {
            'shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'head': self.df.head().to_dict(),
            'log': self.log
        }

    def display_info(self) -> None:
        print(self.df.info())

    def to_csv(self, path: str, sep='\t') -> None:
        self.df.to_csv(path, sep=sep, index=False)

    def to_excel(self, path: str, sheet_name='Sheet1') -> None:
        self.df.to_excel(path, sheet_name=sheet_name, index=False)

class FeatureCleaning (DataCleaning):
    __init__ = DataCleaning.__init__

    @log_action("🔄 Encodage des variables catégorielles")
    def encode_categorical(self, encoding_strategy='onehot') -> None:
        if encoding_strategy == 'onehot':
            self.df = pd.get_dummies(self.df, columns=get_categorical_columns(self.df))
        elif encoding_strategy == 'label':
            le = LabelEncoder()
            for col in get_categorical_columns(self.df):
                self.df[col] = le.fit_transform(self.df[col])

    @log_action("🔢 Normalisation des colonnes numériques")
    def normalize_numeric_columns(self, method='minmax') -> None:
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        self.df[get_numeric_columns(self.df)] = scaler.fit_transform(self.df[get_numeric_columns(self.df)])

    @log_action("🔄 Transformation des colonnes de date en features temporelles")
    def transform_date_columns(self) -> None:
        for col in get_datetime_columns(self.df):
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            self.df[f'{col}_year'] = self.df[col].dt.year
            self.df[f'{col}_month'] = self.df[col].dt.month
            self.df[f'{col}_day'] = self.df[col].dt.day
            self.df.drop(columns=[col], inplace=True)