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

class DataCleaning:
    @pytest.mark.parametrize("file_path, limit", [
        ("/data/en.openfoodfacts.org.products.csv", 1000),
        ("/data/en.openfoodfacts.org.products.csv", 500),
        ("/data/en.openfoodfacts.org.products.csv", 200),
    ])
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self.log = []

    def __repr__(self) -> str:
        return (f"DataCleaner(shape={self.df.shape}, "
                f"num_cols={len(self._get_numeric_columns())}, "
                f"cat_cols={len(self._get_categorical_columns())}, "
                f"date_cols={len(self._get_datetime_columns())}, "
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

    def log_action(action: str) -> callable:
        """Decorator to log actions"""
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                start_time = datetime.now()
                print(f"[INFO {start_time}] - {action}")
                result = func(self, *args, **kwargs)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                self.log.append({"action": action, "duration": duration, "timestamp": start_time})
                return result
            return wrapper
        return decorator

    @classmethod
    def from_csv(cls, file_path : str, limit : int) -> 'DataCleaning':
        df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip', nrows=limit, low_memory=False)
        return cls(df)

    def _get_numeric_columns(self) -> list:
        return self.df.select_dtypes(include=[np.number]).columns.tolist()

    def _get_categorical_columns(self) -> list:
        return self.df.select_dtypes(include=['object']).columns.tolist()

    def _get_datetime_columns(self) -> list:
        return self.df.select_dtypes(include=['datetime']).columns.tolist()

    @log_action("🪪 Suppression des colonnes non informatives")
    def drop_uninformative_columns(self) -> None:
        cols_to_drop = [col for col in self.df.columns if self.df[col].nunique() <= 1]
        self.df.drop(columns=cols_to_drop, inplace=True)

    @log_action("🚮 Suppression des colonnes non pertinentes")
    def drop_irrelevant_columns(self, irrelevant_cols: list) -> None:
        self.df.drop(columns=[col for col in irrelevant_cols if col in self.df.columns], inplace=True)

    @log_action("👊 Traitement des valeurs manquantes")
    def handle_missing_values(self, threshold: float = 0.5, strategies=None) -> None:
        """Handle missing values in the DataFrame using a threshold and strategies."""
        high_missing_cols = self.df.columns[self.df.isnull().mean() > threshold]
        self.df.drop(columns=high_missing_cols, inplace=True)
        strategies = strategies or {}

        for col in self.df.columns[self.df.isnull().any()]:
            if col not in high_missing_cols:
                self.df[col].fillna(strategies.get(col, self.df[col].mode()[0] if self.df[col].dtype == 'object' else self.df[col].mean()), inplace=True)

    @log_action("🔍 Extraction des motifs particuliers")
    def extract_pattern(self, col_name: str, pattern: str, new_col: str) -> None:
        if col_name in self.df.columns:
            self.df[new_col] = self.df[col_name].str.extract(pattern, expand=False)

    @log_action("❎ Correction des erreurs")
    def fix_errors(self, col_name: str, correction_func) -> None:
        if col_name in self.df.columns:
            self.df[col_name] = self.df[col_name].apply(correction_func)

    @log_action("🔍 Extraction des motifs particuliers")
    def extract_pattern(self, col_name: str, pattern: str, new_col: str) -> None:
        if col_name in self.df.columns:
            self.df[new_col] = self.df[col_name].apply(lambda x: re.search(pattern, x).group(0) if pd.notnull(x) and re.search(pattern, x) else None)
        
    @log_action("⚠️ Détection et suppression des outliers")
    def remove_outliers(self, method='IQR', factor=1.5) -> None:
        for col in self._get_numeric_columns():
            if method == 'IQR':
                Q1, Q3 = self.df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                self.df = self.df[(self.df[col] >= Q1 - factor * IQR) & (self.df[col] <= Q3 + factor * IQR)]

    @log_action("🧹 Nettoyage des espaces blancs")
    def clean_whitespace(self) -> None:
        for col in self._get_categorical_columns():
            self.df[col] = self.df[col].str.strip()
            
    @log_action("🗂️ Normalisation des colonnes de date")
    def normalize_date_columns(self, date_format='%Y-%m-%d') -> None:
        for col in self._get_datetime_columns():
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