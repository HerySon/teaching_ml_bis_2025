"""
This module contains a class to process a DataFrame from a CSV file.

Primary Functions & Classes:
    async_executor: Decorator to run a function asynchronously using ThreadPoolExecutor.
    DataFrameProcessor: Class to process a DataFrame from a CSV file.

@author: Feurking
"""

import pandas as pd
import numpy as np

import argparse
import asyncio
import pytest

from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

from functools import wraps
from termcolor import colored

def async_executor(func) -> asyncio.coroutine:
    """Decorator to run a function asynchronously using ThreadPoolExecutor."""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> None:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, func, *args, **kwargs)
    return wrapper

class DataFrameProcessor:
    @pytest.mark.parametrize("file_path, category_threshold, limit", [
        ("../data/en.openfoodfacts.org.products.csv", 10, None),
        ("../data/en.openfoodfacts.org.products.csv", 5, 1000),
        ("../data/en.openfoodfacts.org.products.csv", 20, 500),
    ])
    def __init__(self, file_path: str, category_threshold: int = 10, limit: int = None) -> None:
        """Initialize the DataFrameProcessor with a CSV file path, category threshold, and row limit.
            @param file_path: Path to the CSV file
            @param category_threshold: Maximum number of categories for ordinal columns
            @param limit: Maximum number of rows to load
        """
        self.file_path = file_path
        self.category_threshold = category_threshold
        self.limit = limit
        self.df = self._load_data()
        self._numeric_columns, self._ordinal_columns, self._nominal_columns = self._classify_columns()

    def __repr__(self) -> str:
        return f"DataFrameProcessor(file_path='{self.file_path}', category_threshold={self.category_threshold}, limit={self.limit})"

    def __str__(self) -> str:
        return f"DataFrameProcessor: {Path(self.file_path).name}"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, key) -> pd.Series:
        return self.df[key]

    def __setitem__(self, key, value) -> None:
        self.df[key] = value

    def __delitem__(self, key) -> None:
        del self.df[key]

    def __iter__(self) -> iter:
        return iter(self.df)

    def __contains__(self, key) -> bool:
        return key in self.df

    def _load_data(self) -> pd.DataFrame:
        """Load the data from a CSV file with efficient memory usage."""
        return pd.read_csv(self.file_path, sep='\t', low_memory=False, nrows=self.limit)

    def _classify_columns(self) -> tuple:
        """Classify columns into numeric, ordinal, and nominal based on unique value count."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        object_cols = self.df.select_dtypes(include=['object']).columns

        ordinal_cols = [col for col in object_cols if self.df[col].nunique() <= self.category_threshold]
        nominal_cols = [col for col in object_cols if self.df[col].nunique() > self.category_threshold]

        return numeric_cols, ordinal_cols, nominal_cols

    @async_executor
    def _apply_downcasting(self) -> None:
        """Apply downcasting to numeric columns to reduce memory usage."""
        self.df[self.numeric_columns] = self.df[self.numeric_columns].apply(pd.to_numeric, downcast='integer')

    async def process_dataframe(self) -> pd.DataFrame:
        """Process the dataframe by applying downcasting and printing column categories."""
        await self._apply_downcasting()
        print(colored("\n 📚 Colonnes numériques:", 'blue'), self.numeric_columns)
        print(colored("📚 Colonnes catégorielles ordinales:", 'green'), self.ordinal_columns)
        print(colored("📚 Colonnes catégorielles nominales:", 'yellow'), self.nominal_columns)

        memory_usage = self.df.memory_usage(deep=True).sum() / (1024 ** 2)
        print(f"💾 Utilisation mémoire: {memory_usage:.2f} Mo")

        return self.df

    @property
    def numeric_columns(self) -> list:
        return self._numeric_columns

    @numeric_columns.setter
    def numeric_columns(self, value) -> None:
        if not isinstance(value, list) or any(not isinstance(col, str) for col in value):
            raise ValueError("Les colonnes numériques doivent être une liste de chaînes de caractères")
        self._numeric_columns = value

    @property
    def ordinal_columns(self) -> list:
        return self._ordinal_columns

    @ordinal_columns.setter
    def ordinal_columns(self, value) -> None:
        if not isinstance(value, list) or any(not isinstance(col, str) for col in value):
            raise ValueError("Les colonnes ordinales doivent être une liste de chaînes de caractères")
        self._ordinal_columns = value

    @property
    def nominal_columns(self) -> list:
        return self._nominal_columns

    @nominal_columns.setter
    def nominal_columns(self, value) -> None:
        if not isinstance(value, list) or any(not isinstance(col, str) for col in value):
            raise ValueError("Les colonnes nominales doivent être une liste de chaînes de caractères")
        self._nominal_columns = value

    def filter_relevant_columns(self, min_corr=0.05) -> None:
        """Filter columns based on correlation threshold for numeric variables."""
        corr_matrix = self.df.corr(numeric_only=True)
        relevant_numeric = [col for col in corr_matrix.columns if any(abs(corr_matrix[col]) >= min_corr)]
        print("🔍 Colonnes numériques pertinentes (corr ≥", min_corr, "):", relevant_numeric)

        self.numeric_columns = relevant_numeric

        self.ordinal_columns = [col for col in self.ordinal_columns if self.df[col].nunique() > 1]
        self.nominal_columns = [col for col in self.nominal_columns if self.df[col].nunique() > 1]
