"""
This module contains classes to process and enhance a DataFrame from a CSV file.

Primary Functions & Classes:
    async_executor: Decorator to run a function asynchronously using ThreadPoolExecutor.
    DataFrameProcessor: Class to process basic operations on a DataFrame from a CSV file.
    AdvancedDataFrameProcessor: Class that extends DataFrameProcessor to handle advanced data processing tasks.

@author: Feurking
"""

try:
    import pandas as pd
    import numpy as np
    import pytest
    from termcolor import colored
    from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
    from sklearn.linear_model import LinearRegression
except ImportError as e:
    print(f"Error: {e}. Please make sure to install the required packages.")

import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Literal
from utilities.data_utils import load_data, log_action

def async_executor(func):
    """Decorator to run a function asynchronously using ThreadPoolExecutor."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, func, *args, **kwargs)
    return wrapper

class DataFrameProcessor(object):
    """Base class for processing a DataFrame from a CSV file."""

    def __init__(self, file_path: str, category_threshold: int = 10, limit: int = None) -> None:
        """Initialize the DataFrameProcessor with a CSV file path, category threshold, and row limit."""
        self.file_path = file_path
        self.category_threshold = category_threshold
        self.limit = limit
        self.df = load_data(file_path, limit)
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

    def _classify_columns(self) -> tuple:
        """Classify columns into numeric, ordinal, and nominal based on unique value count."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        object_cols = self.df.select_dtypes(include=['object']).columns

        ordinal_cols = [col for col in object_cols if self.df[col].nunique() <= self.category_threshold]
        nominal_cols = [col for col in object_cols if self.df[col].nunique() > self.category_threshold]

        return numeric_cols, ordinal_cols, nominal_cols
    
    def _knn_imputation(self, n_neighbors=5, weights='uniform', metric='nan_euclidean', **kwargs) -> None:
        """Perform KNN imputation for missing values with more complex configurations."""
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights, metric=metric, **kwargs)
        self.df = pd.DataFrame(imputer.fit_transform(self.df), columns=self.df.columns)

    def _simple_imputation(self, strategy: Literal['most_frequent', 'mean']) -> None:
        """Perform simple imputation for missing values."""
        imputer = SimpleImputer(strategy=strategy)
        self.df = pd.DataFrame(imputer.fit_transform(self.df), columns=self.df.columns)

    def _cca_imputation(self, col: str) -> None:
        """Perform complete case analysis imputation for a specific column."""
        self.df.dropna(subset=[col], inplace=True)

    def _arbitrary_imputation(self, col: str, value: float) -> None:
        """Perform arbitrary value imputation for a specific column."""
        self.df[col].fillna(value, inplace=True)

    def _linear_regression_imputation(self, col: str) -> None:
        """Perform linear regression imputation for a specific column."""
        not_null_df = self.df[self.df[col].notnull()]
        null_df = self.df[self.df[col].isnull()]
        model = LinearRegression()
        model.fit(not_null_df.drop(columns=[col]), not_null_df[col])
        self.df.loc[self.df[col].isnull(), col] = model.predict(null_df.drop(columns=[col]))

    def _mark_imputed_values(self, col: str) -> None:
        """Mark imputed values in a specific column."""
        self.df[col].fillna('Imputed', inplace=True)

    def _correlation(self, min_corr: float) -> None:
        """Filter out columns based on correlation threshold."""
        corr_matrix = self.df.corr(numeric_only=True)
        for col in corr_matrix.columns:
            if all(abs(corr_matrix[col]) < min_corr):
                self.df.drop(columns=[col], inplace=True)

    def _missing_values(self, max_missing: float) -> None:
        """Filter out columns based on missing value threshold."""
        for col in self.df.columns:
            if self.df[col].isnull().mean() > max_missing:
                self.df.drop(columns=[col], inplace=True)

    def _variance(self, min_variance: float) -> None:
        """Filter out columns based on variance threshold."""
        for col in self.df.columns:
            if self.df[col].var() < min_variance:
                self.df.drop(columns=[col], inplace=True)

    @async_executor
    def _apply_downcasting(self) -> None:
        """Apply downcasting to numeric columns to reduce memory usage."""
        self.df[self.numeric_columns] = self.df[self.numeric_columns].apply(pd.to_numeric, downcast='integer')

    @log_action("📊 Processing DataFrame")
    async def process_dataframe(self) -> pd.DataFrame:
        """Process the dataframe by applying downcasting and printing column categories."""
        await self._apply_downcasting()

        print(colored("\n 📚 Numeric columns:", 'blue'))
        print(self.numeric_columns)
        print(colored("📚 Ordinal categorical columns:", 'green'))
        print(self.ordinal_columns)
        print(colored("📚 Nominal categorical columns:", 'yellow'))
        print(self.nominal_columns)

        memory_usage = self.df.memory_usage(deep=True).sum() / (1024 ** 2)
        print(f"💾 Memory usage: {memory_usage:.2f} MB")

        return self.df

    @property
    def numeric_columns(self) -> list:
        return self._numeric_columns

    @numeric_columns.setter
    def numeric_columns(self, value) -> None:
        if not isinstance(value, list) or any(not isinstance(col, str) for col in value):
            raise ValueError("Numeric columns must be a list of strings.")
        self._numeric_columns = value

    @property
    def ordinal_columns(self) -> list:
        return self._ordinal_columns

    @ordinal_columns.setter
    def ordinal_columns(self, value) -> None:
        if not isinstance(value, list) or any(not isinstance(col, str) for col in value):
            raise ValueError("Ordinal columns must be a list of strings.")
        self._ordinal_columns = value

    @property
    def nominal_columns(self) -> list:
        return self._nominal_columns

    @nominal_columns.setter
    def nominal_columns(self, value) -> None:
        if not isinstance(value, list) or any(not isinstance(col, str) for col in value):
            raise ValueError("Nominal columns must be a list of strings.")
        self._nominal_columns = value

class AdvancedDataFrameProcessor(DataFrameProcessor):
    """Advanced class for enhanced data processing tasks, inheriting from DataFrameProcessor."""

    __init__ = DataFrameProcessor.__init__

    def _call_methods__(self, methods: list, **kwargs) -> None:
        """Call specified methods with keyword arguments."""
        for method in methods:
            if hasattr(self, method):
                getattr(self, method)(**kwargs)
            else:
                raise ValueError(f"Method {method} not found in DataFrameProcessor.")
            
    def __col_content__(self) -> pd.Series:
        """Get the content of a specific column."""
        colums = self.df.columns
        for col in colums:
            if self.df[col].isnull().any():
                return col, self.df[col].isnull().sum()
            
        return colums.to_string(), self.df[col].value_counts()

    @log_action("🔄 Imputation of missing values")
    def impute_missing_values(self, method='knn', missing_threshold=0.2, **kwargs) -> None:
        """Impute missing values using the specified method based on the percentage of missing values."""
        methods_of_this_function = {
            'knn': self._knn_imputation,
            'frequent': lambda: self._simple_imputation(strategy='most_frequent'),
            'statistical': lambda: self._simple_imputation(strategy='mean'),
            'iterative': lambda: IterativeImputer(**kwargs),
            'cca': lambda: self._cca_imputation(col=self.__col_content__()[0]),
            'arbitrary': lambda: self._arbitrary_imputation(col=self.__col_content__()[0], value=kwargs.get('value', 0)),
            'linear_regression': lambda: self._linear_regression_imputation(col=self.__col_content__()[0]),
            'mark': lambda: self._mark_imputed_values(col=self.__col_content__()[0]),
            'simple': lambda: self._simple_imputation(strategy='mean')
        }

        method = method.lower()
        
        if method in methods_of_this_function:
            imputer = self.__call_methods__(methods=[method], **kwargs)
            if callable(imputer):
                imputer()
            else:
                self.df = pd.DataFrame(imputer.fit_transform(self.df), columns=self.df.columns)
        else:
            raise ValueError(f"Imputation method {method} not recognized.")

    @log_action("🔍 Filtering irrelevant columns")
    def filter_irrelevant_columns(self, methods=['variance', 'missing_values', 'correlation'], **kwargs) -> None:
        """Filter out columns based on specified methods."""
        methods_of_this_function = {
            'variance': lambda: self._variance(min_variance=kwargs.get('min_variance', 0.1)),
            'missing_values': lambda: self._missing_values(max_missing=kwargs.get('max_missing', 0.2)),
            'correlation': lambda: self._correlation(min_corr=kwargs.get('min_corr', 0.5))
        }

        for method in methods:
            if method in methods_of_this_function:
                self.__call_methods__(methods=[method], **kwargs)
            else:
                raise ValueError(f"Method {method} not recognized in filter_irrelevant_columns.")

    @log_action("🔍 Extracting errors")
    def get_errors(self) -> dict:
        """Detect potential errors in the DataFrame and categorize them."""
        errors = {}

        if self.df.isnull().any().any():
            errors['missing_values'] = self.df.isnull().sum()[self.df.isnull().sum() > 0]
        
        if self.df.duplicated().any():
            errors['duplicate_rows'] = self.df.duplicated().sum()
        
        invalid_data_types = self.df.applymap(lambda x: isinstance(x, (list, dict, set)))
        if invalid_data_types.any().any():
            errors['invalid_data_types'] = self.df.columns[invalid_data_types.any()].tolist()

        outliers = self.detect_outliers()
        if not outliers.empty:
            errors['outliers'] = outliers

        inconsistent_values = self.detect_inconsistent_values()
        if inconsistent_values:
            errors['inconsistent_values'] = inconsistent_values

        return errors

    def resolve_errors(self, strategies: dict = None) -> None:
        """Resolve detected errors based on specified strategies."""
        strategies = strategies or {}

        if 'missing_values' in strategies:
            self.handle_missing_values(strategy=strategies['missing_values'])

        if 'duplicate_rows' in strategies:
            self.df.drop_duplicates(inplace=True)
            print("✅ Duplicate rows removed.")

        if 'invalid_data_types' in strategies:
            for col in strategies['invalid_data_types']:
                self.df.drop(columns=[col], inplace=True)
            print("✅ Columns with invalid data types removed.")

        if 'outliers' in strategies:
            self.handle_outliers(method=strategies['outliers'])

        if 'inconsistent_values' in strategies:
            self.correct_inconsistent_values()

    def handle_missing_values(self, strategy='mean') -> None:
        """Handle missing values with different strategies."""
        if strategy == 'drop':
            self.df.dropna(inplace=True)
        elif strategy == 'mean':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
        elif strategy == 'median':
            for col in self.numeric_columns:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        elif strategy == 'mode':
            for col in self.df.columns:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        print(f"✅ Missing values handled with strategy: {strategy}")

    def detect_outliers(self) -> pd.DataFrame:
        """Detect outliers using the IQR method."""
        outliers = {}
        for col in self.numeric_columns:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            mask = (self.df[col] < (q1 - 1.5 * iqr)) | (self.df[col] > (q3 + 1.5 * iqr))
            outliers[col] = self.df[mask]
        return pd.concat(outliers.values()) if outliers else pd.DataFrame()

    def handle_outliers(self, method='clip') -> None:
        """Handle detected outliers."""
        for col in self.numeric_columns:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            if method == 'remove':
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
            elif method == 'clip':
                self.df[col] = self.df[col].clip(lower_bound, upper_bound)
        print(f"✅ Outliers handled with method: {method}")

    def detect_inconsistent_values(self) -> dict:
        """Detect inconsistencies in the data (e.g., negative ages)."""
        inconsistent_values = {}
        for col in self.numeric_columns:
            if (self.df[col] < 0).any():
                inconsistent_values[col] = self.df[self.df[col] < 0]
        return inconsistent_values

    def correct_inconsistent_values(self) -> None:
        """Correct inconsistent values by taking the absolute value."""
        for col in self.numeric_columns:
            self.df[col] = self.df[col].abs()
        print("✅ Inconsistent values corrected.")

    def validate_column_ranges(self, col_ranges: dict) -> None:
        """Validate that columns adhere to specific value ranges."""
        for col, (min_val, max_val) in col_ranges.items():
            mask = (self.df[col] < min_val) | (self.df[col] > max_val)
            if mask.any():
                print(f"❗ Values out of range detected in {col}.")
                self.df.loc[mask, col] = np.nan

        print("✅ Value ranges validated.")

    def extract_patterns(self, column: str, pattern: str) -> pd.Series:
        """Extract specific patterns from a column and return the result as a Series."""
        extracted_series = self.df[column].str.extract(pattern, expand=False)
        print(f"Extracted patterns from column: {column}")
        return extracted_series