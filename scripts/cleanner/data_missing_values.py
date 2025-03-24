"""
This module contains classes to process and enhance a DataFrame from a CSV file.

Primary Functions & Classes:
    async_executor: Decorator to run a function asynchronously using ThreadPoolExecutor.
    DataFrameProcessor: Class to process basic operations on a DataFrame from a CSV file.
    AdvancedDataFrameProcessor: Class that extends DataFrameProcessor to handle advanced data processing tasks.

@author: Feurking
"""

# [Tache1.0][5pts] Repérer et nettoyer le jeu de données des valeurs problématiques
# python -m scripts --file_path "C:/Users/Quentin/Desktop/machine learninig/teaching_ml_bis_2025/data/en.openfoodfacts.org.products.csv" --output_path "C:/Users/Quentin/Documents/results" --missing_threshold 0.5 --pattern_col "serving_size" --pattern "(\d+)" --irrelevant_cols id unwanted_col

import pandas as pd
import numpy as np
from termcolor import colored
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.linear_model import LinearRegression
import difflib
from functools import lru_cache
import numba

import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Literal

from ..utilities.data_utils import load_data, log_action

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
        """
        Initialize the DataFrameProcessor with a CSV file path, category threshold, and row limit.
        
        @param file_path: Path to the CSV file
        @param category_threshold: Threshold for unique values to classify columns
        @param limit: Maximum number of rows to load from the CSV file
        """
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
        """
        Classify columns into numeric, ordinal, and nominal based on unique value count.
        
        - Return -> tuple :
        Tuple containing lists of numeric, ordinal, and nominal columns
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        object_cols = self.df.select_dtypes(include=['object']).columns

        ordinal_cols = [col for col in object_cols if self.df[col].nunique() <= self.category_threshold]
        nominal_cols = [col for col in object_cols if self.df[col].nunique() > self.category_threshold]

        return numeric_cols, ordinal_cols, nominal_cols
    
    def _knn_imputation(self, n_neighbors=5, weights='uniform', metric='nan_euclidean', **kwargs) -> None:
        """
        Perform KNN imputation for missing values, handling non-numeric columns properly.
        
        
        @param n_neighbors: Number of neighbors to use for imputation
        @param weights: Weight function used in prediction
        @param metric: Distance metric to use for missing values
        @param kwargs: Additional keyword arguments for the KNNImputer
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = self.df.select_dtypes(exclude=[np.number]).columns

        if numeric_cols.empty: return

        if self.df[numeric_cols].isnull().all().all(): return
        
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights, metric=metric, **kwargs)
        self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])

        self.df[non_numeric_cols] = self.df[non_numeric_cols].fillna('Unknown')

        print(f"✅ KNN imputation done on numeric columns: {list(numeric_cols)}")
        if not non_numeric_cols.empty:
            print(f"⚠️ Non-numeric columns filled with 'Unknown': {list(non_numeric_cols)}")

    def _simple_imputation(self, strategy: Literal['most_frequent', 'mean']) -> None:
        """
        Perform simple imputation for missing values.
        
        @param strategy: Strategy to use for imputation (most_frequent, mean)
        """
        imputer = SimpleImputer(strategy=strategy)
        self.df = pd.DataFrame(imputer.fit_transform(self.df), columns=self.df.columns)

    def _cca_imputation(self, col: str = "") -> None:
        """
        Perform complete case analysis imputation for a specific column.

        @param col: Column name to impute
        """
        if col in self.df.columns:
            initial_rows = len(self.df)
            self.df = self.df[self.df[col].notna()]
            removed_rows = initial_rows - len(self.df)
            print(f"✅ CCA imputation done for column '{col}'. Rows removed: {removed_rows}")
        else: print(f"❌ Column '{col}' not found for CCA imputation.")

    def _arbitrary_imputation(self, col: str = "", value: float = 0) -> None:
        """
        Perform arbitrary value imputation for a specific column.

        @param col: Column name to impute
        @param value: Value to use for imputation
        """
        if col not in self.df.columns: return

        missing_count = self.df[col].isnull().sum()
        if missing_count == 0: return

        self.df[col].fillna(value, inplace=True)
        print(f"✅ Arbitrary imputation done for column '{col}' with value {value}. Missing values filled: {missing_count}")

    def _linear_regression_imputation(self, col: str):
        """
        Impute missing values in a column using Linear Regression based on other numeric columns.
        
        @param col: Column name to impute
        """
        print(f"⚠️ Linear regression imputation for column: {col}")

        if self.df[col].notnull().sum() < 2:
            print(f"❌ Pas assez de données pour imputer '{col}' avec régression linéaire. Utilisation de la moyenne.")
            self.df[col].fillna(self.df[col].mean(), inplace=True)
            return

        feature_cols = [c for c in self.df.select_dtypes(include=[np.number]).columns if c != col]

        if len(feature_cols) == 0:
            print(f"❌ Aucune colonne numérique disponible pour la régression. Imputation par la médiane.")
            self.df[col].fillna(self.df[col].median(), inplace=True)
            return

        known_df = self.df[self.df[col].notnull()]
        null_df = self.df[self.df[col].isnull()]

        if known_df.shape[0] == 0: return

        X_train = known_df[feature_cols]
        y_train = known_df[col]

        if X_train.shape[0] == 0: return

        if null_df.shape[0] == 0: return

        model = LinearRegression()
        model.fit(X_train, y_train)

        self.df.loc[self.df[col].isnull(), col] = model.predict(null_df[feature_cols])

        print(f"✅ Imputation terminée pour '{col}' avec régression linéaire.")

    def _mark_imputed_values(self, col: str) -> None:
        """
        Mark imputed values in a column with 'Imputed'.
        
        @param col: Name of the column to modify.
        """
        if col not in self.df.columns: return
        
        missing_mask = self.df[col].isna()
        missing_count = missing_mask.sum()

        if missing_count == 0: return

        self.df.loc[missing_mask, col] = "Imputed"
        print(f"🔄 {missing_count} valeurs imputées dans '{col}' avec 'Imputed'.")

    @lru_cache(maxsize=10)
    def _compute_correlation_matrix(self) -> pd.DataFrame:
        """
        Compute and cache the correlation matrix to avoid recalculations.

        - Return -> pd.DataFrame: 
            Correlation matrix.
        """
        return self.df.corr(numeric_only=True)

    def _correlation(self, min_corr: float = 0.0) -> None:
        """
        Removes columns with an absolute correlation below the defined threshold.

        @param min_corr: Minimum correlation threshold to retain columns.
        """
        if self.df.empty: return
        
        corr_matrix = self._compute_correlation_matrix()
        drop_cols = [col for col in corr_matrix.columns if np.all(np.abs(corr_matrix[col]) < min_corr)]

        if drop_cols:
            self.df.drop(columns=drop_cols, inplace=True)
            print(f"🗑️ Colonnes supprimées (faible corrélation < {min_corr}): {drop_cols}")
        else: print("✅ Aucune colonne supprimée, toutes respectent le seuil.")

    def _missing_values(self, max_missing: float = 0.0) -> None:
        """
        Removes columns with a percentage of missing values exceeding the threshold.

        @param max_missing: Maximum threshold for missing values.
        """
        missing_ratio = self.df.isna().mean()
        drop_cols = missing_ratio.index[missing_ratio > max_missing].tolist()

        if drop_cols:
            self.df.drop(columns=drop_cols, inplace=True)
            print(f"🗑️ Colonnes supprimées (> {max_missing * 100}% de valeurs manquantes): {drop_cols}")
        else: print("✅ Aucune colonne supprimée, toutes respectent le seuil.")

    @staticmethod
    @numba.njit
    def _compute_variance(array: np.ndarray) -> float:
        """
        Compute the variance of a numpy array using numba for massive acceleration.
        
        @param array: Numpy array to compute the variance of.
        @return: Variance of the array.
        """
        mean = np.mean(array)
        return np.mean((array - mean) ** 2)

    def _variance(self, min_variance: float = 0.0) -> None:
        """
        Removes columns with variance below the defined threshold.

        @param min_variance: Minimum variance threshold to retain columns.
        """
        numeric_cols = self.df.select_dtypes(include=[np.number])
        
        variances = {col: self._compute_variance(self.df[col].dropna().values) for col in numeric_cols}
        drop_cols = [col for col, var in variances.items() if var < min_variance]

        if drop_cols:
            self.df.drop(columns=drop_cols, inplace=True)
            print(f"🗑️ Colonnes supprimées (faible variance < {min_variance}): {drop_cols}")
        else: print("✅ Aucune colonne supprimée, toutes respectent le seuil.")

    @async_executor
    def _apply_downcasting(self) -> None:
        """Apply downcasting to numeric columns to reduce memory usage."""
        self.df[self.numeric_columns] = self.df[self.numeric_columns].apply(pd.to_numeric, downcast='integer')

    @log_action("📊 Processing DataFrame")
    async def process_dataframe(self) -> pd.DataFrame:
        """
        Process the dataframe by applying downcasting and printing column categories.
        
        - Return -> pd.DataFrame:
            Processed DataFrame.
        """
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

    def __call_methods__(self, method: str, **kwargs) -> None:
        """
        Call specified methods with keyword arguments.

        @param methods: List of methods to call
        @param kwargs: Additional keyword arguments for each method
        """
        method = "_" + method + "_imputation"

        if hasattr(self, method) :
            getattr(self, method)(**kwargs)
        else:
            raise ValueError(f"Method {method} not found in DataFrameProcessor.")
            
    def __col_content__(self) -> str:
        for col in self.df.columns:
            if self.df[col].isnull().any():
                return col
        return ""
    
    def _start_function(self, method : str, methods : list, **kwargs) -> None:
        for method in methods:
            self.__call_methods__(method, **kwargs)

    @log_action("🔄 Imputation of missing values")
    def impute_missing_values(self, method='knn', missing_threshold=0.5, strategy="mean", col: str = "", **kwargs):
        """
        Impute missing values using the specified method based on the percentage of missing values.
        
        @param method: Imputation method to use (knn, frequent, statistical, iterative, cca, arbitrary, linear_regression, mark)
        @param missing_threshold: Threshold for missing values to trigger imputation
        @param strategy: Strategy to use for simple imputation (mean, median, mode)
        @param col: Column name to apply the imputation on
        @param kwargs: Additional keyword arguments for the imputation method
        """

        self.df.columns = self.df.columns.str.lower()
        col = col.strip().lower()

        if col not in self.df.columns:
            similar_columns = difflib.get_close_matches(col, self.df.columns, n=1, cutoff=0.4)
            if similar_columns:
                print(f"⚠️ Colonne '{col}' non trouvée, remplacement par '{similar_columns[0]}'.")
                col = similar_columns[0]
            else:
                raise ValueError(f"❌ Colonne '{col}' introuvable et aucune suggestion proche disponible.")

        print(f"🔄 Imputation en cours sur la colonne : '{col}'")

        methods_of_this_function = {
            'knn': lambda: self._knn_imputation(missing_threshold=kwargs.get('missing_threshold', missing_threshold), **kwargs),
            'frequent': lambda: self._simple_imputation(strategy='most_frequent'),
            'iterative': lambda: IterativeImputer(**kwargs),
            'cca': lambda: self._cca_imputation(col=col),
            'arbitrary': lambda: self._arbitrary_imputation(col=col, value=kwargs.get('value', 0)),
            'linear_regression': lambda: self._linear_regression_imputation(col=col, **kwargs),
            'mark': lambda: self._mark_imputed_values(col=col),
            'simple': lambda: self._simple_imputation(strategy=strategy)
        }

        method = method.lower()
        if method in methods_of_this_function:
            methods_of_this_function[method]()
        else:
            raise ValueError(f"❌ Méthode '{method}' non reconnue. Méthodes possibles : {list(methods_of_this_function.keys())}")
        
    @log_action("🔍 Filtering irrelevant columns")
    def filter_irrelevant_columns(self, methods=['variance', 'missing_values', 'correlation'], **kwargs) -> None:
        """
        Filter out columns based on specified methods.

        @param methods: List of methods to apply (variance, missing_values, correlation)
        @param kwargs: Additional keyword arguments for each method
        """
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
        """
        Detect potential errors in the DataFrame and categorize them.

        - Return -> dict :
            A dictionary containing the detected errors
        """
        methods_of_this_function = {
            'missing_values': lambda: self.df.isnull().sum()[self.df.isnull().sum() > 0],
            'duplicate_rows': lambda: self.df.duplicated().sum(),
            'invalid_data_types': lambda: self.df.applymap(lambda x: isinstance(x, (list, dict, set))).any(),
            'outliers': lambda: self.detect_outliers(),
            'inconsistent_values': lambda: self.detect_inconsistent_values()
        }

        errors = {}

        for error_type, method in methods_of_this_function.items():
            errors[error_type] = method()
            
        return errors

    def resolve_errors(self, strategies: dict = None) -> None:
        """
        Resolve detected errors based on specified strategies.

        @param strategies: Dictionary of strategies to apply for each error type
        """
        methods_of_this_function = {
            'missing_values': lambda: self.handle_missing_values(strategy=strategies['missing_values']),
            'duplicate_rows': lambda: self.df.drop_duplicates(inplace=True),
            'invalid_data_types': lambda: [self.df.drop(columns=[col], inplace=True) for col in strategies['invalid_data_types']],
            'outliers': lambda: self.handle_outliers(method=strategies['outliers']),
            'inconsistent_values': lambda: self.correct_inconsistent_values()
        }

        strategies = strategies or {}

        for error_type, _ in strategies.items():
            if error_type in methods_of_this_function:
                methods_of_this_function[error_type]()
            else:
                raise ValueError(f"Error type {error_type} not recognized in resolve_errors.")

    def handle_missing_values(self, strategy='mean') -> None:
        """
        Handle missing values with different strategies.

        @param strategy: Strategy to use for missing value imputation (drop, mean, median, mode)
        """

        methods_of_this_function = {
            'drop': lambda: self.df.dropna(inplace=True),
            'mean': lambda: self.df.fillna(self.df.mean(), inplace=True),
            'median': lambda: self.df.fillna(self.df.median(), inplace=True),
            'mode': lambda: self.df.fillna(self.df.mode().iloc[0], inplace=True)
        }

        if strategy in methods_of_this_function:
            methods_of_this_function[strategy]()
        else:
            raise ValueError(f"❌ Strategy '{strategy}' not recognized. Possible strategies: {list(methods_of_this_function.keys())}")
        
        print(f"✅ Missing values handled with strategy: {strategy}")

    def detect_outliers(self) -> pd.DataFrame:
        """
        Detect outliers using the IQR method.

        - Return -> pd.DataFrame :
            A DataFrame containing the detected outliers
        """
        outliers = {}
        for col in self.numeric_columns:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            mask = (self.df[col] < (q1 - 1.5 * iqr)) | (self.df[col] > (q3 + 1.5 * iqr))
            outliers[col] = self.df[mask]
        return pd.concat(outliers.values()) if outliers else pd.DataFrame()

    def handle_outliers(self, method='clip') -> None:
        """
        Handle detected outliers.
        
        @param method: Method to use for handling outliers (remove, clip)
        """
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
        """
        Detect inconsistencies in the data (e.g., negative ages).
        
        - Return -> dict :
            A dictionary containing columns with inconsistent values
        """
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
        """
        Validate that columns adhere to specific value ranges.
        
        @param col_ranges: Dictionary containing column names and their value ranges
        """
        for col, (min_val, max_val) in col_ranges.items():
            mask = (self.df[col] < min_val) | (self.df[col] > max_val)
            if mask.any():
                print(f"❗ Values out of range detected in {col}.")
                self.df.loc[mask, col] = np.nan

        print("✅ Value ranges validated.")

    def extract_patterns(self, column: str, pattern: str) -> pd.Series:
        """
        Extract specific patterns from a column and return the result as a Series.
        
        @param column: Column to extract patterns from
        @param pattern: Regular expression pattern to extract
        
        @return: A Series containing the extracted patterns
        """
        extracted_series = self.df[column].str.extract(pattern, expand=False)
        print(f"Extracted patterns from column: {column}")
        return extracted_series