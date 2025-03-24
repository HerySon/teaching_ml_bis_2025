"""
This module contains a class for encoding non-numeric features.

Primary Functions & Classes:
    FeatureEncoder: Class to encode categorical features efficiently.

@author: Feurking
"""

# [Tache1.1][5.5pts] Encoder les features non numériques

# python -m scripts --file_path "C:/Users/Quentin/Desktop/machine learninig/teaching_ml_bis_2025/data/en.openfoodfacts.org.products.csv" --output_path "C:/Users/Quentin/Desktop/Fuck YourSlef PEP8/teaching_ml_bis_2025/results/encoded_data.csv" --method "one-hot" --target_col "price"

import os
from datetime import datetime
import time

try :
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.cluster import KMeans
    from category_encoders import HashingEncoder, CountEncoder
    from scipy import sparse
    import pytest
    from category_encoders import HashingEncoder, CountEncoder, TargetEncoder, LeaveOneOutEncoder
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
 print(f"Error: {e}. Please make sure to install the required packages.")

from functools import wraps

from ..utilities.data_utils import load_data, log_action, get_categorical_columns

class DataEncoder(object):
    def __init__(self, df: pd.DataFrame, target_col : str) -> None:
        self.df = df.copy()
        self.target_col = target_col
        self.log = []

    @classmethod
    def from_csv(cls, file_path: str, target_col: str, limit: int = 1000) -> 'FeatureEncoder':
        df = load_data(file_path, limit)
        return cls(df, target_col)

    @log_action("📊 Save Encoded Data")
    def save_encoded_df(self, output_path: str = "encoded_df.csv") -> None:
        """
        Saves the encoded DataFrame to the specified file.
        
        @param output_path: The path to save the encoded DataFrame
        """
        self.df.to_csv(output_path, index=False)
        print(f"Encoded DataFrame saved to {output_path}")

    def summarize(self) -> dict:
        """
        Summarizes the DataFrame, including shape, categorical columns, and log.
        
        - Returns -> dict:
            A dictionary containing the shape, categorical columns, and log
        """
        return {
            'shape': self.df.shape,
            'categorical_columns': get_categorical_columns(self.df),
            'log': self.log
        }
    
    @log_action("🔄 One-Hot Encoding with Optimization")
    def _one_hot_encode(self, min_freq: float = 0.01) -> None:
        """ One-Hot Encoding with rare category merging. """
        self._merge_rare_categories(min_freq)

        cat_columns = get_categorical_columns(self.df)
        self.df = pd.get_dummies(self.df, columns=cat_columns, drop_first=True)

        print(f"✅ Applied One-Hot Encoding on: {cat_columns}")

    @log_action("🔢 Count Encoding")
    def _count_encode(self, min_freq: float = 0.01) -> None:
        """ Encodes categorical features using Count Encoding. """
        self._merge_rare_categories(min_freq)

        cat_columns = get_categorical_columns(self.df)

        self.df[cat_columns] = CountEncoder().fit_transform(self.df[cat_columns])
        print(f"✅ Applied Count Encoding on: {cat_columns}")

    @log_action("⚙️ Hash Encoding")
    def _hash_encode(self, n_components: int = 8) -> None:
        """
        Encodes categorical features using Hash Encoding.
        
        @param n_components: The number of components for the Hashing Encoder
        """
        self._merge_rare_categories()

        cat_columns = get_categorical_columns(self.df)

        for col in cat_columns:
            encoded = HashingEncoder(n_components=n_components).fit_transform(self.df[[col]])
            self.df = pd.concat([self.df.drop(columns=[col]), encoded], axis=1)
            print(f"Applied Hash Encoding on column: {col}")

    @log_action("🎯 Target Encoding")
    def _target_encode(self) -> None:
        """Encodes categorical features using Target Encoding."""
        if not self.target_col:
            raise ValueError("⚠️ Target column must be specified for Target Encoding.")

        cat_columns = get_categorical_columns(self.df)
        encoder = TargetEncoder()
        self.df[cat_columns] = encoder.fit_transform(self.df[cat_columns], self.df[self.target_col])
        print(f"✅ Applied Target Encoding on: {cat_columns}")

    @log_action("📊 Mean Encoding")
    def _mean_encode(self) -> None:
        """Encodes categorical features using Mean Encoding."""
        if not self.target_col:
            raise ValueError("⚠️ Target column must be specified for Mean Encoding.")

        cat_columns = get_categorical_columns(self.df)
        for col in cat_columns:
            self.df[col] = self.df[col].map(self.df.groupby(col)[self.target_col].mean())
        print(f"✅ Applied Mean Encoding on: {cat_columns}")

    @log_action("📈 Frequency Encoding")
    def _frequency_encode(self) -> None:
        """Encodes categorical features using Frequency Encoding."""
        cat_columns = get_categorical_columns(self.df)
        for col in cat_columns:
            self.df[col] = self.df[col].map(self.df[col].value_counts(normalize=True))
        print(f"✅ Applied Frequency Encoding on: {cat_columns}")

    @log_action("🔄 Leave-One-Out Encoding")
    def _leave_one_out_encode(self) -> None:
        """Encodes categorical features using Leave-One-Out Encoding."""
        if not self.target_col:
            raise ValueError("⚠️ Target column must be specified for Leave-One-Out Encoding.")

        cat_columns = get_categorical_columns(self.df)
        encoder = LeaveOneOutEncoder()
        self.df[cat_columns] = encoder.fit_transform(self.df[cat_columns], self.df[self.target_col])
        print(f"✅ Applied Leave-One-Out Encoding on: {cat_columns}")

    @log_action("⚖️ Weight of Evidence (WoE) Encoding")
    def _weight_of_evidence_encode(self) -> None:
        """Encodes categorical features using Weight of Evidence."""
        if not self.target_col:
            raise ValueError("⚠️ Target column must be specified for WoE Encoding.")

        cat_columns = get_categorical_columns(self.df)
        for col in cat_columns:
            grouped = self.df.groupby(col)[self.target_col].agg(['sum', 'count'])
            grouped['woe'] = np.log((grouped['sum'] + 0.5) / (grouped['count'] - grouped['sum'] + 0.5))
            self.df[col] = self.df[col].map(grouped['woe'])
        print(f"✅ Applied WoE Encoding on: {cat_columns}")

    @log_action("🛠️ Merge Rare Categories")
    def _merge_rare_categories(self, threshold: float = 0.01) -> None:
        """ Merges rare categories into 'OTHER'. """
        cat_columns = get_categorical_columns(self.df)
        for col in cat_columns:
            self.df[col] = self.df[col].where(self.df[col].map(self.df[col].value_counts(normalize=True)) >= threshold, "OTHER")
        print(f"✅ Merged rare categories in: {cat_columns}")

    @log_action("🗂️ Clustering-Based Encoding (K-means)")
    def _merge_categories_clustering(self, n_clusters: int = 5) -> None:
        """ Groups similar categories using K-means clustering. """
        cat_columns = get_categorical_columns(self.df)
        for col in cat_columns:
            encoded = CountEncoder().fit_transform(self.df[[col]])
            labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(encoded)
            self.df[col] = self.df[col].map(dict(zip(self.df[col].unique(), [f"Cluster_{label}" for label in labels])))
        print(f"✅ Merged categories in: {cat_columns} using K-means.")

    def display_info(self) -> None:
        """Displays information about the DataFrame."""
        print(self.df.info())

class FeatureEncoder(DataEncoder):
    def __init__(self, df: pd.DataFrame, target_col: str) -> None:
        super().__init__(df, target_col)

    @log_action("🔄 One-Hot Encoding with optimization")
    def one_hot_encode_incremental(self, output_dir: str = "encoded_features", min_freq: float = 0.01) -> None:
        """
        Encodes categorical features using one-hot encoding, saving each encoded feature as sparse matrix (.npz).
        
        @param output_dir: The directory to save the encoded features
        @param min_freq: The minimum frequency threshold for rare categories
        """
        self._merge_rare_categories(min_freq)
        os.makedirs(output_dir, exist_ok=True)
        cat_columns = get_categorical_columns(self.df)

        for col in cat_columns:
            self.df[col] = self.df[col].where(self.df[col].map(self.df[col].value_counts(normalize=True)) >= min_freq, "OTHER")

            encoded = OneHotEncoder(sparse_output=True, handle_unknown='ignore').fit_transform(self.df[[col]])
            sparse.save_npz(os.path.join(output_dir, f"{col}_encoded_{time.strftime('%Y%m%d_%H%M%S')}.npz"), encoded)

            print(f"Encoded {col} and saved as {col}_encoded_{time.strftime('%Y%m%d_%H%M%S')}.npz")

    def encoding_selected(self, method: str, **kwargs) -> None:
        """
        Encodes categorical features using the specified method.
        
        @param method: The encoding method to apply
        @param kwargs: Additional keyword arguments for the encoding method
        """
        methods_of_the_function = {
            "one-hot": lambda: self._one_hot_encode,
            "count": lambda: self._count_encode,
            "hash": lambda: self._hash_encode,
            "target": lambda: self._target_encode,
            "mean": lambda: self._mean_encode,
            "frequency": lambda: self._frequency_encode,
            "leave-one-out": lambda: self._leave_one_out_encode,
            "woe": lambda: self._weight_of_evidence_encode,
            "clustering": lambda: self._merge_categories_clustering
        }

        if method not in methods_of_the_function:
            raise ValueError(f"Method {method} not found in FeatureEncoder.")
        methods_of_the_function[method](**kwargs)

    @log_action("🗂️ Merge Categories via Clustering (K-means)")
    def merge_categories_clustering(self, n_clusters: int = 5) -> None:
        """
        Merges similar categories using K-means clustering.
        
        @param n_clusters: The number of clusters for K-means clustering
        """
        cat_columns = get_categorical_columns(self.df)

        for col in cat_columns:
            encoded = CountEncoder().fit_transform(self.df[[col]])
            labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(encoded)

            self.df[col] = self.df[col].map(dict(zip(self.df[col].unique(), [f"Cluster_{label}" for label in labels])))
            print(f"Merged categories in column '{col}' using K-means clustering into {n_clusters} clusters.")