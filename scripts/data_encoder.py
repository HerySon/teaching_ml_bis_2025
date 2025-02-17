"""
This module contains a class for encoding non-numeric features.

Primary Functions & Classes:
    FeatureEncoder: Class to encode categorical features efficiently.

@author: Feurking
"""

import pandas as pd
import numpy as np

import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from category_encoders import HashingEncoder, CountEncoder

from scipy import sparse
from datetime import datetime
from functools import wraps

import re
import time

import pytest

import matplotlib.pyplot as plt

from .utils.data_utils import load_data, log_action, get_categorical_columns
import seaborn as sns

class FeatureEncoder(object):

    @pytest.mark.parametrize("df", [ ("/data/en.openfoodfacts.org.products.csv") ])
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self.log = []

    @classmethod
    def from_csv(cls, file_path: str, limit: int) -> 'FeatureEncoder':
        df = load_data(file_path, limit)
        return cls(df)

    @log_action("🔄 One-Hot Encoding with optimization")
    def one_hot_encode_incremental(self, output_dir: str = "encoded_features", min_freq: float = 0.01) -> None:
        """Encodes categorical features using one-hot encoding, saving each encoded feature as sparse matrix (.npz)."""
        self.merge_rare_categories(min_freq)
        os.makedirs(output_dir, exist_ok=True)
        cat_columns = get_categorical_columns(self.df)

        for col in cat_columns:
            self.df[col] = self.df[col].where(self.df[col].map(self.df[col].value_counts(normalize=True)) >= min_freq, "OTHER")

            encoded = OneHotEncoder(sparse_output=True, handle_unknown='ignore').fit_transform(self.df[[col]])
            sparse.save_npz(os.path.join(output_dir, f"{col}_encoded_{time.strftime('%Y%m%d_%H%M%S')}.npz"), encoded)

            print(f"Encoded {col} and saved as {col}_encoded_{time.strftime('%Y%m%d_%H%M%S')}.npz")

    @log_action("🔍 Count Encoding")
    def count_encode(self, min_freq: float = 0.01) -> None:
        """Encodes categorical features using Count Encoding."""
        self.merge_rare_categories(min_freq)

        cat_columns = get_categorical_columns(self.df)

        self.df[cat_columns] = CountEncoder().fit_transform(self.df[cat_columns])
        print(f"Applied Count Encoding on columns: {cat_columns}")

    @log_action("⚙️ Hash Encoding")
    def hash_encode(self, n_components: int = 8) -> None:
        """Encodes categorical features using Hashing Encoding."""
        self.merge_rare_categories()

        cat_columns = get_categorical_columns(self.df)

        self.df[cat_columns] = HashingEncoder(n_components=n_components).fit_transform(self.df[cat_columns])
        print(f"Applied Hash Encoding on columns: {cat_columns}")

    @log_action("🛠️ Merge Rare Categories")
    def merge_rare_categories(self, threshold: float = 0.01) -> None:
        """Merges categories that appear in less than a specified threshold (e.g., 1%) into a single 'OTHER' category."""
        cat_columns = get_categorical_columns(self.df)

        for col in cat_columns:
            self.df[col] = self.df[col].where(self.df[col].map(self.df[col].value_counts(normalize=True)) >= threshold, "OTHER")
            print(f"Merged rare categories in column '{col}' with threshold {threshold}")

    @log_action("🗂️ Merge Categories via Clustering (K-means)")
    def merge_categories_clustering(self, n_clusters: int = 5) -> None:
        """Merges similar categories using K-means clustering."""
        cat_columns = get_categorical_columns(self.df)

        for col in cat_columns:
            encoded = CountEncoder().fit_transform(self.df[[col]])
            labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(encoded)

            self.df[col] = self.df[col].map(dict(zip(self.df[col].unique(), [f"Cluster_{label}" for label in labels])))
            print(f"Merged categories in column '{col}' using K-means clustering into {n_clusters} clusters.")

            self.plot_clusters(encoded, labels, col)

    def plot_clusters(self, encoded, labels, feature_column) -> None:
        """Plots clusters after K-means.
        
            @param encoded: Encoded DataFrame
            @param labels: Cluster labels
            @param feature_column: Name of the feature column
        """
        n_components = min(2, encoded.shape[1])
        pca_result = PCA(n_components=n_components).fit_transform(sparse.csr_matrix(encoded).toarray())
        
        columns = [f'PCA{i+1}' for i in range(n_components)]
        cluster_df = pd.DataFrame(pca_result, columns=columns)
        cluster_df['Cluster'] = labels

        sns.scatterplot(
            x=columns[0], y=columns[1] if n_components == 2 else columns[0],
            hue='Cluster', data=cluster_df, palette="Set1", s=100, alpha=0.6
        )

        plt.title(f'Clusters for {feature_column}')
        plt.show()

    @log_action("📊 Save Encoded Data")
    def save_encoded_df(self, output_path: str = "encoded_df.csv") -> None:
        """Saves the encoded DataFrame to the specified file."""
        self.df.to_csv(output_path, index=False)
        print(f"Encoded DataFrame saved to {output_path}")

    def summarize(self) -> dict:
        """Summarizes the DataFrame, including shape, categorical columns, and log."""
        return {
            'shape': self.df.shape,
            'categorical_columns': get_categorical_columns(self.df),
            'log': self.log
        }

    def display_info(self) -> None:
        """Displays information about the DataFrame."""
        print(self.df.info())