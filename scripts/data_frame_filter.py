"""
This module contains the DataFrameSelector class for filtering columns in a DataFrame.

Primary Functions & Classes:
    DataFrameSelector: Class to filter columns in a DataFrame.

@author: Feurking
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.animation import FuncAnimation

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import os
import random
import string
import pytest

from .utils.data_utils import load_data, get_numeric_columns, log_action

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')
warnings.filterwarnings("ignore", category=UserWarning, module='tkinter')

class DataFrameSelector (object):
    @pytest.mark.parametrize("limit, path", [ (1000, "/data/en.openfoodfacts.org.products.csv") ])
    def __init__(self, limit: int, path : str) -> None:
        """Initializes the DataFrameSelector with a DataFrame."""
        self.df = load_data(path, limit)

        self.numeric_columns = get_numeric_columns(self.df)
        self.ordinal_columns = self._get_ordinal_columns()
        self.nominal_columns = self._get_nominal_columns()
        
    def _get_ordinal_columns(self) -> list:
        """Get ordinal (categorical) columns in the DataFrame."""
        ordinal_columns = [
            col for col in self.df.select_dtypes(include=['object']).columns
            if self.df[col].nunique() <= 10
        ]

        return ordinal_columns

    def _get_nominal_columns(self) -> list:
        """Get nominal (categorical) columns in the DataFrame."""
        nominal_columns = [
            col for col in self.df.select_dtypes(include=['object']).columns
            if self.df[col].nunique() > 10 
        ]

        return nominal_columns

    def filter_columns_by_correlation(self, min_corr: float = 0.1) -> None:
        """Filter numeric columns based on correlation threshold.
        
            @param min_corr: Minimum correlation threshold
        """
        corr_matrix = self.df.corr(numeric_only=True)

        relevant_numeric_columns = [col for col in corr_matrix.columns if any(abs(corr_matrix[col]) >= min_corr)]

        self.numeric_columns = relevant_numeric_columns

    def filter_columns_by_category_count(self, min_categories: int = 5) -> None:
        """Filter categorical columns by the number of unique categories."""
        self.ordinal_columns = [col for col in self.ordinal_columns if self.df[col].nunique() >= min_categories]
        self.nominal_columns = [col for col in self.nominal_columns if self.df[col].nunique() >= min_categories]

    @log_action("🔍 Get Relevant Subset")
    def get_relevant_subset(self) -> pd.DataFrame:
        """Return a relevant subset of the DataFrame with filtered columns."""
        filtered_df = self.df[self.numeric_columns + self.ordinal_columns + self.nominal_columns]
        return filtered_df
    
    @log_action("📊 Get Representative Sample")
    @pytest.mark.parametrize("sample_size, random_state, stratify_by", [ (0.1, 42, None) ])
    def get_representative_sample(self, sample_size: float = 0.1, random_state: int = 42, stratify_by: str = None) -> pd.DataFrame:
        """Get a representative sample of the DataFrame with imputed missing values.

            @param sample_size: Fraction of the DataFrame to sample
            @param random_state: Random seed for reproducibility
            @param stratify_by: Column to stratify the sample by
        """
        if stratify_by:
            _, sample_df = train_test_split(self.df, test_size=sample_size, random_state=random_state, stratify=self.df[stratify_by])
        else:
            sample_df = self.df.sample(frac=sample_size, random_state=random_state)

        missing_data = sample_df.isnull().sum()

        threshold = 0.9
        sample_df = sample_df.loc[:, missing_data / len(sample_df) < threshold]

        imputer = SimpleImputer(strategy='most_frequent' if sample_df.select_dtypes(include=['object']).shape[1] > 0 else 'mean')
        
        sample_df_imputed = pd.DataFrame(imputer.fit_transform(sample_df), columns=sample_df.columns)
        sample_df_imputed = sample_df_imputed.reset_index(drop=True)

        return sample_df_imputed

    def generate_random_name(self) -> str:
        """Generate a random name for a file."""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))

    @log_action("📊 Save Plot")
    @pytest.mark.parametrize("col, plot_type, sample_df, filename", [
            ('code', 'numeric', None, 'code_numeric'),
            ('code', 'categorical', None, 'code_categorical'),
            ('energy_100g', 'numeric', None, 'energy_100g_numeric'),
            ('energy_100g', 'categorical', None, 'energy_100g_categorical')
    ])
    def save_plot(self, col, plot_type: str, sample_df: pd.DataFrame = None, filename: str = 'plot') -> None:
        """
        Save plot as a PNG file.
        
        @param col: Column to plot
        @param plot_type: Type of plot (numeric or categorical)
        @param sample_df: Sample DataFrame to plot
        @param filename: Name of the file by default 'plot'
        """

        os.makedirs('results/plot/', exist_ok=True)
        filename = f'results/plot/{filename}{self.generate_random_name()}.png'

        self.df.columns = self.df.columns.str.strip()

        fig, ax = plt.subplots(figsize=(12, 8))

        if self.df[col].empty: return

        if plot_type == 'numeric':
            sns.histplot(self.df[col].dropna(), color='blue', kde=True, stat="density", ax=ax, label='Original')

            if sample_df is not None: sns.histplot(sample_df[col].dropna(), color='red', kde=True, stat="density", ax=ax, label='Sample')

            print(self.df[col].describe(), self.df[col].head(10), self.df[col].isna().sum())

            ax.set_title(f'Distribution of Numeric Column: {col}')
            ax.set_ylabel('Density')
            ax.set_xlabel(col)

        elif plot_type == 'categorical':
            top_10_categories = self.df[col].value_counts().head(10).index
            filtered_df = self.df[self.df[col].isin(top_10_categories)]
            
            sns.countplot(data=filtered_df, x=col, color='blue', ax=ax, label='Original')

            if sample_df is not None:
                filtered_sample_df = sample_df[sample_df[col].isin(top_10_categories)]
                sns.countplot(data=filtered_sample_df, x=col, color='red', ax=ax, label='Sample')

            ax.set_title(f'Distribution of Categorical Column: {col}')
            ax.set_ylabel('Count')
            ax.set_xlabel(col)

            plt.xticks(rotation=45)

        ax.legend(loc='upper right', fontsize='small')
        sns.despine()

        plt.savefig(filename)
        plt.close(fig)