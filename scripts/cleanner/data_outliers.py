"""
Module for detecting and handling outliers in a dataset.

Classes
    DataOutlier: Class for detecting and handling outliers.
    OutlierDetection: Subclass of DataOutlier with additional outlier detection methods.

@author: Feurking
"""

# [Tache3.0][4.5pts] Gérer les valeurs abbérantes

# python -m scripts --file_path "C:/Users/Quentin/Desktop/machine learninig/teaching_ml_bis_2025/data/en.openfoodfacts.org.products.csv" --output_path "C:/Users/Quentin/Desktop/Fuck YourSlef PEP8/teaching_ml_bis_2025/results/test.csv" --method remove

import os

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import zscore
from sklearn.ensemble import IsolationForest

from ..utilities.data_utils import load_data, get_numeric_columns

class DataOutlier(object):
    def __init__(self, df: pd.DataFrame, output_dir: str = "outlier_analysis") -> None:
        """
        Class for detecting and handling outliers.

        @param df: DataFrame containing the data
        @param output_dir: Directory to save visualizations
        """
        self.df = load_data(df)
        self._output_dir = output_dir

        os.makedirs(self._output_dir, exist_ok=True)

    @property
    def output_dir(self) -> str:
        """Getter for output_dir."""
        return self._output_dir

    @output_dir.setter
    def output_dir(self, new_dir: str) -> None:
        """Setter for output_dir."""
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        self._output_dir = new_dir

    @staticmethod
    def is_numeric_column(df: pd.DataFrame, col: str) -> bool:
        """
        Static method to check if a column is numeric.

        @param df: DataFrame containing the data
        @param col: Column name
        @return: True if the column is numeric, False otherwise
        """
        return pd.api.types.is_numeric_dtype(df[col])
    
    def _remove_outliers(self, col: str) -> None:
        """
        Remove outliers from the DataFrame.

        @param col: Column name
        """
        self.df = self.df[self.df[f'outlier_tukey_{col}'] == 0]

    def _impute_mean_outliers(self, col: str) -> None:
        """
        Impute outliers in the DataFrame using the mean.

        @param col: Column name
        """
        mean_value = self.df[col].mean()
        self.df.loc[self.df[f'outlier_tukey_{col}'] == 1, col] = mean_value

    def _impute_median_outliers(self, col: str) -> None:
        """
        Impute outliers in the DataFrame using the median.

        @param col: Column name
        """
        median_value = self.df[col].median()
        self.df.loc[self.df[f'outlier_tukey_{col}'] == 1, col] = median_value

    def _impute_value_outliers(self, col: str, value: float) -> None:
        """
        Impute outliers in the DataFrame using a specific value.

        @param col: Column name
        @param value: Value to use for imputation
        """
        self.df.loc[self.df[f'outlier_tukey_{col}'] == 1, col] = value

    def _impute_outliers(self, col: str, method: str, value: float = None) -> None:
        """
        Impute outliers in the DataFrame.

        @param col: Column name
        @param method: Imputation method ('mean', 'median', 'value')
        @param value: Value to use for 'value' method
        """
        methods_of_the_function = {
            "mean": lambda: self._impute_mean_outliers,
            "median": lambda: self._impute_median_outliers,
            "value": lambda: self._impute_value_outliers
        }

        try :
            methods_of_the_function[method](col, value)
        except KeyError:
            raise ValueError(f"Unknown imputation method: {method}")

    def detect_outliers_tukey(self, col: str) -> None:
        """
        Detect outliers using Tukey's criterion (IQR).

        @param col: Column name
        @return: DataFrame with a new column 'outlier_tukey'
        """
        q1, q3 = self.df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        self.df[f'outlier_tukey_{col}'] = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).astype(int)

    def detect_outliers_zscore(self, col: str, threshold: float = 3.0) -> None:
        """
        Detect outliers using the Z-score.

        @param col: Column name
        @param threshold: Threshold to consider a value as an outlier
        @return: DataFrame with a new column 'outlier_zscore'
        """
        self.df[f'outlier_zscore_{col}'] = (np.abs(zscore(self.df[col])) > threshold).astype(int)

class OutlierDetection(DataOutlier):
    __init__ = DataOutlier.__init__

    def detect_outliers_isolation_forest(self, col: str) -> None:
        """
        Detect outliers using the Isolation Forest algorithm.

        @param col: Column name
        @return: DataFrame with a new column 'outlier_iforest'
        """
        model = IsolationForest(contamination=0.05, random_state=42)

        self.df[f'outlier_iforest_{col}'] = model.fit_predict(self.df[[col]])
        self.df[f'outlier_iforest_{col}'] = (self.df[f'outlier_iforest_{col}'] == -1).astype(int)

    def summarize_outliers(self, col: str) -> None:
        """
        Summarize the number of outliers detected by each method.

        @param col: Column name
        """
        summary = {
            "Tukey": self.df[f'outlier_tukey_{col}'].sum(),
            "Z-Score": self.df[f'outlier_zscore_{col}'].sum(),
            "Isolation Forest": self.df[f'outlier_iforest_{col}'].sum(),
        }

        print(f"Outlier summary for column '{col}': {summary}")

    def save_cleaned_data(self, output_file: str = "cleaned_data.csv") -> None:
        """
        Save the cleaned DataFrame to a file.

        @param output_file: Name of the output file
        """
        output_path = os.path.join(self._output_dir, output_file)
        self.df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")

    def plot_outliers(self, col: str) -> None:
        """
        Display a boxplot to visualize outliers.

        @param col: Column name
        """
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=self.df[col], color='lightblue')
        plt.title(f'Boxplot for {col}')

        safe_col_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in col)
        
        plt.savefig(os.path.join(self._output_dir, f"{safe_col_name}_boxplot.png"))
        plt.close()

    def handle_outliers(self, col: str, strategy: str = "remove", value: float = None) -> None:
        """
        Handle outliers based on the chosen strategy.

        @param col: Column name
        @param strategy: Strategy to apply ('remove', 'impute_mean', 'impute_median', 'impute_value')
        @param value: Value to use for 'impute_value' strategy
        """
        methods_of_the_function = {
            "remove": self._remove_outliers,
            "impute_mean": lambda col: self._impute_outliers(col, "mean"),
            "impute_median": lambda col: self._impute_outliers(col, "median"),
            "impute_value": lambda col: self._impute_outliers(col, "value", value)
        }

        if strategy in methods_of_the_function:
            methods_of_the_function[strategy](col)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def run_outlier_analysis(self, strategy: str = "remove", value: float = 2, numerical_cols: list = None) -> None:
        """
        Perform outlier analysis on all numerical columns.

        @param numerical_cols: List of numerical columns
        @param strategy: Strategy to apply ('remove', 'impute_mean', 'impute_median', 'impute_value')
        @param value: Value to use for 'impute_value' strategy
        """
        print("Starting outlier analysis...\n")

        numerical_cols = numerical_cols or get_numeric_columns(self.df)
        
        for col in filter(lambda c: self.is_numeric_column(self.df, c), numerical_cols):
            print(f"Analyzing column: {col}")
            
            for method in [self.detect_outliers_tukey, self.detect_outliers_zscore, self.detect_outliers_isolation_forest]:
                method(col)

            self.summarize_outliers(col)
            # self.plot_outliers(col)
            self.handle_outliers(col, strategy, value)
            
        self.save_cleaned_data()
        print("\nAnalysis complete. Results saved in:", self._output_dir)