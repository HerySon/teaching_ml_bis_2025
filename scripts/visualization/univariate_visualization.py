"""
Module for performing univariate visualizations on a dataset.

Classes
    Plotting: Base class for plotting univariate visualizations.
    AdvancedVisualization: Class for performing advanced visualizations on a dataset.

@author: Feurking
"""

# [Tache4.0][4.5pts] Faire des visualisations de données univariées

# python -m scripts --file_path "C:/Users/Quentin/Desktop/machine learninig/teaching_ml_bis_2025/data/en.openfoodfacts.org.products.csv" --output_path "C:/Users/Quentin/Desktop/Fuck YourSlef PEP8/teaching_ml_bis_2025/results/test.csv"

import os
import re
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, normaltest

from ..utilities.data_utils import load_data, log_action

class Plotting(object):
    def __init__(self, plots, axes, data, col, plot_type) -> None:
        self.plots = plots
        self.axes = axes
        self.data = data
        self.col = col
        self.plot_type = plot_type

    @log_action("📊 Histogram Plot: generated")
    def _histogram_ploting(self, data, col, axes) -> None:
        sns.histplot(data, kde=True, bins=self.bins, color='skyblue', stat='density', ax=axes)
        axes.set_title(f'Histogram & KDE for {col}', fontsize=14)
        axes.set_xlabel(col, fontsize=12)
        axes.set_ylabel('Density', fontsize=12)

    @log_action("📊 Box Plot: generated")
    def _boxplot_ploting(self, data, col, axes) -> None:
        sns.boxplot(x=data, color='lightgreen', ax=axes)
        axes.set_title(f'Boxplot for {col}', fontsize=14)
        axes.set_xlabel(col, fontsize=12)

    def _outliers(self, data, col, axes) -> None:
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = data[(data < lower_bound) | (data > upper_bound)]
        sns.scatterplot(x=data.index, y=data, ax=axes, label='Data')

        sns.scatterplot(x=outliers.index, y=outliers, color='red', ax=axes, label='Outliers')

        axes.set_title(f'Outlier Detection for {col}', fontsize=14)
        axes.set_xlabel('Index', fontsize=12)
        axes.set_ylabel(col, fontsize=12)

    @log_action("📊 Count Plot: generated")
    def _countplot(self, data, col, cleaned_col_name) -> None:
        plt.figure(figsize=(12, 6))
        sns.countplot(data=self.df, x=col, palette='Set2', order=data.value_counts().index)
        plt.title(f'Frequency plot for {col}', fontsize=16)
        plt.xlabel(col, fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{cleaned_col_name}_countplot.png"))
        plt.close()

    def __call__(self, method: str, **kwargs) -> None:
        if callable(method):
            method(data=kwargs.get('data'), col=kwargs.get('col'), axes=kwargs.get('axes'))
        else:
            raise ValueError(f"Invalid method reference provided: {method}")

class AdvancedVisualization(Plotting):
    """
    Class for performing advanced visualizations on a dataset,
    including univariate and bivariate analyses.
    """

    def __init__(self, df: pd.DataFrame, output_dir: str = "visualizations", bins: int = 30) -> None:
        """
        Initializes the class with a DataFrame and prepares visualizations.

        @param df: The DataFrame containing the data to analyze.
        @param output_dir: The directory where visualizations will be saved.
        @param bins: The number of bins for histograms.
        """
        self.df = load_data(df)
        self.output_dir = output_dir
        self.bins = bins

        os.makedirs(self.output_dir, exist_ok=True)

    def __call_methods__(self, method: str, **kwargs) -> None:
        """
        Call specified methods with keyword arguments.

        @param methods: List of methods to call
        @param kwargs: Additional keyword arguments for each method
        """
        if callable(method):
            method(data=kwargs.get('data'), col=kwargs.get('col'), axes=kwargs.get('axes'))
        else:
            raise ValueError(f"Invalid method reference provided: {method}")

    @staticmethod
    def clean_column_name(col_name: str) -> str:
        """
        Cleans the column name to make it compatible with the file system.

        @param col_name: The column name to clean.
        @return: The cleaned file name.
        """
        cleaned_name = re.sub(r'[\\/*?:"<>|]', "_", col_name)
        return cleaned_name.replace(" ", "_")

    def check_missing_data(self) -> None:
        """Checks and displays missing values in the DataFrame."""
        missing_data = self.df.isnull().sum()
        if missing_data.any():
            print("Missing values found:")
            print(missing_data[missing_data > 0])
        else:
            print("No missing values detected.")

    def plot_numerical_features(self) -> None:
        """
        Generates visualizations for numerical columns:
        histograms, boxplots, and descriptive statistics.
        """
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            cleaned_col_name = self.clean_column_name(col)

            _, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            sns.histplot(self.df[col], kde=True, bins=self.bins, color='skyblue', stat='density', ax=axes[0])

            axes[0].set_title(f'Histogramme et KDE pour {col}', fontsize=14)
            axes[0].set_xlabel(col, fontsize=12)
            axes[0].set_ylabel('Densité', fontsize=12)

            if self.df[col].nunique() > 1:
                sns.boxplot(x=self.df[col].dropna(), color='lightgreen', ax=axes[1])

                axes[1].set_title(f'Boxplot pour {col}', fontsize=14)
                axes[1].set_xlabel(col, fontsize=12)
            else:
                print(f"Colonne {col} ignorée pour boxplot (trop peu de variations).")

            axes[1].set_title(f'Boxplot for {col}', fontsize=14)
            axes[1].set_xlabel(col, fontsize=12)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{cleaned_col_name}_numerical_plots.png"))
            plt.close()

    def analyze_numerical_feature(self, col: str, plots: list = None) -> None:
        """
        Analyzes a single numerical feature with customizable visualizations and statistics.

        @param col: The column name to analyze.
        @param plots: List of plots to generate. Options: ['histogram', 'boxplot', 'outliers'].
        """
        methods_of_the_function = {
            "histogram": lambda: self._histogram_ploting,
            "boxplot": lambda: self._boxplot_ploting,
            "outliers": lambda: self._outliers
        }

        if plots is None:
            plots = ['histogram', 'boxplot', 'outliers']

        cleaned_col_name = self.clean_column_name(col)
        data = self.df[col].dropna()

        mean = data.mean()
        median = data.median()
        std_dev = data.std()

        skewness = skew(data)
        kurt = kurtosis(data)
        normality_pval = normaltest(data).pvalue

        print(f"\nAnalyzing numerical feature: {col}")

        if isinstance(mean, pd.Series):
            mean, median, std_dev = mean.iloc[0], median.iloc[0], std_dev.iloc[0]

        print(f"Mean: {mean:.2f}, Median: {median:.2f}, Std Dev: {std_dev:.2f}")
        print(f"Skewness: {skewness:.2f}, Kurtosis: {kurt:.2f}")
        print(f"Normality test p-value: {normality_pval:.4f}")

        num_plots = len(plots)
        _, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))

        if num_plots == 1:
            axes = [axes]

        for i, plot_type in enumerate(plots):
            self.__call_methods__(method=methods_of_the_function[plot_type], data=data, col=col, axes=axes[i])

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{cleaned_col_name}_custom_analysis.png"))
        plt.close()

    def analyze_categorical_feature(self, col: str, plots: list = None) -> None:
        """
        Analyzes a single categorical feature with customizable visualizations.

        @param col: The column name to analyze.
        @param plots: List of plots to generate. Options: ['countplot'].
        """
        if plots is None: plots = ['countplot']

        cleaned_col_name = self.clean_column_name(col)
        data = self.df[col].dropna()

        print(f"\nAnalyzing categorical feature: {col}")
        print(f"Unique values: {data.nunique()}")
        print(f"Value counts:\n{data.value_counts()}")

        for _ in plots: self._countplot(data, col, cleaned_col_name)

    def run_univariate_analysis(self, numerical_plots: list = None, categorical_plots: list = None) -> None:
        """
        Executes univariate analysis for all features in the dataset with customizable plots.

        @param numerical_plots: List of plots for numerical features. Options: ['histogram', 'boxplot', 'outliers'].
        @param categorical_plots: List of plots for categorical features. Options: ['countplot'].
        """
        print("Starting univariate analysis...\n")

        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            self.analyze_numerical_feature(col, plots=numerical_plots)

        categorical_cols = self.df.select_dtypes(include=[object]).columns
        for col in categorical_cols:
            self.analyze_categorical_feature(col, plots=categorical_plots)

        print("\nUnivariate analysis complete. Visualizations have been saved in the directory:")
        print(f"{self.output_dir}")