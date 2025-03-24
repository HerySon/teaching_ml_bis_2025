"""
Dimensionality reduction using PCA, SVD, t-SNE, or Isomap.

This script provides a class to perform dimensionality reduction using various methods such as PCA, SVD, t-SNE, and Isomap.

The class allows for scaling the data, optimizing the number of components, applying dimensionality reduction, and selecting the most influential features.

@author: Feurking
"""

# [Tache4.3][4.5points] Implémenter une méthode linéaire de réduction de variables

# python -m scripts --file_path "C:/Users/Quentin/Desktop/machine learninig/teaching_ml_bis_2025/data/en.openfoodfacts.org.products.csv" --target_col "price" --method "PCA"

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, Isomap

import matplotlib.pyplot as plt

from ..utilities.data_utils import load_data

class DimensionalityReduction(object):
    def __init__(self, data: pd.DataFrame, target: pd.Series, n_components: int = None, method: str = "PCA") -> None:
        """
        Initialize the class with data, number of components, and reduction method.
        
        @param data: DataFrame containing the features
        @param target: Series containing the target variable
        @param n_components: Number of principal components to retain (None to optimize)
        @param method: Dimensionality reduction method ("PCA", "SVD", "t-SNE", "Isomap")
        """
        if isinstance(method, list):
            raise ValueError("Method cannot be a list. Please provide a valid method name as a string.")
        
        self.data = self._load_and_validate_data(data)
        self.target = target
        self.n_components = n_components
        self.method = method

        self.scaler = StandardScaler()
        self.reducer = self._initialize_reducer()

    @staticmethod
    def _load_and_validate_data(data) -> pd.DataFrame:
        return load_data(data)
    
    def _pca_reducer(self, svd_solver : str = 'auto', whiten : bool = True, random_state : int = 42) -> PCA:
        """
        Initialize the PCA method with additional options for efficiency and flexibility.
        This method allows for sparse data handling and supports randomized SVD for faster computation.

        @param svd_solver: SVD solver to use ('auto', 'full', 'arpack', 'randomized')
        @param whiten: Whether to whiten the data
        @param random_state: Random seed for reproducibility
        """
        return PCA(n_components=self.n_components, svd_solver=svd_solver, whiten=whiten, random_state=random_state)
    
    def _truncate_svd_reducer(self, random_state : int = 42) -> TruncatedSVD:
        """
        Initialize the Truncated SVD method.
        
        @param random_state: Random seed for reproducibility
        """
        return TruncatedSVD(n_components=self.n_components, random_state=random_state)
    
    def _tsne_reducer(self, random_state : int = 42) -> TSNE:
        """
        Initialize the t-SNE method.
        
        @param random_state: Random seed for reproducibility
        """
        return TSNE(n_components=self.n_components, random_state=random_state)
    
    def _isomap_reducer(self, n_neighbors : int = 5, random_state : int = 42) -> Isomap:
        """
        Initialize the Isomap method.
        
        @param n_neighbors: Number of neighbors to consider for each point
        @param random_state: Random seed for reproducibility
        """
        return Isomap(n_components=self.n_components, n_neighbors=n_neighbors, random_state=random_state)

    def _initialize_reducer(self) -> object:
        """
        Initialize the dimensionality reduction method.
        
        - Return -> object:
            The initialized reducer object
        """
        methods_of_the_function = {
            "PCA": lambda: self._pca_reducer(),
            "SVD": lambda: self._truncate_svd_reducer(),
            "t-SNE": lambda: self._tsne_reducer(),
            "Isomap": lambda: self._isomap_reducer()
        }

        if self.method in methods_of_the_function:
            return methods_of_the_function[self.method]()
        else:
            raise ValueError("Invalid method. Choose from 'PCA', 'SVD', 't-SNE', 'Isomap'.")

    def scale_data(self) -> None:
        """Standardize the data for dimensionality reduction."""
        self.scaled_data = self.scaler.fit_transform(self.data)
        print("Data normalized.")

    def optimize_n_components(self, threshold: float = 0.95) -> None:
        """
        Optimize the number of components to retain based on explained variance or reconstruction error.
        This method selects the number of components that explain at least `threshold` of the data variance (PCA)
        or minimizes reconstruction error (other methods).
        
        @param threshold: Percentage of explained variance to achieve (between 0 and 1, applicable for PCA)
        """
        if self.method == "PCA":
            explained_variance_ratio = np.cumsum(self.reducer.fit(self.scaled_data).explained_variance_ratio_)
            self.n_components = np.searchsorted(explained_variance_ratio, threshold) + 1
            print(f"Optimal number of components to explain {threshold * 100}% of the variance: {self.n_components}")
            
        elif hasattr(self.reducer, 'inverse_transform'):
            errors = []

            for n in range(1, min(self.scaled_data.shape) + 1):
                temp_reducer = self._initialize_reducer()

                temp_reducer.n_components = n
                reduced_data = temp_reducer.fit_transform(self.scaled_data)
                reconstructed_data = temp_reducer.inverse_transform(reduced_data)
                mse = np.mean(np.square(self.scaled_data - reconstructed_data))

                errors.append(mse)
                
            self.n_components = np.argmin(errors) + 1
            print(f"Optimal number of components based on reconstruction error: {self.n_components}")
        else:
            print("Optimization of the number of components is not supported for the selected method.")

    def apply_reduction(self) -> None:
        """
        Apply dimensionality reduction with the optimal number of components (or user-defined).
        """
        if not hasattr(self, 'scaled_data'):
            self.scale_data()
        self.reduced_data = self.reducer.fit_transform(self.scaled_data)
        print(f"Data reduced to {self.n_components or self.reducer.n_components} dimensions using {self.method}.")

    def explained_variance(self) -> None:
        """
        Display the explained variance for each principal component (PCA only).
        """
        if self.method != "PCA":
            print("Explained variance is not available for this method.")
            return

        variance_explained = self.reducer.explained_variance_ratio_
        print(f"Explained variance by each component:\n{variance_explained}")
        self._plot_cumulative_variance(variance_explained)

    @staticmethod
    def _plot_cumulative_variance(variance_explained) -> None:
        """Plot the cumulative explained variance."""
        plt.figure(figsize=(8, 6))
        plt.plot(np.cumsum(variance_explained), marker='o', linestyle='--')
        plt.title("Cumulative Explained Variance by Components")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.show()

    def transform_data(self) -> pd.DataFrame:
        """Return the transformed data after applying dimensionality reduction."""
        return pd.DataFrame(self.reduced_data)

    def select_features(self, top_n: int = 3) -> list:
        """
        Select the most influential features based on the principal component coefficients.
        
        @param top_n: Number of most influential features to retain per component
        """
        if not hasattr(self.reducer, 'components_'): return []

        components_df = pd.DataFrame(self.reducer.components_, columns=self.data.columns)
        return [
            component.abs().nlargest(top_n).index.tolist()
            for _, component in components_df.iterrows()
        ]

    @staticmethod
    def _get_top_features(components_df, top_n) -> list:
        """Get the top N features for each component."""
        top_features = []

        for _, component in components_df.iterrows():
            top_features_for_component = component.abs().nlargest(top_n).index.tolist()
            top_features.append(top_features_for_component)
            
        return top_features
    
    def optimize_and_reduce(self, threshold: float = 0.95, top_n: int = 3) -> list:
        """
        Combine optimization of the number of components, dimensionality reduction, and feature selection.
        
        @param threshold: Percentage of explained variance to achieve (between 0 and 1, applicable for PCA)
        @param top_n: Number of most influential features to retain per component
        """
        self.scale_data()

        if self.method == "PCA":
            self.optimize_n_components(threshold)
            self.reducer.n_components = self.n_components

        self.reduced_data = self.reducer.fit_transform(self.scaled_data)
        print(f"Data reduced to {self.n_components or self.reducer.n_components} dimensions using {self.method}.")

        if hasattr(self.reducer, 'components_'):
            components_df = pd.DataFrame(self.reducer.components_, columns=self.data.columns)
            selected_features = self._get_top_features(components_df, top_n)
            print(f"Selected features based on top {top_n} influential features per component: {selected_features}")
            return selected_features
        else: return None

    def reconstruct_data(self) -> pd.DataFrame:
        """
        Reconstruct the original data using the reduced dimensions.
        This is useful for evaluating the quality of dimensionality reduction.
        """
        if hasattr(self.reducer, 'inverse_transform'):
            reconstructed_data = self.reducer.inverse_transform(self.reduced_data)
            return pd.DataFrame(reconstructed_data, columns=self.data.columns)
        else: return None

    def evaluate_reconstruction_error(self) -> float:
        """
        Evaluate the reconstruction error after dimensionality reduction.
        This is useful for assessing the quality of the reduction.
        """
        if hasattr(self.reducer, 'inverse_transform') and self.reduced_data is not None:
            reconstructed_data = self.reducer.inverse_transform(self.reduced_data)
            mse = np.mean(np.square(self.data - reconstructed_data))
            print(f"Mean Squared Reconstruction Error: {mse}")

            return mse
        
        print("Reconstruction error cannot be computed for the selected method.")

        return None