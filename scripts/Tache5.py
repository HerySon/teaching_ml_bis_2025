import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class OpenFoodFactsAnalysis:
    """
    A class to apply various data analysis techniques like PCA and K-Means
    clustering on the Open Food Facts dataset, including visualization of results.
    """

    def __init__(self, dataset_path, nrows=100000):
        """
        Initializes the class by loading the Open Food Facts dataset.

        Arguments:
        - dataset_path (str): Path to the CSV file containing the dataset.
        - nrows (int): Number of rows to load (default is 100,000 for performance optimization).
        """
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)

        # Load the dataset
        self.df = pd.read_csv(dataset_path, sep="\t", on_bad_lines='skip', nrows=nrows, low_memory=False)
        self.processed_data = None

    def clean_data(self):
        """
        Cleans the DataFrame by removing irrelevant columns, duplicates, and columns
        with too many missing values.
        """
        self.df = self.remove_irrelevant_columns()
        self.df = self.remove_duplicates()
        self.df = self.remove_high_nan_columns()
        self.processed_data = self.pre_process_data()

    def remove_irrelevant_columns(self):
        """
        Removes columns that are not relevant for analysis.
        """
        irrelevant_columns = [
            "code", "url", "creator", "created_t", "created_datetime", "last_modified_t",
            "last_modified_datetime", "packaging", "packaging_tags", "brands_tags", "categories_tags",
            "origins_tags", "manufacturing_places", "labels_tags", "image_ingredients_url", 
            "image_nutrition_url", "image_url", "last_updated_t", "last_updated_datetime"
        ]
        self.df.drop(columns=[col for col in irrelevant_columns if col in self.df.columns], inplace=True)
        return self.df

    def remove_duplicates(self):
        """Removes duplicates from the DataFrame."""
        self.df.drop_duplicates(keep="first", inplace=True)
        return self.df

    def remove_high_nan_columns(self, threshold=70):
        """
        Removes columns with a high percentage of missing values.

        Arguments:
        - threshold (float): Maximum allowed percentage of missing values (default is 70%).
        """
        nan_value_ratio = self.df.isna().mean() * 100
        cols_to_remove = nan_value_ratio[nan_value_ratio > threshold].index
        self.df.drop(columns=cols_to_remove, inplace=True)
        return self.df

    def pre_process_data(self):
        """
        Selects relevant columns, removes missing values, normalizes, and prepares
        the data for analysis.
        """
        relevant_columns = [col for col in self.df.columns if "_100g" in col]
        data = self.df[relevant_columns].dropna()

        # Remove highly correlated columns
        corr_matrix = data.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        column_to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.8)]
        data.drop(columns=column_to_drop, inplace=True)

        return data

    def standardize_data(self, data):
        """Normalizes the data using StandardScaler."""
        scaler = StandardScaler()
        return scaler.fit_transform(data)

    def kmeans_clustering(self, n_clusters=3):
        """
        Applies K-Means clustering on the normalized data and returns the results.
        """
        scaled_data = self.standardize_data(self.processed_data)

        # Apply KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=25)
        self.processed_data['cluster'] = kmeans.fit_predict(scaled_data)

        print(f"K-Means inertia: {kmeans.inertia_}")
        print(f"Points distribution by cluster: \n{self.processed_data['cluster'].value_counts()}")
        
        return self.processed_data

    def find_optimal_clusters(self, max_clusters=10):
        """
        Uses the elbow method to determine the optimal number of clusters.
        """
        inertias = []
        scaled_data = self.standardize_data(self.processed_data)

        for n in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=n, random_state=25)
            kmeans.fit(scaled_data)
            inertias.append(kmeans.inertia_)

        plt.plot(range(1, max_clusters + 1), inertias, marker='o')
        plt.title("Elbow Method")
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia")
        plt.show()

    def pca_visualization(self):
        """
        Reduces the data to 2 dimensions using PCA and visualizes the clusters.
        """
        # Apply PCA to reduce to 2 dimensions
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.standardize_data(self.processed_data))

        # Add PCA results to the data
        self.processed_data['pca1'], self.processed_data['pca2'] = pca_result[:, 0], pca_result[:, 1]

        # Create a scatter plot of the clusters
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=self.processed_data, x='pca1', y='pca2', hue='cluster', palette="Set1", alpha=0.7)
        plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
        plt.title("Clustering after dimensionality reduction (PCA)")
        plt.show()


# Example usage
dataset_path = "../data/en.openfoodfacts.org.products.csv"
analysis = OpenFoodFactsAnalysis(dataset_path)
analysis.clean_data()
analysis.kmeans_clustering(n_clusters=3)
analysis.find_optimal_clusters()
analysis.pca_visualization()
