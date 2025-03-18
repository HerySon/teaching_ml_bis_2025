"""
Module pour l'application de DBSCAN sur le dataset Open Food Facts.

Contient les classes `DataPreprocessor`, `DBSCANAnalyzer`, et `Plotter` permettant de :
- Nettoyer le dataset en supprimant les colonnes non pertinentes.
- Supprimer les colonnes contenant un taux trop élevé de valeurs manquantes.
- Appliquer DBSCAN pour le clustering.
- Visualiser les résultats du clustering.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


class DataPreprocessor:
    """
    Classe pour le prétraitement des données du dataset Open Food Facts.
    """

    def __init__(self, file_path="datasets/en.openfoodfacts.org.products.csv", nrows=100000):
        """
        Initialise la classe en chargeant un échantillon du dataset Open Food Facts.

        Arguments :
            file_path (str) : Chemin du fichier CSV à charger.
            nrows (int) : Nombre de lignes à charger (par défaut 10 000).
        """
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)

        try:
            self.df = pd.read_csv(
                file_path,
                sep="\t",
                on_bad_lines='skip',
                nrows=nrows,
                low_memory=False
            )
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            self.df = pd.DataFrame()

    def remove_irrelevant_columns(self):
        """
        Supprime les colonnes non pertinentes pour l'analyse.

        Retour :
            pd.DataFrame : Le DataFrame nettoyé, sans les colonnes jugées inutiles.
        """
        columns_to_drop = [
            "code", "url", "creator", "created_t", "created_datetime",
            "last_modified_t", "last_modified_datetime", "packaging", "packaging_tags",
            "brands_tags", "categories_tags", "categories_fr",
            "origins_tags", "manufacturing_places", "manufacturing_places_tags",
            "labels_tags", "labels_fr", "emb_codes", "emb_codes_tags",
            "first_packaging_code_geo", "cities", "cities_tags", "purchase_places",
            "countries_tags", "countries_fr", "image_ingredients_url",
            "image_ingredients_small_url", "image_nutrition_url", "image_nutrition_small_url",
            "image_small_url", "image_url", "last_updated_t", "last_updated_datetime",
            "last_modified_by", "last_image_t", "last_image_datetime"
        ]

        self.df.drop(columns=[col for col in columns_to_drop if col in self.df.columns], errors='ignore', inplace=True)
        return self.df

    def remove_high_nan_columns(self, threshold=70):
        """
        Supprime les colonnes contenant un pourcentage de valeurs manquantes supérieur au seuil défini.

        Arguments :
            threshold (float) : Pourcentage maximal de valeurs manquantes toléré dans une colonne.
                                Par défaut, ce seuil est fixé à 70%.

        Retour :
            pd.DataFrame : Le DataFrame après suppression des colonnes trop incomplètes.
        """
        nan_ratio = self.df.isna().mean() * 100  # Calcul du pourcentage de valeurs NaN par colonne
        cols_to_remove = nan_ratio[nan_ratio > threshold].index.tolist()  # Sélection des colonnes à supprimer

        self.df.drop(columns=cols_to_remove, inplace=True)  # Suppression des colonnes sélectionnées
        return self.df

    def pre_processing(self):
        """
        Effectue les étapes de prétraitement suivantes sur le DataFrame :
        1. Sélectionne les colonnes avec '_100g' dans leur nom.
        2. Supprime les colonnes fortement corrélées (seuil > 0.8).
        3. Supprime les lignes avec des valeurs manquantes.
        4. Standardise les données (normalisation).

        Retour :
            pd.DataFrame : Les données normalisées et traitées.
        """
        if self.df.empty:
            print("DataFrame is empty. Please check the file path and try again.")
            return self.df

        # Sélectionner les colonnes '_100g'
        col100g = [col for col in self.df.columns if "_100g" in col]
        data_100g = self.df[col100g]

        # Calcul de la matrice de corrélation
        corr_matrix = data_100g.corr().abs()

        # Déterminer les colonnes fortement corrélées (seuil > 0.8)
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]

        # Supprimer les colonnes redondantes
        data_100g = data_100g.drop(columns=to_drop)

        # Supprimer les lignes contenant des valeurs manquantes
        data_100g = data_100g.dropna()

        return data_100g


class DBSCANAnalyzer:
    """
    Classe pour appliquer DBSCAN sur les données prétraitées.
    """

    def __init__(self, data):
        """
        Initialise la classe avec les données prétraitées.

        Arguments :
            data (pd.DataFrame) : Les données prétraitées.
        """
        if data.empty:
            raise ValueError("Data is empty. Please provide a valid DataFrame.")
        self.data = data

    def apply_dbscan(self, eps=5, min_samples=5):
        """
        Applique DBSCAN après normalisation des données et traitement des outliers.

        Arguments :
            eps (float) : Distance maximale entre deux échantillons pour qu'ils soient considérés comme voisins.
            min_samples (int) : Nombre minimum d'échantillons dans un voisinage pour qu'un point soit considéré comme un noyau.

        Retour :
            pd.DataFrame : Le DataFrame avec les clusters ajoutés.
        """
        # Standardisation des données
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)

        # Appliquer t-SNE pour réduire à 2D
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(scaled_data)

        # Appliquer DBSCAN sur les résultats de t-SNE
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        clusters = dbscan.fit_predict(X_tsne)

        # Ajouter les clusters au DataFrame
        self.data["tsne1"], self.data["tsne2"] = X_tsne[:, 0], X_tsne[:, 1]
        self.data["cluster"] = clusters

        return self.data

    def get_kdist_plot(self, k):
        """
        Affiche le graphique des distances k-distances pour déterminer le bon paramètre eps pour DBSCAN.

        Arguments :
            k (int) : Nombre de voisins à utiliser pour le calcul des distances.
        """
        # Standardisation des données
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)

        nbrs = NearestNeighbors(n_neighbors=k).fit(scaled_data)
        distances, _ = nbrs.kneighbors(scaled_data)

        distances = np.sort(distances, axis=0)
        distances = distances[:, k-1]

        plt.figure(figsize=(8, 8))
        plt.plot(distances)
        plt.xlabel('Points/Objects in the dataset')
        plt.ylabel(f'Sorted {k}-nearest neighbor distance')
        plt.grid(True, linestyle="--", color='black', alpha=0.4)
        plt.show()


class Plotter:
    """
    Classe pour visualiser les résultats du clustering.
    """

    @staticmethod
    def plot_clusters(data):
        """
        Visualise les clusters DBSCAN après t-SNE, en mettant en évidence les points de bruit.

        Arguments :
            data (pd.DataFrame) : Le DataFrame contenant les données et les clusters.
        """
        plt.figure(figsize=(10, 6))
        
        # Plot clusters
        sns.scatterplot(
            x=data["tsne1"], y=data["tsne2"], hue=data["cluster"], palette="tab10", alpha=0.5, edgecolor='w'
        )
        
        # Highlight noise points
        noise = data[data["cluster"] == -1]
        plt.scatter(noise["tsne1"], noise["tsne2"], color='red', label='Noise', alpha=0.5, edgecolor='w')
        
        plt.title("Clusters DBSCAN après t-SNE")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.show()


# Example usage of the DataPreprocessor, DBSCANAnalyzer, and Plotter class methods

# Initialize the DataPreprocessor class
preprocessor = DataPreprocessor()

# Remove irrelevant columns
print("Removing irrelevant columns...")
df_cleaned = preprocessor.remove_irrelevant_columns()
print(f"Shape after removing irrelevant columns: {df_cleaned.shape}")

# Remove high NaN columns
print("\nRemoving high NaN columns...")
df_no_high_nan = preprocessor.remove_high_nan_columns(threshold=70)
print(f"Shape after removing high NaN columns: {df_no_high_nan.shape}")

# Preprocess the data
print("\nPreprocessing the data...")
data = preprocessor.pre_processing()
print(f"Shape after preprocessing: {data.shape}")

if not data.empty:
    # Initialize the DBSCANAnalyzer class
    dbscan_analyzer = DBSCANAnalyzer(data)
    
    # Apply DBSCAN
    print("\nApplying DBSCAN...")
    clusters = dbscan_analyzer.apply_dbscan(eps=2.5, min_samples=6)
    print(f"Shape after applying DBSCAN: {clusters.shape}")
    print(clusters.head())
    
    # Plot clusters
    print("\nPlotting clusters...")
    Plotter.plot_clusters(clusters)
    
    # Optionally, plot k-distance plot
    print("\nPlotting k-distance plot...")
    dbscan_analyzer.get_kdist_plot(40)
else:
    print("Preprocessed data is empty. Please check the preprocessing steps.")
