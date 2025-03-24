"""
Module pour l'Analyse en Composantes Principales (ACP) sur le dataset Open Food Facts.

Contient les classes `DataPreprocessor`, `UMAPAnalyzer`, et `TSNEAnalyzer` permettant de :
- Nettoyer le dataset en supprimant les colonnes non pertinentes.
- Éliminer les doublons afin d'assurer la qualité des données.
- Supprimer les colonnes contenant un taux trop élevé de valeurs manquantes.
- Appliquer UMAP et t-SNE pour réduire la dimensionnalité des données.
- Visualiser les résultats de UMAP et t-SNE.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.manifold import TSNE


class DataPreprocessor:
    """
    Classe pour le prétraitement des données du dataset Open Food Facts.
    """

    def __init__(self, file_path="datasets/en.openfoodfacts.org.products.csv", nrows=50000):
        """
        Initialise la classe en chargeant un échantillon du dataset Open Food Facts.

        Arguments :
            file_path (str) : Chemin du fichier CSV à charger.
            nrows (int) : Nombre de lignes à charger (par défaut 10 000).
        """
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)

        self.df = pd.read_csv(
            file_path,
            sep="\t",
            on_bad_lines='skip',
            nrows=nrows,
            low_memory=False
        )

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

    def remove_duplicates(self):
        """
        Supprime les doublons dans le DataFrame.

        Retour :
            pd.DataFrame : Le DataFrame sans doublons.
        """
        self.df.drop_duplicates(keep="first", inplace=True)
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
        # Sélectionner les colonnes '_100g'
        col100g = [col for col in self.df.columns if '_100g' in col]
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


class UMAPAnalyzer:
    """
    Classe pour l'Analyse UMAP sur les données prétraitées.
    """

    def __init__(self, data):
        """
        Initialise la classe avec les données prétraitées.

        Arguments :
            data (pd.DataFrame) : Les données prétraitées.
        """
        self.data = data
        self.scaled_data = None
        self.umap_embedding = None

    def apply_umap(self, n_neighbors=15, min_dist=0.1, n_components=2):
        """
        Applique la standardisation des données et effectue une Analyse UMAP.

        Arguments :
            n_neighbors (int) : Nombre de voisins pour UMAP (par défaut 15).
            min_dist (float) : Distance minimale pour UMAP (par défaut 0.1).
            n_components (int) : Nombre de composantes à conserver (par défaut 2).

        Retour :
            np.ndarray : Les données projetées dans l'espace UMAP.
        """
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.data)

        # Application de UMAP
        umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
        self.umap_embedding = umap_model.fit_transform(self.scaled_data)

        return self.umap_embedding

    def plot_umap(self):
        """
        Affiche un nuage de points des données projetées sur les composantes UMAP.
        """
        if self.umap_embedding is None:
            raise ValueError("UMAP n'a pas encore été appliqué. Exécutez `apply_umap()` d'abord.")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.umap_embedding[:, 0], y=self.umap_embedding[:, 1], alpha=0.5)
        plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title("Nuage de points des données après UMAP")
        plt.show()


class TSNEAnalyzer:
    """
    Classe pour l'Analyse t-SNE sur les données prétraitées.
    """

    def __init__(self, data):
        """
        Initialise la classe avec les données prétraitées.

        Arguments :
            data (pd.DataFrame) : Les données prétraitées.
        """
        self.data = data
        self.scaled_data = None
        self.tsne_embedding = None

    def apply_tsne(self, perplexity=30, n_iter=1000, n_components=2):
        """
        Applique la standardisation des données et effectue une Analyse t-SNE.

        Arguments :
            perplexity (int) : Perplexité pour t-SNE (par défaut 30).
            n_iter (int) : Nombre d'itérations pour t-SNE (par défaut 1000).
            n_components (int) : Nombre de composantes à conserver (par défaut 2).

        Retour :
            np.ndarray : Les données projetées dans l'espace t-SNE.
        """
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.data)

        # Application de t-SNE
        tsne_model = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42)
        self.tsne_embedding = tsne_model.fit_transform(self.scaled_data)

        return self.tsne_embedding

    def plot_tsne(self):
        """
        Affiche un nuage de points des données projetées sur les composantes t-SNE.
        """
        if self.tsne_embedding is None:
            raise ValueError("t-SNE n'a pas encore été appliqué. Exécutez `apply_tsne()` d'abord.")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.tsne_embedding[:, 0], y=self.tsne_embedding[:, 1], alpha=0.5)
        plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title("Nuage de points des données après t-SNE")
        plt.show()


# Example usage of the DataPreprocessor, UMAPAnalyzer, and TSNEAnalyzer class methods

# Initialize the DataPreprocessor class
preprocessor = DataPreprocessor()

# Remove irrelevant columns
print("Removing irrelevant columns...")
df_cleaned = preprocessor.remove_irrelevant_columns()
print(df_cleaned.head())

# Remove high NaN columns
print("\nRemoving high NaN columns...")
df_no_high_nan = preprocessor.remove_high_nan_columns(threshold=70)
print(df_no_high_nan.head())

# Preprocess the data
print("\nPreprocessing the data...")
data = preprocessor.pre_processing()
print(data.head())

# Initialize the UMAPAnalyzer class
umap_analyzer = UMAPAnalyzer(data)

# Apply UMAP
print("\nApplying UMAP...")
umap_embedding = umap_analyzer.apply_umap()
print("UMAP Embedding:\n", umap_embedding)

# Plot UMAP
print("\nPlotting UMAP...")
umap_analyzer.plot_umap()

# Initialize the TSNEAnalyzer class
tsne_analyzer = TSNEAnalyzer(data)

# Apply t-SNE
print("\nApplying t-SNE...")
tsne_embedding = tsne_analyzer.apply_tsne()
print("t-SNE Embedding:\n", tsne_embedding)

# Plot t-SNE
print("\nPlotting t-SNE...")
tsne_analyzer.plot_tsne()
