"""
Module pour l'Analyse en Composantes Principales (ACP) sur le dataset Open Food Facts.

Contient les classes `DataPreprocessor` et `PCAAnalyzer` permettant de :
- Nettoyer le dataset en supprimant les colonnes non pertinentes.
- Éliminer les doublons afin d'assurer la qualité des données.
- Supprimer les colonnes contenant un taux trop élevé de valeurs manquantes.
- Appliquer l'ACP pour réduire la dimensionnalité des données.
- Visualiser les résultats de l'ACP.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class DataPreprocessor:
    """
    Classe pour le prétraitement des données du dataset Open Food Facts.
    """

    def __init__(self, file_path="datasets/en.openfoodfacts.org.products.csv", nrows=10000):
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


class PCAAnalyzer:
    """
    Classe pour l'Analyse en Composantes Principales (ACP) sur les données prétraitées.
    """

    def __init__(self, data):
        """
        Initialise la classe avec les données prétraitées.

        Arguments :
            data (pd.DataFrame) : Les données prétraitées.
        """
        self.data = data
        self.scaled_data = None
        self.pca_components = None
        self.explained_variance_ratio = None

    def apply_pca(self, n_components=5):
        """
        Applique la standardisation des données et effectue une Analyse en Composantes Principales (ACP).

        Arguments :
            n_components (int) : Nombre de composantes principales à conserver (par défaut 5).

        Retour :
            tuple :
                - np.ndarray : Les données projetées dans l'espace des composantes principales.
                - np.ndarray : La proportion de variance expliquée par chaque composante.
        """
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.data)

        # Application de l'ACP
        pca = PCA(n_components=n_components)
        self.pca_components = pca.fit_transform(self.scaled_data)

        # Récupération de la variance expliquée par chaque composante principale
        self.explained_variance_ratio = pca.explained_variance_ratio_

        return self.pca_components, self.explained_variance_ratio

    def plot_explained_variance(self):
        """
        Génère un graphique illustrant la variance expliquée par chaque composante principale.
        """
        plt.figure(figsize=(8, 6))
        plt.bar(range(1, len(self.explained_variance_ratio) + 1), self.explained_variance_ratio, alpha=0.7)
        plt.xlabel('Composantes principales')
        plt.ylabel('Variance expliquée')
        plt.title('Variance expliquée par chaque composante principale')
        plt.show()

    def cumulative_variance_plot(self):
        """
        Affiche un graphique de la variance expliquée cumulée.
        """
        cumulative_variance = np.cumsum(self.explained_variance_ratio)

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='b', label='Variance cumulée')
        plt.xlabel('Nombre de composantes principales')
        plt.ylabel('Variance cumulée expliquée')
        plt.title('Variance cumulée expliquée')

        plt.axhline(y=0.95, color='r', linestyle='-', label='95% Variance expliquée')
        plt.axvline(x=np.argmax(cumulative_variance >= 0.95) + 1, color='r', linestyle='--', label=f'{np.argmax(cumulative_variance >= 0.95) + 1} composantes')

        plt.legend()
        plt.show()

    def plot_pca_biplot(self, truncate_outliers=True, threshold=1.5):
        """
        Affiche un biplot de l'ACP combinant :
        - La projection des observations sur les deux premières composantes principales.
        - Les vecteurs des variables contribuant à ces composantes.
        """
        data_100g = self.data

        if truncate_outliers:
            Q1 = data_100g.quantile(0.25)
            Q3 = data_100g.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            data_100g = data_100g[~((data_100g < lower_bound) | (data_100g > upper_bound)).any(axis=1)]

        scaled_data_100g = StandardScaler().fit_transform(data_100g)

        pca = PCA(n_components=2)
        pca_scores = pca.fit_transform(scaled_data_100g)
        loadings = pca.components_.T

        plt.figure(figsize=(10, 7))
        sns.scatterplot(x=pca_scores[:, 0], y=pca_scores[:, 1], alpha=0.5, label="Observations")

        for i, var in enumerate(data_100g.columns):
            plt.arrow(0, 0, loadings[i, 0] * 3, loadings[i, 1] * 3, color='r', alpha=0.5, head_width=0.05)
            plt.text(loadings[i, 0] * 3.2, loadings[i, 1] * 3.2, var, color='r', fontsize=10)

        plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
        plt.xlabel("Composante principale 1")
        plt.ylabel("Composante principale 2")
        plt.title("Biplot de l'ACP")
        plt.legend()
        plt.show()

    def plot_pca_scatter(self, truncate_outliers=True, threshold=1.5):
        """
        Affiche un nuage de points des données projetées sur les deux premières composantes principales.
        """
        if self.pca_components is None:
            raise ValueError("L'ACP n'a pas encore été appliquée. Exécutez `apply_pca()` d'abord.")

        data_100g = self.data

        if truncate_outliers:
            Q1 = data_100g.quantile(0.25)
            Q3 = data_100g.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            data_100g = data_100g[~((data_100g < lower_bound) | (data_100g > upper_bound)).any(axis=1)]

        scaled_data_100g = StandardScaler().fit_transform(data_100g)

        pca = PCA(n_components=2)
        pca_scores = pca.fit_transform(scaled_data_100g)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=pca_scores[:, 0], y=pca_scores[:, 1], alpha=0.5)
        plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
        plt.xlabel("Composante principale 1")
        plt.ylabel("Composante principale 2")
        plt.title("Nuage de points des données après ACP")
        plt.show()


# Example usage of the DataPreprocessor and PCAAnalyzer class methods

# Initialize the DataPreprocessor class
preprocessor = DataPreprocessor()

# Remove irrelevant columns
print("Removing irrelevant columns...")
df_cleaned = preprocessor.remove_irrelevant_columns()
print(df_cleaned.head())

# Remove high NaN columns
print("\nRemoving high NaN columns...")
df_no_high_nan = preprocessor.remove_high_nan_columns(threshold=90)
print(df_no_high_nan.head())

# Preprocess the data
print("\nPreprocessing the data...")
data = preprocessor.pre_processing()
print(data.head())

# Initialize the PCAAnalyzer class
pca_analyzer = PCAAnalyzer(data)

# Apply PCA
print("\nApplying PCA...")
pca_components, explained_variance_ratio = pca_analyzer.apply_pca()
print("PCA Components:\n", pca_components)
print("Explained Variance Ratio:\n", explained_variance_ratio)

# Plot explained variance
print("\nPlotting explained variance...")
pca_analyzer.plot_explained_variance()

# Plot cumulative variance
print("\nPlotting cumulative variance...")
pca_analyzer.cumulative_variance_plot()

# Plot PCA biplot
print("\nPlotting PCA biplot...")
pca_analyzer.plot_pca_biplot()

# Plot PCA scatter plot
print("\nPlotting PCA scatter plot...")
pca_analyzer.plot_pca_scatter()
