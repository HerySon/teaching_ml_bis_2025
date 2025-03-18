"""
Module pour l'analyse avancée du dataset Open Food Facts.

Contient les classes `DataPreprocessor`, `DataAnalyzer`, et `Plotter` permettant de :
- Nettoyer le dataset en supprimant les colonnes non pertinentes.
- Supprimer les colonnes contenant un taux trop élevé de valeurs manquantes.
- Analyser les corrélations entre les variables (Pearson, Spearman).
- Identifier les variables redondantes.
- Tester l'indépendance entre variables catégorielles avec le chi-carré.
- Vérifier la normalité des distributions.
- Visualiser les relations entre variables via des heatmaps et des modèles polynomiaux.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, shapiro
from sklearn.impute import KNNImputer


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
            on_bad_lines="skip",
            nrows=nrows,
            low_memory=False
        )

    def remove_irrelevant_columns(self):
        """
        Supprime les colonnes non pertinentes pour l'analyse.

        Retour :
            pd.DataFrame : Le DataFrame nettoyé.
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
            "image_small_url", "image_url", "last_updated_t", "last_updated_datetime", "last_modified_by", "last_image_t"
        ]

        self.df.drop(columns=[col for col in columns_to_drop if col in self.df.columns], errors='ignore', inplace=True)
        return self.df

    def remove_duplicates(self):
        """
        Supprime les doublons en considérant toutes les colonnes.

        Retour :
            pd.DataFrame : Le DataFrame sans doublons.
        """
        self.df.drop_duplicates(keep="first", inplace=True)
        return self.df

    def remove_high_nan_columns(self, threshold=90):
        """
        Supprime les colonnes ayant un pourcentage de valeurs manquantes supérieur au seuil.

        Arguments :
            threshold (float) : Pourcentage de valeurs manquantes au-dessus duquel une colonne est supprimée.

        Retour :
            pd.DataFrame : Le DataFrame sans les colonnes trop incomplètes.
        """
        nan_ratio = self.df.isna().mean() * 100
        cols_to_remove = nan_ratio[nan_ratio > threshold].index.tolist()
        self.df.drop(columns=cols_to_remove, inplace=True)
        return self.df

    def knn_imputer(self, n_neighbors):
        """
        Impute les valeurs manquantes d'une colonne spécifique en utilisant KNN Imputer,
        en se basant sur les colonnes corrélées.

        Arguments :
            n_neighbors (int) : Nombre de voisins à utiliser pour l'imputation.

        Retour :
            pd.DataFrame : Le DataFrame avec les valeurs manquantes imputées.
        """
        df_numeric = self.df.select_dtypes(include=['float64', 'int64'])
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = imputer.fit_transform(df_numeric)
        df_subset_imputed = pd.DataFrame(imputed_data, columns=df_numeric.columns)
        return df_subset_imputed


class DataAnalyzer:
    """
    Classe pour analyser les données du dataset Open Food Facts.
    """

    def __init__(self, df):
        """
        Initialise la classe avec les données prétraitées.

        Arguments :
            df (pd.DataFrame) : Les données prétraitées.
        """
        self.df = df

    def heatmap(self, threshold):
        """
        Affiche une heatmap des corrélations entre les variables du dataset,
        en filtrant celles qui sont supérieures ou inférieures au seuil spécifié.

        Arguments :
            threshold (float) : Le seuil de corrélation pour filtrer les valeurs.
                                 Les corrélations supérieures ou inférieures à ce seuil seront affichées.
        """
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64'])
        correlation_matrix = numeric_cols.corr().abs()
        correlation_matrix = correlation_matrix[(correlation_matrix.abs() > threshold)]

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
        plt.title(f'Correlation Heatmap with threshold {threshold}')
        plt.show()

    def spearman_correlation(self, threshold):
        """
        Calcule la corrélation de Spearman entre toutes les variables.
        Cette méthode permet de capturer des relations monotones non linéaires.

        Arguments :
            threshold (float) : Le seuil de corrélation pour filtrer les valeurs.
                                 Les corrélations supérieures à ce seuil seront affichées.

        Retour :
            spearman_corr (pd.DataFrame) : La matrice de corrélation de Spearman.
        """
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64'])
        correlation_matrix = numeric_cols.corr(method="spearman").abs()
        correlation_matrix = correlation_matrix[(correlation_matrix.abs() > threshold) & (correlation_matrix != 1)]

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
        plt.title(f'Correlation Heatmap with threshold {threshold}')
        plt.show()

    def chi2(self, colonne1, colonne2):
        """
        Effectue un test du chi-carré pour vérifier l'indépendance entre deux variables catégorielles.

        Arguments :
            colonne1 (str) : Le nom de la première variable.
            colonne2 (str) : Le nom de la seconde variable.
        """
        chi2_stat, p_value, dof, expected = chi2_contingency(pd.crosstab(self.df[colonne1], self.df[colonne2]))

        if p_value < 0.05:
            print(f"Les variables {colonne1} et {colonne2} sont dépendantes.")
        else:
            print(f"Les variables {colonne1} et {colonne2} sont indépendantes.")

    def find_highly_correlated_features(self, threshold):
        """
        Identifie les variables ayant une corrélation élevée et qui pourraient être redondantes.

        Arguments :
            threshold (float) : Le seuil de corrélation pour identifier les variables redondantes.

        Retour :
            redundant_features (list) : Liste des variables redondantes.
        """
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64'])
        corr_matrix = numeric_cols.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)

        redundant_features = [
            column for column in corr_matrix.columns if any(corr_matrix[column] > threshold)
        ]

        print(f"Variables redondantes : {redundant_features}")

        return redundant_features

    def check_normality(self, column):
        """
        Vérifie si une variable suit une distribution normale à l'aide du test de Shapiro-Wilk.

        Arguments :
            column (str) : Le nom de la colonne à tester.
        """
        stat, p_value = shapiro(self.df[column].dropna())
        if p_value > 0.05:
            print(f"La variable {column} suit une distribution normale.")
        else:
            print(f"La variable {column} ne suit pas une distribution normale.")


class Plotter:
    """
    Classe pour visualiser les relations entre les variables du dataset Open Food Facts.
    """

    @staticmethod
    def plot_polynomial_relationship(df, x_col, y_col, degree=2):
        """
        Affiche une relation polynomiale entre deux variables.

        Arguments :
            df (pd.DataFrame) : Le DataFrame contenant les données.
            x_col (str) : Le nom de la colonne sur l'axe des abscisses.
            y_col (str) : Le nom de la colonne sur l'axe des ordonnées.
            degree (int) : Le degré du polynôme à ajuster.
        """
        plt.figure(figsize=(8, 5))
        sns.regplot(x=df[x_col], y=df[y_col], order=degree, scatter_kws={"s": 10}, line_kws={"color": "red"})
        plt.title(f"Relation polynomiale (degré {degree}) entre {x_col} et {y_col}")
        plt.show()


# Example usage of the DataPreprocessor, DataAnalyzer, and Plotter class methods

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

# Remove duplicates
print("\nRemoving duplicates...")
df_no_duplicates = preprocessor.remove_duplicates()
print(df_no_duplicates.head())

# Impute missing values using KNN Imputer
print("\nImputing missing values using KNN Imputer...")
df_imputed = preprocessor.knn_imputer(n_neighbors=5)
print(df_imputed.head())

# Initialize the DataAnalyzer class
analyzer = DataAnalyzer(df_imputed)

# Generate heatmap
print("\nGenerating heatmap...")
analyzer.heatmap(threshold=0.5)

# Calculate Spearman correlation
print("\nCalculating Spearman correlation...")
analyzer.spearman_correlation(threshold=0.3)

# Find highly correlated features
print("\nFinding highly correlated features...")
redundant_features = analyzer.find_highly_correlated_features(threshold=0.7)
print(f"Redundant features: {redundant_features}")

# Check normality of a specific column
print("\nChecking normality of 'sodium_100g' column...")
analyzer.check_normality("sodium_100g")

# Initialize the Plotter class and plot polynomial relationship
print("\nPlotting polynomial relationship...")
Plotter.plot_polynomial_relationship(df_imputed, "sodium_100g", "energy_100g", degree=2)
