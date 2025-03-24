import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

class OutlierDetector:
    """
    Classe pour détecter les outliers dans un dataset en utilisant différentes méthodes.

    Méthodes disponibles :
    - tukey_method : Détecte les outliers en utilisant la méthode de Tukey (IQR).
    - z_score_method : Détecte les outliers en utilisant la méthode du Z-score.
    - isolation_forest : Détecte les outliers en utilisant la méthode Isolation Forest.
    - local_outlier_factor : Détecte les outliers en utilisant la méthode Local Outlier Factor (LOF).
    - elliptic_envelope : Détecte les outliers en utilisant la méthode Elliptic Envelope.
    - plot_outliers : Visualise les outliers détectés dans une colonne en utilisant une méthode spécifiée.
    - apply_tsne : Applique t-SNE pour réduire la dimensionnalité des données et visualise les résultats.
    """

    def __init__(self, file_path="datasets/en.openfoodfacts.org.products.csv", nrows=10000):
        """
        Initialise la classe en chargeant un échantillon du dataset Open Food Facts.

        Arguments :
            file_path (str) : Chemin du fichier CSV à charger.
            nrows (int) : Nombre de lignes à charger (par défaut 100 000).
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
    
    def fill_missing_values(self):
        """
        Remplit les valeurs manquantes avec la moyenne des colonnes numériques.
        """
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        return self.df
    
    def remove_high_nan_columns(self, threshold=80):
        """
        Supprime les colonnes ayant un pourcentage de valeurs manquantes supérieur au seuil.

        Arguments :
            threshold (float) : Pourcentage de valeurs manquantes au-dessus duquel une colonne est supprimée.

        Retour :
            pd.DataFrame : Le DataFrame sans les colonnes trop incomplètes.
        """
        # Calcul du pourcentage de valeurs manquantes par colonne
        nan_ratio = self.df.isna().mean() * 100

        # Colonnes à supprimer (taux de NaN supérieur au seuil)
        cols_to_remove = nan_ratio[nan_ratio > threshold].index.tolist()

        # Suppression des colonnes
        self.df.drop(columns=cols_to_remove, inplace=True)
        return self.df

    def remove_duplicates(self):
        """
        Supprime les doublons en considérant toutes les colonnes.

        Retour :
            pd.DataFrame : Le DataFrame sans doublons.
        """
        self.df.drop_duplicates(keep="first", inplace=True)
        return self.df

    def tukey_method(self, column):
        """
        Détecte les outliers dans une colonne en utilisant la méthode de Tukey (IQR).

        Arguments :
            column (str) : Nom de la colonne à analyser.

        Retour :
            pd.Series : Un booléen indiquant si chaque point est un outlier.
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (self.df[column] < lower_bound) | (self.df[column] > upper_bound)

    def z_score_method(self, column, threshold=3):
        """
        Détecte les outliers dans une colonne en utilisant la méthode du Z-score.

        Arguments :
            column (str) : Nom de la colonne à analyser.
            threshold (float) : Seuil de Z-score pour considérer un point comme un outlier (par défaut 3).

        Retour :
            pd.Series : Un booléen indiquant si chaque point est un outlier.
        """
        # Fill missing values with the mean of the column
        z_scores = np.abs(stats.zscore(self.df[column].fillna(self.df[column].mean())))
        return z_scores > threshold

    def isolation_forest(self, contamination=0.1):
        """
        Détecte les outliers dans les données en utilisant la méthode Isolation Forest.

        Arguments :
            contamination (float) : Proportion de points à considérer comme des outliers (par défaut 0.05).

        Retour :
            pd.Series : Un booléen indiquant si chaque point est un outlier.
        """
        numeric_data = self.df.select_dtypes(include=["int64", "float64"])
        if numeric_data.empty:
            raise ValueError("No numeric data available for Isolation Forest after filling missing values.")
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(numeric_data)
        return outliers == -1

    def local_outlier_factor(self, n_neighbors=25, contamination=0.15):
        """
        Détecte les outliers dans les données en utilisant la méthode Local Outlier Factor (LOF).

        Arguments :
            n_neighbors (int) : Nombre de voisins à utiliser pour calculer le LOF (par défaut 20).
            contamination (float) : Proportion de points à considérer comme des outliers (par défaut 0.05).

        Retour :
            pd.Series : Un booléen indiquant si chaque point est un outlier.
        """

        numeric_data = self.df.select_dtypes(include=["int64", "float64"])
        if numeric_data.isnull().values.any():
            raise ValueError("There are still NaN values in the numeric data.")
        if numeric_data.empty:
            raise ValueError("No numeric data available for Local Outlier Factor after filling missing values.")
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        outliers = lof.fit_predict(numeric_data)
        return outliers == -1

    def elliptic_envelope(self, contamination=0.10):
        """
        Détecte les outliers dans les données en utilisant la méthode Elliptic Envelope.

        Arguments :
            contamination (float) : Proportion de points à considérer comme des outliers (par défaut 0.05).

        Retour :
            pd.Series : Un booléen indiquant si chaque point est un outlier.
        """
        numeric_data = self.df.select_dtypes(include=["int64", "float64"])
        if numeric_data.empty:
            raise ValueError("No numeric data available for Elliptic Envelope after filling missing values.")
        envelope = EllipticEnvelope(contamination=contamination, random_state=42)
        outliers = envelope.fit_predict(numeric_data)
        return outliers == -1

    def plot_outliers(self, column, method):
        """
        Visualise les outliers détectés dans une colonne en utilisant une méthode spécifiée.

        Arguments :
            column (str) : Nom de la colonne à analyser.
            method (str) : Méthode à utiliser pour détecter les outliers ('tukey', 'z_score', 'isolation_forest', 'lof', 'elliptic_envelope').
        """
        if method == 'tukey':
            outliers = self.tukey_method(column)
        elif method == 'z_score':
            outliers = self.z_score_method(column)
        elif method == 'isolation_forest':
            outliers = self.isolation_forest()
        elif method == 'lof':
            outliers = self.local_outlier_factor()
        elif method == 'elliptic_envelope':
            outliers = self.elliptic_envelope()
        else:
            raise ValueError("Invalid method. Choose from 'tukey', 'z_score', 'isolation_forest', 'lof', 'elliptic_envelope'.")

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.df.index, y=self.df[column], hue=outliers, palette={True: 'red', False: 'blue'}, alpha=0.5)
        plt.title(f"Outliers detected using {method} method")
        plt.xlabel("Index")
        plt.ylabel(column)
        plt.legend(title="Outlier", loc='upper right', labels=['No', 'Yes'])
        plt.show()

# Example usage of the OutlierDetector class

# Initialize the OutlierDetector class
outlier_detector = OutlierDetector()

# Remove duplicates
outlier_detector.remove_duplicates()

outlier_detector.remove_high_nan_columns()

# Fill missing values
outlier_detector.fill_missing_values()

# Detect outliers using Tukey's method
print("\nDetecting outliers using Tukey's method...")
tukey_outliers = outlier_detector.tukey_method("sugars_100g")
print(tukey_outliers)

# Detect outliers using Z-score method
print("\nDetecting outliers using Z-score method...")
z_score_outliers = outlier_detector.z_score_method('sugars_100g')
print(z_score_outliers)

# Detect outliers using Isolation Forest
print("\nDetecting outliers using Isolation Forest...")
try:
    iso_forest_outliers = outlier_detector.isolation_forest()
    print(iso_forest_outliers)
except ValueError as e:
    print(e)

# Detect outliers using Local Outlier Factor (LOF)
print("\nDetecting outliers using Local Outlier Factor (LOF)...")
try:
    lof_outliers = outlier_detector.local_outlier_factor()
    print(lof_outliers)
except ValueError as e:
    print(e)

# Detect outliers using Elliptic Envelope
print("\nDetecting outliers using Elliptic Envelope...")
try:
    elliptic_envelope_outliers = outlier_detector.elliptic_envelope()
    print(elliptic_envelope_outliers)
except ValueError as e:
    print(e)

# Plot outliers detected using Tukey's method
print("\nPlotting outliers detected using Tukey's method...")
outlier_detector.plot_outliers('sugars_100g', method='tukey')

# Plot outliers detected using Z-score method
print("\nPlotting outliers detected using Z-score method...")
outlier_detector.plot_outliers('sugars_100g', method='z_score')

# Plot outliers detected using Isolation Forest
print("\nPlotting outliers detected using Isolation Forest...")
try:
    outlier_detector.plot_outliers('sugars_100g', method='isolation_forest')
except ValueError as e:
    print(e)

# Plot outliers detected using Local Outlier Factor (LOF)
print("\nPlotting outliers detected using Local Outlier Factor (LOF)...")
try:
    outlier_detector.plot_outliers('sugars_100g', method='lof')
except ValueError as e:
    print(e)

# Plot outliers detected using Elliptic Envelope
print("\nPlotting outliers detected using Elliptic Envelope...")
try:
    outlier_detector.plot_outliers('sugars_100g', method='elliptic_envelope')
except ValueError as e:
    print(e)
