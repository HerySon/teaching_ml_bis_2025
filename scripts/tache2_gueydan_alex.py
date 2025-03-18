"""
Module pour le scaling des données du dataset Open Food Facts.

Contient la classe `Tache2` permettant de :
- Appliquer différentes techniques de scaling aux colonnes numériques du dataset.
- Utiliser des méthodes telles que MinMaxScaler, StandardScaler, RobustScaler, Normalizer, et PowerTransformer.
- Mettre à l'échelle les données pour les préparer à l'analyse ou à l'utilisation dans des modèles de machine learning.
"""

from sklearn.preprocessing import (
    MinMaxScaler, Normalizer, RobustScaler, StandardScaler, PowerTransformer
)
import pandas as pd


class Tache2:
    """
    Classe permettant d'appliquer des techniques de scaling sur les données numériques du dataset Open Food Facts.
    Elle inclut des méthodes pour :
    - Appliquer le MinMaxScaler pour ramener les données dans la plage [0, 1].
    - Appliquer le StandardScaler pour centrer les données (moyenne = 0) et réduire leur écart-type à 1.
    - Appliquer le RobustScaler pour rendre les données robustes aux valeurs aberrantes.
    - Appliquer le Normalizer pour normaliser les données (norme = 1).
    - Appliquer le PowerTransformer pour rendre les données plus proches d'une distribution normale.
    """

    def __init__(self):
        pd.set_option("display.max_columns", None)  # Afficher toutes les colonnes
        pd.set_option("display.max_rows", None)  # Afficher toutes les lignes

        # Chargement du dataset en spécifiant des options pour éviter les erreurs de format et optimiser les performances
        self.df = pd.read_csv(
            "datasets/en.openfoodfacts.org.products.csv",
            sep="\t",
            on_bad_lines='skip',
            nrows=100000,
            low_memory=False
        )

        print(self.df.shape)  # Affichage de la taille du DataFrame après chargement

    def remove_high_nan_columns(self, threshold=90):
        """
        Supprime les colonnes avec un pourcentage de valeurs manquantes supérieur au seuil.

        Arguments :
            threshold (float) : Pourcentage de NaN au-dessus duquel une colonne est supprimée.

        Retour :
            pd.DataFrame : DataFrame après suppression des colonnes avec trop de NaN.
        """
        # Calcul du pourcentage de NaN par colonne
        nan_ratio = self.df.isna().mean() * 100

        # Identification des colonnes à supprimer
        cols_to_remove = nan_ratio[nan_ratio > threshold].index.tolist()

        # Suppression des colonnes avec trop de NaN
        self.df.drop(columns=cols_to_remove, inplace=True)

        return self.df  # Retourner le DataFrame modifié

    def normalization(self):
        """
        Applique la normalisation (Normalizer) sur les colonnes numériques.

        La normalisation transforme chaque ligne pour qu'elle ait une norme égale à 1.

        Retour :
            pd.DataFrame : DataFrame normalisé avec les colonnes numériques modifiées.
        """
        # Sélection des colonnes numériques
        numeric_df = self.df.select_dtypes(include=["int64", "float64"])

        # Création de l'objet Normalizer
        scaler = Normalizer()

        # Appliquer la normalisation
        scaled_data = scaler.fit_transform(numeric_df)

        # Retourner un DataFrame avec les données normalisées
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)

        return scaled_df  # DataFrame normalisé

    def min_max_scaling(self):
        """
        Applique le MinMaxScaler sur les colonnes numériques pour les ramener dans la plage [0, 1].

        Retour :
            pd.DataFrame : DataFrame avec les colonnes numériques normalisées.
        """
        # Sélection des colonnes numériques
        numeric_df = self.df.select_dtypes(include=["int64", "float64"])

        # Application du MinMaxScaler pour normaliser les données
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        # Création d'un DataFrame avec les données normalisées
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)

        return scaled_df  # Retourne les données normalisées

    def standard_scaling(self):
        """
        Applique le StandardScaler sur les colonnes numériques pour les centrer
        (moyenne = 0) et réduire leur écart-type à 1.

        Retour :
            pd.DataFrame : DataFrame avec les colonnes numériques standardisées.
        """
        # Sélection des colonnes numériques
        numeric_df = self.df.select_dtypes(include=["int64", "float64"])

        # Application du StandardScaler pour centrer et réduire les données
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        # Création d'un DataFrame avec les données standardisées
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)

        return scaled_df  # Retourne les données standardisées

    def robust_scaling(self):
        """
        Applique le RobustScaler sur les colonnes numériques pour les rendre
        robustes aux valeurs aberrantes. Les valeurs NaN sont remplacées par la
        médiane avant la transformation.

        Retour :
            pd.DataFrame : DataFrame avec les colonnes numériques robustes.
        """
        # Sélection des colonnes numériques
        numeric_df = self.df.select_dtypes(include=["int64", "float64"])

        # Application du RobustScaler pour traiter les données avec des valeurs aberrantes
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        # Création d'un DataFrame avec les données robustes
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)

        return scaled_df  # Retourne les données robustes

    def power_scaling(self):
        """
        Applique le PowerTransformer sur les colonnes numériques pour transformer les données
        en une distribution plus proche de la normale.

        Retour :
            pd.DataFrame : DataFrame avec les colonnes numériques transformées.
        """
        # Sélection des colonnes numériques
        numeric_df = self.df.select_dtypes(include=["int64", "float64"])

        # Application du PowerTransformer pour rendre les données plus normales
        scaler = PowerTransformer(method='yeo-johnson')  # 'yeo-johnson' permet de traiter aussi les valeurs négatives
        scaled_data = scaler.fit_transform(numeric_df)

        # Création d'un DataFrame avec les données transformées
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)

        return scaled_df  # Retourne les données transformées


# Example usage of the Tache2 class methods

# Initialize the Tache2 class
tache = Tache2()

# Remove high NaN columns
print("Removing high NaN columns...")
df_no_high_nan = tache.remove_high_nan_columns(threshold=90)
print(df_no_high_nan.head())

# Apply normalization
print("\nApplying normalization...")
df_normalized = tache.normalization()
print(df_normalized.describe())

# Apply MinMax scaling
print("\nApplying MinMax scaling...")
df_min_max_scaled = tache.min_max_scaling()
print(df_min_max_scaled.describe())

# Apply standard scaling
print("\nApplying standard scaling...")
df_standard_scaled = tache.standard_scaling()
print(df_standard_scaled.describe())

# Apply robust scaling
print("\nApplying robust scaling...")
df_robust_scaled = tache.robust_scaling()
print(df_robust_scaled.describe())

# Apply power scaling
print("\nApplying power scaling...")
df_power_scaled = tache.power_scaling()
print(df_power_scaled.describe())
