"""
Module pour le nettoyage du dataset Open Food Facts.

Contient la classe `Tache1` permettant de :
- Supprimer les variables non pertinentes pour l'analyse.
- Gérer les variables ayant trop de valeurs manquantes.
- Imputer les valeurs manquantes pour certaines variables.
- Extraire des motifs spécifiques (par exemple, les quantités dans `serving_size`).
- Traiter les variables présentant des erreurs.
"""

import pandas as pd
import re
from sklearn.impute import KNNImputer


class Tache1:
    """
    Classe permettant de nettoyer et prétraiter un dataset Open Food Facts.
    Elle inclut des méthodes pour :
    - Supprimer les variables non pertinentes.
    - Gérer les valeurs manquantes en imputant ou supprimant.
    - Extraire des informations spécifiques dans des variables comme `serving_size`.
    - Traiter les erreurs dans certaines variables.
    """

    def __init__(self):
        """
        Initialise la classe et charge un échantillon de 10 000 lignes du dataset Open Food Facts
        en utilisant le séparateur tabulation.

        Attributs :
            df (DataFrame): Un DataFrame Pandas contenant l'échantillon du dataset.
        """
        # Forçage de pandas à afficher autant de caractères qu'il peut sur une ligne
        pd.set_option("display.max_columns", None)

        # Même chose avec le nombre de lignes
        pd.set_option("display.max_rows", None)

        # Chargement du dataset, en prenant un échantillon de 10 000 lignes pour ne pas trop surcharger
        self.df = pd.read_csv(
            "datasets/en.openfoodfacts.org.products.csv",
            sep="\t",
            on_bad_lines="skip",
            nrows=10000,
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
        # Calcul du pourcentage de valeurs manquantes par colonne
        nan_ratio = self.df.isna().mean() * 100

        # Colonnes à supprimer (taux de NaN supérieur au seuil)
        cols_to_remove = nan_ratio[nan_ratio > threshold].index.tolist()

        # Suppression des colonnes
        self.df.drop(columns=cols_to_remove, inplace=True)
        return self.df

    def get_column_count(self):
        """Retourne le nombre total de colonnes du DataFrame."""
        return self.df.shape[1]

    def clean_column(self, nom_colonne):
        """
        Nettoie la colonne 'serving_size' :
        - Extrait la quantité principale en grammes ou millilitres.
        - Convertit les unités en valeurs standardisées.
        - Supprime les valeurs aberrantes.

        Retour :
            pd.DataFrame : Dataset avec une colonne 'serving_size_clean' nettoyée.
        """
        # Expression régulière pour détecter la quantité et l'unité
        # Permet de détecter les g, kg, ml, l, etc.
        pattern = r'(\d+[\.,]?\d*)\s*(g|kg|kilogram|kilograms|l|litre|litres|cl|ml)'

        def extract_serving(value):
            # Trouver tous les nombres suivis d'unités dans la colonne
            matches = re.findall(pattern, str(value))
            if not matches:
                return None

            # Prendre le premier nombre trouvé
            quantity, unit = matches[0]

            # Remplacer les virgules par des points et convertir en float
            quantity = quantity.replace(",", ".")
            try:
                quantity = float(quantity)
            except ValueError:
                return None  # Si conversion impossible

            # Convertir les unités en standard
            unit = unit.lower()
            if unit in ["kg", "kilogram", "kilograms"]:
                quantity *= 1000
                unit = "g"
            elif unit in ["mg"]:
                quantity /= 1000
                unit = "g"
            elif unit in ["l", "litre", "litres"]:
                quantity *= 1000
                unit = "ml"
            elif unit in ["cl"]:
                quantity *= 10
                unit = "ml"
            elif unit == "":
                unit = "g"

            # Éliminer les valeurs aberrantes
            if quantity <= 0 or quantity > 10000:
                return None

            return f"{quantity}"

        # Appliquer la transformation
        self.df[nom_colonne] = self.df['serving_size'].apply(extract_serving)
        return self.df

    def knn_imputer(self, n_neighbors):
        """
        Impute les valeurs manquantes d'une colonne spécifique en utilisant KNN Imputer,
        en se basant sur les colonnes corrélées.

        :param n_neighbors: Nombre de voisins à utiliser pour l'imputation

        Retour :
            pd.DataFrame : Le DataFrame avec les valeurs imputées.
        """
        # Calcul de la matrice de corrélation
        df_numeric = self.df.select_dtypes(include=['float64', 'int64'])

        # Appliquer le KNN Imputer uniquement sur les colonnes sélectionnées
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = imputer.fit_transform(df_numeric)

        # Créer un DataFrame avec les données imputées
        df_subset_imputed = pd.DataFrame(imputed_data, columns=df_numeric.columns)
        return df_subset_imputed


# Example usage of the Tache1 class methods

# Initialize the Tache1 class
tache = Tache1()

# Remove irrelevant columns
print("Removing irrelevant columns...")
df_cleaned = tache.remove_irrelevant_columns()
print(df_cleaned.head())

# Remove high NaN columns
print("\nRemoving high NaN columns...")
df_no_high_nan = tache.remove_high_nan_columns(threshold=90)
print(df_no_high_nan.head())

# Remove duplicates
print("\nRemoving duplicates...")
df_no_duplicates = tache.remove_duplicates()
print(df_no_duplicates.head())

# Get column count
print("\nGetting column count...")
column_count = tache.get_column_count()
print(f"Total number of columns: {column_count}")

# Clean 'serving_size' column
print("\nCleaning 'serving_size' column...")
df_cleaned_serving_size = tache.clean_column('serving_size_clean')
print(df_cleaned_serving_size[['serving_size', 'serving_size_clean']].head())

# Impute missing values using KNN Imputer
print("\nImputing missing values using KNN Imputer...")
df_imputed = tache.knn_imputer(n_neighbors=5)
print(df_imputed.head())

# Print specific columns after imputation
print("\nPrinting specific columns after imputation...")
print(df_imputed[["energy-kcal_100g", "energy_100g"]].head(300))
