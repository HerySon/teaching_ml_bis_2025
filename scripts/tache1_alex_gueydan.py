"""
Module pour la manipulation du dataset Open Food Facts.

Contient la classe `Tache1` permettant de :
- Supprimer les colonnes non pertinentes.
- Supprimer les doublons.
- Supprimer les colonnes avec trop de valeurs manquantes.
- Nettoyer des colonnes spécifiques.
- Imputer les valeurs manquantes avec l'algorithme KNN.
"""

import pandas as pd
import re
from sklearn.impute import KNNImputer


class Tache1:
    """
    Classe permettant de manipuler et traiter un dataset Open Food Facts.
    Elle inclut des méthodes pour supprimer les colonnes non pertinentes, supprimer les doublons,
    filtrer les colonnes avec un pourcentage élevé de valeurs manquantes, nettoyer des colonnes spécifiques,
    et imputer les valeurs manquantes en utilisant KNN.
    """

    def __init__(self, file_path="datasets/en.openfoodfacts.org.products.csv", sample_size=10000):
        """
        Initialise la classe et charge un échantillon du dataset Open Food Facts.

        Arguments :
            file_path (str) : Chemin du fichier CSV à charger.
            sample_size (int) : Nombre de lignes à charger (par défaut : 10 000).

        Attributs :
            df (DataFrame) : Un DataFrame contenant l'échantillon du dataset.
        """
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)

        self.df = pd.read_csv(
            file_path,
            sep="\t",
            on_bad_lines="skip",
            nrows=sample_size,
            low_memory=False,
        )

    def remove_irrelevant_columns(self):
        """
        Supprime les colonnes non pertinentes pour l'analyse.

        Retour :
            pd.DataFrame : Le DataFrame nettoyé.
        """
        columns_to_drop = [
            "code", "url", "creator", "created_t", "created_datetime", "last_modified_t", "last_modified_datetime",
            "packaging", "packaging_tags", "brands_tags", "categories_tags", "categories_fr", "origins_tags",
            "manufacturing_places", "manufacturing_places_tags", "labels_tags", "labels_fr", "emb_codes",
            "emb_codes_tags", "first_packaging_code_geo", "cities", "cities_tags", "purchase_places", "countries_tags",
            "countries_fr", "image_ingredients_url", "image_ingredients_small_url", "image_nutrition_url",
            "image_nutrition_small_url", "image_small_url", "image_url", "last_updated_t", "last_updated_datetime",
            "last_modified_by", "last_image_t"
        ]
        self.df.drop(columns=[col for col in columns_to_drop if col in self.df.columns], errors="ignore", inplace=True)
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

    def get_column_count(self):
        """Retourne le nombre total de colonnes du DataFrame."""
        return self.df.shape[1]

    def clean_column(self, nom_colonne):
        """
        Nettoie la colonne spécifiée (par défaut 'serving_size') :
        - Extrait la quantité principale en grammes ou millilitres.
        - Convertit les unités en valeurs standardisées.
        - Supprime les valeurs aberrantes.

        Arguments :
            nom_colonne (str) : Nom de la colonne à nettoyer.

        Retour :
            pd.DataFrame : Dataset avec la colonne nettoyée.
        """
        pattern = r"(\d+[\.,]?\d*)\s*(g|kg|kilogram|kilograms|l|litre|litres|cl|ml)"

        def extract_serving(value):
            matches = re.findall(pattern, str(value))
            if not matches:
                return None
            quantity, unit = matches[0]
            quantity = quantity.replace(",", ".")
            try:
                quantity = float(quantity)
            except ValueError:
                return None
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
            if quantity <= 0 or quantity > 10000:
                return None
            return f"{quantity}"

        self.df[nom_colonne] = self.df["serving_size"].apply(extract_serving)
        return self.df

    def knn_impute_column(self, column, n_neighbors, threshold):
        """
        Impute les valeurs manquantes d'une colonne spécifique en utilisant KNN Imputer,
        en se basant sur les colonnes corrélées.

        Arguments :
            column (str) : Nom de la colonne à imputer.
            n_neighbors (int) : Nombre de voisins à utiliser pour l'imputation.
            threshold (float) : Seuil de corrélation pour sélectionner les colonnes utiles.

        Retour :
            pd.DataFrame : Le DataFrame avec la colonne imputée.
        """
        df_numeric = self.df.select_dtypes(include=["float64", "int64"])
        corr_matrix = df_numeric.corr()
        correlated_cols = corr_matrix.index[corr_matrix[column].abs() >= threshold].tolist()

        if column not in correlated_cols:
            correlated_cols.append(column)

        df_subset = self.df[correlated_cols]
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_subset_imputed = pd.DataFrame(imputer.fit_transform(df_subset), columns=correlated_cols)
        self.df[column] = df_subset_imputed[column]
        return self.df


def clean_data_set(file_path="datasets/en.openfoodfacts.org.products.csv", sample_size=10000):
    """
    Fonction pour nettoyer le dataset complet.
    
    Arguments :
        file_path (str) : Chemin du fichier CSV à charger.
        sample_size (int) : Nombre de lignes à charger (par défaut : 10 000).
        
    Retour :
        pd.DataFrame : Le DataFrame nettoyé.
    """
    tache = Tache1(file_path, sample_size)
    tache.remove_irrelevant_columns()
    tache.remove_high_nan_columns()
    tache.remove_duplicates()
    return tache.df


if __name__ == "__main__":
    tache = Tache1()
    df_clean = tache.remove_irrelevant_columns()
    #df_clean = tache.clean_column("serving_size")
    #print(df_clean[['serving_size', 'serving_size_clean']].head(200))
    df_imputed = tache.knn_impute_column("energy-kcal_100g", 5, 0.8)
