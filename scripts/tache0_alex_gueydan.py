"""
Module pour la manipulation du dataset Open Food Facts.

Contient la classe `Tache0` permettant de :
- Sélectionner différentes colonnes (numériques, ordinales, non ordinales).
- Optimiser la mémoire avec un downcasting des nombres.
- Filtrer les variables catégorielles selon un seuil.
"""

import pandas as pd


class Tache0:
    """
    Classe permettant de manipuler et traiter un dataset Open Food Facts.
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

        self._df = pd.read_csv(
            file_path,
            sep="\t",
            on_bad_lines="skip",
            nrows=sample_size,
            low_memory=False,
        )

    @property
    def df(self):
        """Retourne le DataFrame chargé."""
        return self._df

    def select_numeric_columns(self):
        """
        Sélectionne et retourne les colonnes numériques (int64, float64).

        Retour :
            DataFrame : Un DataFrame contenant uniquement les colonnes numériques.
        """
        return self.df.select_dtypes(include=["int64", "float64"])

    def select_ordinal_columns(self, cols):
        """
        Sélectionne les colonnes ordinales spécifiées.

        Arguments :
            cols (list) : Liste des noms de colonnes ordinales.

        Retour :
            DataFrame : Un DataFrame contenant uniquement les colonnes ordinales spécifiées.
        """
        if all(col in self.df.columns for col in cols):
            return self.df[cols]
        return None

    def select_non_ordinal_columns(self, ordinal_cols):
        """
        Sélectionne les colonnes non ordinales (catégoriques).

        Arguments :
            ordinal_cols (list) : Liste des colonnes ordinales à exclure.

        Retour :
            DataFrame : Un DataFrame contenant uniquement les colonnes non ordinales.
        """
        categorical_cols = self.df.select_dtypes(include=["object"]).columns
        return self.df[[col for col in categorical_cols if col not in ordinal_cols]]

    def select_non_ordinal_columns_without_date(self, ordinal_cols):
        """
        Sélectionne les colonnes non ordinales, en excluant celles qui contiennent "date".

        Arguments :
            ordinal_cols (list) : Liste des colonnes ordinales à exclure.

        Retour :
            DataFrame : Un DataFrame avec uniquement les colonnes non ordinales sans "date".
        """
        categorical_cols = self.df.select_dtypes(include=["object"]).columns
        return self.df[
            [col for col in categorical_cols if col not in ordinal_cols and "date" not in col.lower()]
        ]

    def downcast_numerics(self):
        """
        Réduit l'utilisation de mémoire en convertissant les nombres en types plus petits.

        Retour :
            DataFrame : Le DataFrame avec les colonnes numériques optimisées.
        """
        for col in self.df.select_dtypes(include=["int64", "float64"]).columns:
            if self.df[col].dtype == "int64":
                self._df[col] = pd.to_numeric(self.df[col], downcast="integer")
            else:
                self._df[col] = pd.to_numeric(self.df[col], downcast="float")
        return self.df

    def numbers_variables(self, threshold):
        """
        Sélectionne les colonnes catégorielles ayant un nombre de catégories unique inférieur ou égal à `threshold`.

        Arguments :
            threshold (int) : Nombre maximum de catégories uniques autorisé.

        Retour :
            DataFrame : Le DataFrame avec les colonnes filtrées.
        """
        categorical_cols = self.df.select_dtypes(include=["object"]).columns
        filtered_cols = [col for col in categorical_cols if self.df[col].nunique() <= threshold]
        return self.df[filtered_cols]

    def unique_categories_count(self, column_name):
        """
        Retourne le nombre de valeurs uniques d'une colonne spécifiée.

        Arguments :
            column_name (str) : Nom de la colonne.

        Retour :
            int : Nombre de valeurs uniques dans la colonne.
        """
        return self.df[column_name].nunique() if column_name in self.df.columns else None


# Example usage of the Tache0 class methods

# Initialize the Tache0 class
tache = Tache0()

# Select numeric columns
print("Selecting numeric columns...")
numeric_df = tache.select_numeric_columns()
print(numeric_df.head())

# Select ordinal columns
print("\nSelecting ordinal columns...")
ordinal_cols = ['column1', 'column2']  # Replace with actual ordinal column names
ordinal_df = tache.select_ordinal_columns(ordinal_cols)
print(ordinal_df.head() if ordinal_df is not None else "Some columns are missing.")

# Select non-ordinal columns
print("\nSelecting non-ordinal columns...")
non_ordinal_df = tache.select_non_ordinal_columns(ordinal_cols)
print(non_ordinal_df.head())

# Select non-ordinal columns without date
print("\nSelecting non-ordinal columns without date...")
non_ordinal_no_date_df = tache.select_non_ordinal_columns_without_date(ordinal_cols)
print(non_ordinal_no_date_df.head())

# Downcast numerics
print("\nDowncasting numerics...")
downcasted_df = tache.downcast_numerics()
print(downcasted_df.head())

# Select categorical columns with a limited number of unique values
print("\nSelecting categorical columns with <= 100 unique values...")
category_df = tache.numbers_variables(100)
print(category_df.head())

# Count unique categories in a specific column
print("\nCounting unique categories in a specific column...")
unique_count = tache.unique_categories_count('column_name')  # Replace with actual column name
print(f"Unique categories count: {unique_count}")
