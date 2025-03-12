"""
Module pour l'analyse avancée du dataset Open Food Facts.

Contient la classe `Tache42` permettant de :
- Analyser les corrélations entre les variables (Pearson, Spearman).
- Identifier les variables redondantes.
- Tester l'indépendance entre variables catégorielles avec le chi-carré.
- Vérifier la normalité des distributions.
- Visualiser les relations entre variables via des heatmaps et des modèles polynomiaux.

Elle contient aussi des fonction de la Tache 1, dans l'objectif d'amélioréer les performances du dataset de travail
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, shapiro
from sklearn.impute import KNNImputer


class Tache42:
    """
    Classe permettant d'analyser et de manipuler le dataset Open Food Facts.

    Cette classe propose plusieurs méthodes pour :
    - Visualiser les corrélations entre les variables.
    - Effectuer des tests statistiques pour examiner les relations entre variables.
    - Tester la normalité des variables.
    - Identifier les variables redondantes à partir de leur corrélation.
    """

    def __init__(self):
            """
            Initialise la classe et charge un échantillon de 10 000 lignes du dataset Open Food Facts
            en utilisant le séparateur tabulation.

            Attributs :
                df (DataFrame): Un DataFrame Pandas contenant l'échantillon du dataset.
            """
            #Forcage de pandas a afficher autant de caractère qu'il peut sur une ligne
            pd.set_option("display.max_columns", None)

            #Meme chose avec le nombre de lignes
            pd.set_option("display.max_rows", None)

            #Chargement du dataset, en prenant un échantillon de 10000 lignes pour ne pas trop surcharger
            self.df = pd.read_csv("datasets/en.openfoodfacts.org.products.csv", sep="\t", on_bad_lines='skip', nrows=10000, low_memory=False)

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

    def knn_imputer(self, n_neighbors):
        """
        Impute les valeurs manquantes d'une colonne spécifique en utilisant KNN Imputer,
        en se basant sur les colonnes corrélées.
        
        :param n_neighbors: Nombre de voisins à utiliser pour l'imputation
        """
        # Calcul de la matrice de corrélation
        df_numeric = self.df.select_dtypes(include=['float64', 'int64'])

        # Appliquer le KNN Imputer uniquement sur les colonnes sélectionnées
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = imputer.fit_transform(df_numeric)
        df_subset_imputed = pd.DataFrame(imputed_data, columns=df_numeric.columns)

        return df_subset_imputed

    def heatmap(self, threshold):
        """
        Affiche une heatmap des corrélations entre les variables du dataset,
        en filtrant celles qui sont supérieures ou inférieures au seuil spécifié.

        Arguments :
            threshold (float) : Le seuil de corrélation pour filtrer les valeurs.
                                 Les corrélations supérieures ou inférieures à ce seuil seront affichées.
        """
        # Calcul de la matrice de corrélation pour les colonnes numériques
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64'])
        correlation_matrix = numeric_cols.corr().abs()

        # Appliquer le seuil pour filtrer les corrélations
        correlation_matrix = correlation_matrix[(correlation_matrix.abs() > threshold)]

        # Créer la heatmap avec la matrice filtrée
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
            spearman_corr (DataFrame) : La matrice de corrélation de Spearman.
        """
        # Calcul de la matrice de corrélation de Spearman
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64'])
        correlation_matrix = numeric_cols.corr(method="spearman").abs()

        # Appliquer le seuil pour filtrer les corrélations
        correlation_matrix = correlation_matrix[(correlation_matrix.abs() > threshold) & (correlation_matrix != 1)]

        # Créer la heatmap avec la matrice filtrée
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
        # Effectuer le test Chi-Carré
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

        # Créer une version de la matrice sans la diagonale (corrélation avec soi-même)
        np.fill_diagonal(corr_matrix.values, 0)

        # Identifier les colonnes ayant une corrélation supérieure au seuil
        redundant_features = [
            column for column in corr_matrix.columns if any(corr_matrix[column] > threshold)
        ]

        print(f"Variables redondantes : {redundant_features}")

        return redundant_features

    def plot_polynomial_relationship(self, x_col, y_col, degree=2):
        """
        Affiche une relation polynomiale entre deux variables.

        Arguments :
            x_col (str) : Le nom de la colonne sur l'axe des abscisses.
            y_col (str) : Le nom de la colonne sur l'axe des ordonnées.
            degree (int) : Le degré du polynôme à ajuster.
        """
        plt.figure(figsize=(8, 5))
        sns.regplot(x=self.df[x_col], y=self.df[y_col], order=degree, scatter_kws={"s": 10}, line_kws={"color": "red"})
        plt.title(f"Relation polynomiale (degré {degree}) entre {x_col} et {y_col}")
        plt.show()

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



# tache = Tache42()
# tache.remove_irrelevant_columns()
# tache.remove_high_nan_columns()
# tache.remove_duplicates()
# tache.heatmap(0.5)
# tache.spearman_correlation(0.3)
# tache.find_highly_correlated_features(0.7)
# tache.plot_polynomial_relationship("sodium_100g", "energy_100g", degree=2)
# tache.test_polynomial_features(["sodium_100g", "fat_100g"], "energy_100g", degree=2)
# tache.find_highly_correlated_features(threshold=0.9)
