"""
Module pour l'analyse avancée du dataset Open Food Facts.

Contient la classe `Tache42` permettant de :
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
from tache1_alex_gueydan import clean_data_set
from scipy.stats import chi2_contingency, shapiro


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
        Initialise la classe et charge un échantillon du dataset Open Food Facts.
        
        Attributs :
            df (DataFrame) : Un DataFrame contenant l'échantillon du dataset.
        """
        # Forcer pandas à afficher toutes les colonnes et lignes
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)

        # Chargement du dataset
        self.df = clean_data_set()

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



# 🔹 Utilisation :
tache = Tache42()
#tache.heatmap(0.7)
# tache.spearman_correlation(0.3)
#tache.find_highly_correlated_features(0.7)
#tache.plot_polynomial_relationship("sodium_100g", "energy_100g", degree=2)
# tache.test_polynomial_features(["sodium_100g", "fat_100g"], "energy_100g", degree=2)
# tache.find_highly_correlated_features(threshold=0.9)
tache.decision_tree_analysis("energy-kcal_100g")
