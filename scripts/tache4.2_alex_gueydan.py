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

try:
    from tache1_alex_gueydan import clean_data_set
    from scipy.stats import chi2_contingency, shapiro
except ImportError as e:
    print(f"Erreur d'import : {e}")
    clean_data_set = None


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
        if clean_data_set:
            self.df = clean_data_set()
        else:
            self.df = pd.DataFrame()  # Dataset vide en cas d'erreur d'import

    def heatmap(self, threshold):
        """
        Affiche une heatmap des corrélations entre les variables du dataset.

        Arguments :
            threshold (float) : Seuil de corrélation pour filtrer les valeurs affichées.
        """
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"])
        correlation_matrix = numeric_cols.corr().abs()

        correlation_matrix = correlation_matrix[(correlation_matrix.abs() > threshold)]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix, annot=True, cmap="coolwarm", center=0, linewidths=0.5
        )
        plt.title(f"Correlation Heatmap (threshold {threshold})")
        plt.show()

    def spearman_correlation(self, threshold):
        """
        Calcule et affiche la corrélation de Spearman entre toutes les variables.

        Arguments :
            threshold (float) : Seuil de corrélation pour filtrer les valeurs affichées.
        """
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"])
        correlation_matrix = numeric_cols.corr(method="spearman").abs()

        correlation_matrix = correlation_matrix[
            (correlation_matrix.abs() > threshold) & (correlation_matrix != 1)
        ]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix, annot=True, cmap="coolwarm", center=0, linewidths=0.5
        )
        plt.title(f"Spearman Correlation Heatmap (threshold {threshold})")
        plt.show()

    def chi2_test(self, colonne1, colonne2):
        """
        Effectue un test du chi-carré pour vérifier l'indépendance entre deux variables catégorielles.

        Arguments :
            colonne1 (str) : Nom de la première variable.
            colonne2 (str) : Nom de la seconde variable.
        """
        _, p_value, _, _ = chi2_contingency(pd.crosstab(self.df[colonne1], self.df[colonne2]))

        if p_value < 0.05:
            print(f"Les variables {colonne1} et {colonne2} sont dépendantes.")
        else:
            print(f"Les variables {colonne1} et {colonne2} sont indépendantes.")

    def find_highly_correlated_features(self, threshold):
        """
        Identifie les variables ayant une corrélation élevée et qui pourraient être redondantes.

        Arguments :
            threshold (float) : Seuil de corrélation pour identifier les variables redondantes.

        Retour :
            redundant_features (list) : Liste des variables redondantes.
        """
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"])
        corr_matrix = numeric_cols.corr().abs()

        np.fill_diagonal(corr_matrix.values, 0)

        redundant_features = [
            column for column in corr_matrix.columns if any(corr_matrix[column] > threshold)
        ]

        print(f"Variables redondantes : {redundant_features}")

        return redundant_features

    def plot_polynomial_relationship(self, x_col, y_col, degree=2):
        """
        Affiche une relation polynomiale entre deux variables.

        Arguments :
            x_col (str) : Colonne de l'axe des abscisses.
            y_col (str) : Colonne de l'axe des ordonnées.
            degree (int) : Degré du polynôme à ajuster.
        """
        plt.figure(figsize=(8, 5))
        sns.regplot(
            x=self.df[x_col],
            y=self.df[y_col],
            order=degree,
            scatter_kws={"s": 10},
            line_kws={"color": "red"},
        )
        plt.title(f"Relation polynomiale (degré {degree}) entre {x_col} et {y_col}")
        plt.show()

    def check_normality(self, column):
        """
        Vérifie si une variable suit une distribution normale avec le test de Shapiro-Wilk.

        Arguments :
            column (str) : Nom de la colonne à tester.
        """
        _, p_value = shapiro(self.df[column].dropna())

        if p_value > 0.05:
            print(f"La variable {column} suit une distribution normale.")
        else:
            print(f"La variable {column} ne suit pas une distribution normale.")


# 🔹 Utilisation :
tache = Tache42()
# tache.heatmap(0.7)
# tache.spearman_correlation(0.3)
# tache.find_highly_correlated_features(0.7)
# tache.plot_polynomial_relationship("sodium_100g", "energy_100g", degree=2)
# tache.chi2_test("nutrition_grade", "nova_group")
# tache.check_normality("energy_100g")
