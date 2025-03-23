"""
Module de test pour la détection des valeurs aberrantes dans le dataset OpenFoodFacts.
Ce module permet de comparer différentes méthodes de détection des valeurs aberrantes :
- Méthode de Tukey (IQR)
- Z-Score
- Enveloppe elliptique
- Isolation Forest
- Local Outlier Factor (LOF)

Le module génère des visualisations comparatives et des statistiques pour chaque méthode.
"""
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sys
    import os
except ImportError as e:
    print(f"Erreur lors de l'importation des modules : {e}")

# Configuration de matplotlib pour l'affichage interactif
plt.ion()  # Mode interactif

# Ajout du répertoire parent au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.outlier_detection import (
    detect_outliers_tukey,
    detect_outliers_zscore,
    detect_outliers_elliptic,
    detect_outliers_isolation_forest,
    detect_outliers_lof
)


def load_data():
    """Charge les données d'OpenFoodFacts"""
    path = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
    df = pd.read_csv(path, nrows=10000, sep='\t', encoding="utf-8")
    return df


def plot_outliers_comparison(data, column, methods_results):
    """Crée un graphique comparant les différentes méthodes de détection"""
    # Création d'une nouvelle figure
    fig = plt.figure(figsize=(15, 10))

    # Création d'un histogramme des données
    plt.subplot(2, 1, 1)
    # Conversion de la Series en DataFrame pour seaborn
    df_plot = pd.DataFrame({column: data})
    sns.histplot(data=df_plot, x=column, bins=50)
    plt.title(f"Distribution des {column}")

    # Création d'un graphique des valeurs aberrantes détectées
    plt.subplot(2, 1, 2)
    for method_name, (outliers, _) in methods_results.items():
        plt.scatter(data[~outliers],
                    np.zeros_like(data[~outliers]) + list(methods_results.keys()).index(method_name),
                    alpha=0.5, label=f"Normal ({method_name})")
        plt.scatter(data[outliers],
                    np.zeros_like(data[outliers]) + list(methods_results.keys()).index(method_name),
                    alpha=0.5, label=f"Outlier ({method_name})")

    plt.yticks(range(len(methods_results)), list(methods_results.keys()))
    plt.title("Comparaison des valeurs aberrantes détectées")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Sauvegarde de la figure
    plt.savefig(f"results/outliers_comparison_{column}.png", bbox_inches='tight', dpi=300)

    # Affichage de la figure
    plt.show(block=True)  # Force l'affichage bloquant

    # Attendre que l'utilisateur ferme la fenêtre
    plt.waitforbuttonpress()

    # Fermeture de la figure
    plt.close(fig)


def print_summary(methods_results):
    """Affiche un résumé des résultats pour chaque méthode"""
    print("\nRésumé des détections de valeurs aberrantes :")
    print("-" * 50)
    for method_name, (outliers, stats) in methods_results.items():
        n_outliers = np.sum(outliers)
        percentage = (n_outliers / len(outliers)) * 100
        print(f"\nMéthode : {method_name}")
        print(f"Nombre de valeurs aberrantes : {n_outliers}")
        print(f"Pourcentage : {percentage:.2f}%")
        print("-" * 30)


def main():
    # Chargement des données
    print("Chargement des données...")
    df = load_data()

    # Sélection des colonnes numériques pertinentes
    numeric_columns = ['energy-kcal_100g', 'fat_100g', 'proteins_100g', 'carbohydrates_100g']

    for column in numeric_columns:
        print(f"\nAnalyse de la colonne : {column}")

        # Nettoyage des données
        data = df[column].dropna()

        # Application des différentes méthodes
        methods_results = {
            'Tukey': detect_outliers_tukey(data),
            'Z-Score': detect_outliers_zscore(data),
            'Elliptic Envelope': detect_outliers_elliptic(data),
            'Isolation Forest': detect_outliers_isolation_forest(data),
            'LOF': detect_outliers_lof(data)
        }

        # Affichage des résultats
        print_summary(methods_results)

        # Création du graphique de comparaison
        plot_outliers_comparison(data, column, methods_results)

        # Pause pour permettre de voir le graphique
        input("Appuyez sur Entrée pour passer à la colonne suivante...")


if __name__ == "__main__":
    main()
