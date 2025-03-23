"""
Module de test pour le module data_cleaner.

Ce module teste les fonctionnalités de nettoyage de données et d'imputation
fournies par le module data_cleaner, en comparant les méthodes d'imputation
simple et KNN sur un échantillon de données OpenFoodFacts.
"""
try:
    import os
    import sys
    import time
    import warnings

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import pandas as pd
    from scripts.data_cleaner import clean_dataset
except ImportError as e:
    print(f"Erreur lors de l'importation des modules : {e}")


def main():
    """
    Test de la fonction clean_dataset avec un petit échantillon de données.
    Compare les méthodes d'imputation simple et KNN.
    """
    # Suppression des avertissements
    warnings.filterwarnings('ignore')

    # Charger un petit échantillon du dataset pour tester
    print("Chargement de l'échantillon de données...")
    try:
        path = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
        df_sample = pd.read_csv(path, nrows=500, sep='\t', encoding="utf-8")
        print(f"Échantillon chargé avec succès : {df_sample.shape[0]} lignes et "
              f"{df_sample.shape[1]} colonnes")
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return

    # Calculer le pourcentage de valeurs manquantes avant nettoyage
    missing_before = df_sample.isna().sum().sum() / (df_sample.shape[0] * df_sample.shape[1])

    # Test avec la méthode d'imputation simple
    print("\nNettoyage du dataset avec la méthode d'imputation simple...")
    start_time_simple = time.time()
    df_cleaned_simple = clean_dataset(df_sample, imputation_method='simple')
    end_time_simple = time.time()
    time_simple = end_time_simple - start_time_simple

    # Calculer le pourcentage de valeurs manquantes après le nettoyage (méthode simple)
    missing_after_simple = (df_cleaned_simple.isna().sum().sum() /
                            (df_cleaned_simple.shape[0] * df_cleaned_simple.shape[1]))

    # Test avec la méthode d'imputation KNN
    print("\nNettoyage du dataset avec la méthode d'imputation KNN...")
    start_time_knn = time.time()
    df_cleaned_knn = clean_dataset(df_sample, imputation_method='knn', n_neighbors=5)
    end_time_knn = time.time()
    time_knn = end_time_knn - start_time_knn

    # Calculer le pourcentage de valeurs manquantes après le nettoyage (méthode KNN)
    missing_after_knn = (df_cleaned_knn.isna().sum().sum() /
                         (df_cleaned_knn.shape[0] * df_cleaned_knn.shape[1]))

    # Afficher les résultats de comparaison
    print("\n=== Résultats de comparaison des méthodes d'imputation ===")
    print(f"Taille originale : {df_sample.shape[0]} lignes x {df_sample.shape[1]} colonnes")
    print(f"Taille après nettoyage (simple) : {df_cleaned_simple.shape[0]} lignes x "
          f"{df_cleaned_simple.shape[1]} colonnes")
    print(f"Taille après nettoyage (KNN) : {df_cleaned_knn.shape[0]} lignes x "
          f"{df_cleaned_knn.shape[1]} colonnes")
    print(f"Pourcentage de valeurs manquantes avant : {missing_before:.2%}")
    print(f"Pourcentage de valeurs manquantes après (simple) : {missing_after_simple:.2%}")
    print(f"Pourcentage de valeurs manquantes après (KNN) : {missing_after_knn:.2%}")
    print(f"Temps d'exécution (simple) : {time_simple:.2f} secondes")
    print(f"Temps d'exécution (KNN) : {time_knn:.2f} secondes")
    print(f"Rapport de temps KNN/Simple : {time_knn / time_simple:.2f}x")

    # Vérifier les différences entre les deux méthodes
    print("\n=== Comparaison des résultats d'imputation ===")

    # Comparer uniquement les colonnes numériques (où KNN est appliqué)
    numeric_cols = df_cleaned_simple.select_dtypes(include=['number']).columns

    if len(numeric_cols) > 0:
        # Calculer la différence moyenne entre les méthodes pour les colonnes numériques
        numeric_diff = (df_cleaned_knn[numeric_cols] - df_cleaned_simple[numeric_cols]).abs()
        mean_diff_by_col = numeric_diff.mean()

        # Calculer la différence moyenne globale
        overall_mean_diff = mean_diff_by_col.mean()
        print(f"Différence moyenne des valeurs imputées (colonnes numériques) : "
              f"{overall_mean_diff:.4f}")

        # Afficher les colonnes avec les plus grandes différences
        print("\nTop 5 colonnes avec les plus grandes différences d'imputation :")
        top_diff_cols = mean_diff_by_col.sort_values(ascending=False).head(5)
        for col, diff in top_diff_cols.items():
            print(f"  - {col}: {diff:.4f}")

        # Afficher un échantillon des différences
        sample_cols = min(3, len(numeric_cols))
        if sample_cols > 0:
            print("\nExemple de différences d'imputation pour quelques colonnes numériques:")
            for col in mean_diff_by_col.sort_values(ascending=False).index[:sample_cols]:
                # Sélectionner les lignes où les valeurs étaient initialement manquantes
                mask = df_sample[col].isna()
                # Vérifier s'il y a des valeurs manquantes pour cette colonne
                if mask.any():
                    print(f"\nColonne: {col}")
                    simple_vals = df_cleaned_simple.loc[mask, col].head(3)
                    knn_vals = df_cleaned_knn.loc[mask, col].head(3)
                    diff_vals = (knn_vals - simple_vals).abs()

                    # Afficher les valeurs côte à côte
                    comparison = pd.DataFrame({
                        'Simple': simple_vals,
                        'KNN': knn_vals,
                        'Différence': diff_vals
                    })
                    print(comparison)

    # Afficher un aperçu des données nettoyées avec chaque méthode
    print("\nAperçu des données nettoyées (Simple) :")
    print(df_cleaned_simple.head(2))

    print("\nAperçu des données nettoyées (KNN) :")
    print(df_cleaned_knn.head(2))


if __name__ == "__main__":
    main()
