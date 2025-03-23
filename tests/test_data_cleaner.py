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
    from typing import Dict, Tuple

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import pandas as pd
    from scripts.data_cleaner import clean_dataset
except ImportError as e:
    print(f"Erreur lors de l'importation des modules : {e}")


def load_sample_data() -> pd.DataFrame:
    """
    Charge un échantillon du dataset OpenFoodFacts.
    
    Returns:
        DataFrame contenant l'échantillon de données
    """
    path = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
    try:
        df_sample = pd.read_csv(path, nrows=500, sep='\t', encoding="utf-8")
        print(
            f"Échantillon chargé avec succès : {df_sample.shape[0]} lignes et "
            f"{df_sample.shape[1]} colonnes"
        )
        return df_sample
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"Erreur lors du chargement des données : {e}")
        raise


def calculate_missing_percentage(df: pd.DataFrame) -> float:
    """
    Calcule le pourcentage de valeurs manquantes dans un DataFrame.
    
    Args:
        df: DataFrame à analyser
        
    Returns:
        Pourcentage de valeurs manquantes
    """
    return df.isna().sum().sum() / (df.shape[0] * df.shape[1])


def compare_imputation_methods(
        df_sample: pd.DataFrame) -> Dict[str, Tuple[float, float, pd.DataFrame]]:
    """
    Compare les méthodes d'imputation simple et KNN.
    
    Args:
        df_sample: DataFrame d'entrée
        
    Returns:
        Dictionnaire contenant les résultats pour chaque méthode
    """
    results = {}

    # Test avec la méthode d'imputation simple
    print("\nNettoyage du dataset avec la méthode d'imputation simple...")
    start_time = time.time()
    df_cleaned = clean_dataset(df_sample, imputation_method='simple')
    time_taken = time.time() - start_time
    missing_percentage = calculate_missing_percentage(df_cleaned)
    results['simple'] = (time_taken, missing_percentage, df_cleaned)

    # Test avec la méthode d'imputation KNN
    print("\nNettoyage du dataset avec la méthode d'imputation KNN...")
    start_time = time.time()
    df_cleaned = clean_dataset(df_sample, imputation_method='knn', n_neighbors=5)
    time_taken = time.time() - start_time
    missing_percentage = calculate_missing_percentage(df_cleaned)
    results['knn'] = (time_taken, missing_percentage, df_cleaned)

    return results


def analyze_numeric_differences(df_sample: pd.DataFrame, results: Dict[str, pd.DataFrame]) -> None:
    """
    Analyse les différences entre les méthodes d'imputation pour les colonnes numériques.
    
    Args:
        df_sample: DataFrame original
        results: Dictionnaire contenant les résultats des méthodes d'imputation
    """
    numeric_cols = results['simple'].select_dtypes(include=['number']).columns
    if not numeric_cols:
        return

    # Calculer les différences
    numeric_diff = (results['knn'][numeric_cols] - results['simple'][numeric_cols]).abs()
    mean_diff_by_col = numeric_diff.mean()
    overall_mean_diff = mean_diff_by_col.mean()

    print(f"\nDifférence moyenne des valeurs imputées (colonnes numériques) : "
          f"{overall_mean_diff:.4f}")

    # Afficher les colonnes avec les plus grandes différences
    print("\nTop 5 colonnes avec les plus grandes différences d'imputation :")
    top_diff_cols = mean_diff_by_col.sort_values(ascending=False).head(5)
    for col, diff in top_diff_cols.items():
        print(f"  - {col}: {diff:.4f}")

    # Afficher un échantillon des différences
    display_sample_differences(df_sample, results, numeric_cols)


def display_sample_differences(
        df_sample: pd.DataFrame,
        results: Dict[str, pd.DataFrame],
        numeric_cols: pd.Index
) -> None:
    """
    Affiche un échantillon des différences d'imputation.
    
    Args:
        df_sample: DataFrame original
        results: Dictionnaire contenant les résultats des méthodes d'imputation
        numeric_cols: Colonnes numériques à analyser
    """
    sample_cols = min(3, len(numeric_cols))
    if sample_cols <= 0:
        return

    print("\nExemple de différences d'imputation pour quelques colonnes numériques:")
    for col in numeric_cols[:sample_cols]:
        mask = df_sample[col].isna()
        if not mask.any():
            continue

        print(f"\nColonne: {col}")
        simple_vals = results['simple'].loc[mask, col].head(3)
        knn_vals = results['knn'].loc[mask, col].head(3)
        diff_vals = (knn_vals - simple_vals).abs()

        comparison = pd.DataFrame({
            'Simple': simple_vals,
            'KNN': knn_vals,
            'Différence': diff_vals
        })
        print(comparison)


def main():
    """
    Test de la fonction clean_dataset avec un petit échantillon de données.
    Compare les méthodes d'imputation simple et KNN.
    """
    # Suppression des avertissements
    warnings.filterwarnings('ignore')

    try:
        # Charger l'échantillon de données
        print("Chargement de l'échantillon de données...")
        df_sample = load_sample_data()

        # Calculer le pourcentage de valeurs manquantes avant nettoyage
        missing_before = calculate_missing_percentage(df_sample)

        # Comparer les méthodes d'imputation
        results = compare_imputation_methods(df_sample)
        time_simple, missing_after_simple, df_cleaned_simple = results['simple']
        time_knn, missing_after_knn, df_cleaned_knn = results['knn']

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

        # Analyser les différences entre les méthodes
        print("\n=== Comparaison des résultats d'imputation ===")
        analyze_numeric_differences(df_sample, {
            'simple': df_cleaned_simple,
            'knn': df_cleaned_knn
        })

        # Afficher un aperçu des données nettoyées
        print("\nAperçu des données nettoyées (Simple) :")
        print(df_cleaned_simple.head(2))
        print("\nAperçu des données nettoyées (KNN) :")
        print(df_cleaned_knn.head(2))

    except Exception as e:
        print(f"Une erreur est survenue lors de l'exécution des tests : {e}")
        raise


if __name__ == "__main__":
    main()
