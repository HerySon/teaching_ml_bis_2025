"""
Module de nettoyage de données pour les datasets d'OpenFoodFacts.

Ce module fournit des fonctions pour nettoyer et imputer les valeurs manquantes
dans les datasets OpenFoodFacts en utilisant différentes méthodes.
"""
try:
    import warnings
    from typing import List, Optional, Literal, Tuple
    import numpy as np
    import pandas as pd
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"Erreur lors de l'importation des modules : {e}")


def clean_dataset(
        df: pd.DataFrame,
        columns_to_drop: Optional[List[str]] = None,
        missing_threshold: float = 0.8,
        imputation_method: Literal['simple', 'knn'] = 'simple',
        n_neighbors: int = 5
) -> pd.DataFrame:
    """
    Nettoie un dataset en effectuant les opérations suivantes:
    - Supprime les colonnes non pertinentes spécifiées
    - Supprime les lignes vides et dupliquées
    - Supprime les lignes avec trop de valeurs manquantes (par défaut > 80%)
    - Impute des valeurs pour les valeurs manquantes restantes
    
    Args:
        df: Le DataFrame pandas à nettoyer
        columns_to_drop: Liste des colonnes à supprimer (si None, utilise une liste prédéfinie)
        missing_threshold: Seuil de valeurs manquantes au-delà duquel une ligne est supprimée
        imputation_method: Méthode d'imputation ('simple' pour médiane/mode, 'knn' pour KNN)
        n_neighbors: Nombre de voisins pour la méthode KNN
    
    Returns:
        DataFrame pandas nettoyé
    """
    # Faire une copie pour éviter de modifier l'original
    df_clean = df.copy()

    # Liste prédéfinie de colonnes à supprimer (non pertinentes pour l'analyse)
    if columns_to_drop is None:
        columns_to_drop = [
            # Colonnes liées à la gestion des données
            'code', 'url', 'creator', 'created_t', 'created_datetime',
            'last_modified_t', 'last_modified_datetime', 'last_modified_by',
            'last_updated_t', 'last_updated_datetime',

            # Colonnes avec beaucoup de valeurs manquantes ou information redondante
            'packaging_text', 'cities', 'allergens_en', 'brand_owner',
            'image_url', 'image_small_url', 'image_ingredients_url',
            'image_ingredients_small_url', 'image_nutrition_url',
            'image_nutrition_small_url'
        ]

    # 1. Supprimer les colonnes non pertinentes
    # Ne supprime que les colonnes qui existent dans le DataFrame
    columns_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
    df_clean = df_clean.drop(columns=columns_to_drop, errors='ignore')

    # 2. Supprimer les lignes vides et dupliquées
    df_clean = df_clean.dropna(how='all')
    df_clean = df_clean.drop_duplicates()

    # 3. Supprimer les lignes avec trop de valeurs manquantes
    missing_percentage = df_clean.isnull().mean(axis=1)
    df_clean = df_clean[missing_percentage < missing_threshold]

    # 4. Imputer les valeurs manquantes selon la méthode choisie
    if imputation_method == 'simple':
        df_clean = _impute_simple(df_clean)
    elif imputation_method == 'knn':
        df_clean = _impute_knn(df_clean, n_neighbors)

    return df_clean


def _impute_simple(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputation simple: médiane pour les colonnes numériques,
    valeur la plus fréquente pour les colonnes catégorielles.
    
    Args:
        df: DataFrame à traiter
        
    Returns:
        DataFrame avec valeurs imputées
    """
    df_clean = df.copy()

    # Pour les colonnes numériques, utiliser la médiane
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        # Vérifier s'il y a des valeurs non-NA pour calculer la médiane
        if df_clean[col].notna().any():
            median_value = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_value)
        else:
            # S'il n'y a que des NA, remplacer par 0
            df_clean[col] = df_clean[col].fillna(0)

    # Pour les colonnes catégorielles, utiliser la valeur la plus fréquente ou "Unknown"
    object_cols = df_clean.select_dtypes(include=['object']).columns
    for col in object_cols:
        if df_clean[col].isna().any():  # Vérifier s'il y a des valeurs manquantes
            if df_clean[col].notna().any():
                # La valeur la plus fréquente, ou "Unknown" s'il n'y en a pas
                most_common = (df_clean[col].mode().iloc[0]
                               if not df_clean[col].mode().empty else "Unknown")
                df_clean[col] = df_clean[col].fillna(most_common)
            else:
                # S'il n'y a que des NA, remplacer par "Unknown"
                df_clean[col] = df_clean[col].fillna("Unknown")

    return df_clean


def _impute_knn(df: pd.DataFrame, n_neighbors: int) -> pd.DataFrame:
    """
    Imputation KNN pour les colonnes numériques,
    et imputation simple pour les colonnes catégorielles.
    
    Args:
        df: DataFrame à traiter
        n_neighbors: Nombre de voisins pour KNN
        
    Returns:
        DataFrame avec valeurs imputées
    """
    df_clean = df.copy()

    # Imputation KNN pour colonnes numériques
    numeric_cols = df_clean.select_dtypes(include=['number']).columns

    # Supprimer temporairement les avertissements pour les calculs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Traiter chaque colonne numérique séparément
        for col in numeric_cols:
            if df_clean[col].isna().any():  # S'il y a des valeurs manquantes
                _impute_column_knn(df_clean, col, n_neighbors, numeric_cols)

    # Pour les colonnes catégorielles, utiliser l'imputation simple
    object_cols = df_clean.select_dtypes(include=['object']).columns
    for col in object_cols:
        if df_clean[col].isna().any():
            if df_clean[col].notna().any():
                most_common = (df_clean[col].mode().iloc[0]
                               if not df_clean[col].mode().empty else "Unknown")
                df_clean[col] = df_clean[col].fillna(most_common)
            else:
                df_clean[col] = df_clean[col].fillna("Unknown")

    return df_clean


def _impute_column_knn(
        df: pd.DataFrame,
        target_col: str,
        n_neighbors: int,
        numeric_cols: List[str]
) -> None:
    """
    Impute une colonne numérique spécifique en utilisant KNN.
    Modifie le DataFrame en place.
    
    Args:
        df: DataFrame contenant les données
        target_col: Colonne cible à imputer
        n_neighbors: Nombre de voisins pour KNN
        numeric_cols: Liste des colonnes numériques disponibles
    """
    # Identifier les colonnes prédicteurs potentiels
    predictors = _get_predictors(df, target_col, numeric_cols)

    if not predictors:  # Pas de prédicteurs disponibles
        _impute_column_simple(df, target_col)
        return

    # Créer sous-dataframe avec colonne cible et prédicteurs
    cols_to_use = predictors + [target_col]
    sub_df = df[cols_to_use].copy()

    try:
        # Standardiser et appliquer KNN
        sub_df_scaled, cols_with_variance = _standardize_dataframe(sub_df)

        if not cols_with_variance:
            _impute_column_simple(df, target_col)
            return

        # Configurer et appliquer l'imputation KNN
        k = min(n_neighbors, len(sub_df) - 1)
        if k <= 0:
            _impute_column_simple(df, target_col)
            return

        imputed_values = _apply_knn(sub_df_scaled, k)

        # Reconvertir à l'échelle d'origine
        imputed_df = pd.DataFrame(
            imputed_values,
            index=sub_df.index,
            columns=sub_df_scaled.columns
        )

        # Déstandardiser uniquement la colonne cible
        if target_col in cols_with_variance:
            scaler = StandardScaler()
            scaler.fit(sub_df[target_col].fillna(0).values.reshape(-1, 1))
            values = imputed_df[target_col].values.reshape(-1, 1)
            imputed_df[target_col] = scaler.inverse_transform(values).flatten()

        # Mettre à jour la colonne cible
        df[target_col] = imputed_df[target_col]

    except (ValueError, RuntimeError) as e:
        warnings.warn(f"Erreur lors de l'imputation KNN de {target_col}: {str(e)}")
        _impute_column_simple(df, target_col)


def _impute_column_simple(df: pd.DataFrame, col: str) -> None:
    """
    Impute une colonne avec la méthode simple (médiane ou 0).
    Modifie le DataFrame en place.
    
    Args:
        df: DataFrame contenant les données
        col: Colonne à imputer
    """
    if df[col].notna().any():
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
    else:
        df[col] = df[col].fillna(0)


def _get_predictors(
        df: pd.DataFrame,
        target_col: str,
        numeric_cols: List[str]
) -> List[str]:
    """
    Sélectionne les meilleurs prédicteurs pour une colonne cible.
    
    Args:
        df: DataFrame contenant les données
        target_col: Colonne cible
        numeric_cols: Liste des colonnes numériques disponibles
        
    Returns:
        Liste des colonnes prédicteurs sélectionnées
    """
    # Identifier colonnes avec des valeurs non-NA pour servir de prédicteurs
    potential_predictors = []
    for c in numeric_cols:
        if c != target_col and df[c].notna().any() and df[c].std() > 0:
            potential_predictors.append(c)

    if not potential_predictors:
        return []

    # Limiter le nombre de prédicteurs (maximum 10)
    if len(potential_predictors) > 10:
        return _select_best_predictors(df, target_col, potential_predictors)
    else:
        return potential_predictors


def _select_best_predictors(
        df: pd.DataFrame,
        target_col: str,
        potential_predictors: List[str]
) -> List[str]:
    """
    Sélectionne les meilleurs prédicteurs basés sur la corrélation.
    
    Args:
        df: DataFrame contenant les données
        target_col: Colonne cible
        potential_predictors: Liste des prédicteurs potentiels
        
    Returns:
        Liste des 10 meilleurs prédicteurs
    """
    # Vérifier si la colonne cible a de la variabilité
    if not (df[target_col].notna().any() and df[target_col].std() > 0):
        return potential_predictors[:10]

    correlations = {}
    for pred in potential_predictors:
        try:
            # Calculer corrélation pour lignes avec valeurs non-NA
            mask = df[[target_col, pred]].notna().all(axis=1)
            if mask.sum() >= 3:  # Au moins 3 points
                corr = df.loc[mask, target_col].corr(df.loc[mask, pred])
                if not np.isnan(corr):
                    correlations[pred] = abs(corr)
        except ValueError:
            # Ignorer cette paire en cas d'erreur
            continue

    # Trier par corrélation décroissante
    if correlations:
        sorted_predictors = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        return [p[0] for p in sorted_predictors[:10]]

    return potential_predictors[:10]


def _standardize_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Standardise les colonnes d'un DataFrame.
    
    Args:
        df: DataFrame à standardiser
        
    Returns:
        DataFrame standardisé et liste des colonnes avec variance
    """
    scaler = StandardScaler()
    sub_df_filled = df.fillna(0)  # Remplir NA par 0 pour standardisation

    # Identifier colonnes avec variabilité
    cols_with_variance = [c for c in sub_df_filled.columns if sub_df_filled[c].std() > 0]

    # Standardiser uniquement colonnes avec variance
    sub_df_scaled = pd.DataFrame(index=df.index)
    for c in sub_df_filled.columns:
        if c in cols_with_variance:
            values = sub_df_filled[c].values.reshape(-1, 1)
            sub_df_scaled[c] = scaler.fit_transform(values).flatten()
        else:
            sub_df_scaled[c] = 0

    return sub_df_scaled, cols_with_variance


def _apply_knn(df: pd.DataFrame, n_neighbors: int) -> np.ndarray:
    """
    Applique l'imputation KNN à un DataFrame.
    
    Args:
        df: DataFrame standardisé
        n_neighbors: Nombre de voisins
        
    Returns:
        Array NumPy avec valeurs imputées
    """
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
    return imputer.fit_transform(df)
