import pandas as pd
import numpy as np
import warnings
from typing import List, Optional, Literal
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def clean_dataset(df: pd.DataFrame,
                  columns_to_drop: Optional[List[str]] = None,
                  missing_threshold: float = 0.8,
                  imputation_method: Literal['simple', 'knn'] = 'simple',
                  n_neighbors: int = 5,
                  random_state: int = 42) -> pd.DataFrame:
    """
    Nettoie un dataset en effectuant les opérations suivantes:
    - Supprime les colonnes non pertinentes spécifiées
    - Supprime les lignes vides et dupliquées
    - Supprime les lignes avec trop de valeurs manquantes (par défaut > 80%)
    - Impute des valeurs pour les valeurs manquantes restantes
    
    Args:
        df: Le DataFrame pandas à nettoyer
        columns_to_drop: Liste des colonnes à supprimer (si None, utilise une liste prédéfinie)
        missing_threshold: Seuil de valeurs manquantes au-delà duquel une ligne est supprimée (0.0 à 1.0)
        imputation_method: Méthode d'imputation à utiliser ('simple' pour médiane/mode, 'knn' pour KNN)
        n_neighbors: Nombre de voisins à utiliser pour la méthode KNN (uniquement si imputation_method='knn')
        random_state: Pour la reproductibilité des résultats
    
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
    # Calculer le pourcentage de valeurs manquantes pour chaque ligne
    missing_percentage = df_clean.isnull().mean(axis=1)
    df_clean = df_clean[missing_percentage < missing_threshold]

    # 4. Imputer les valeurs manquantes
    if imputation_method == 'simple':
        # Imputation simple (médiane pour les numériques, mode pour les catégorielles)
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

        # Pour les colonnes de type objet (texte), utiliser la valeur la plus fréquente ou "Unknown"
        object_cols = df_clean.select_dtypes(include=['object']).columns
        for col in object_cols:
            if df_clean[col].isna().any():  # Vérifier s'il y a des valeurs manquantes
                # Vérifier s'il y a des valeurs non-NA pour trouver la plus fréquente
                if df_clean[col].notna().any():
                    # Utiliser la valeur la plus fréquente, ou "Unknown" s'il n'y en a pas
                    most_common = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else "Unknown"
                    df_clean[col] = df_clean[col].fillna(most_common)
                else:
                    # S'il n'y a que des NA, remplacer par "Unknown"
                    df_clean[col] = df_clean[col].fillna("Unknown")

    elif imputation_method == 'knn':
        # Imputation KNN (pour les colonnes numériques uniquement)
        numeric_cols = df_clean.select_dtypes(include=['number']).columns

        # Supprimer temporairement les avertissements pour les calculs de corrélation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # Pour les colonnes numériques, traiter chaque colonne séparément pour éviter des problèmes de dimensionnalité
            for col in numeric_cols:
                if df_clean[col].isna().any():  # S'il y a des valeurs manquantes dans cette colonne
                    # Identifier les colonnes avec des valeurs non-NA pour servir de prédicteurs
                    potential_predictors = []
                    for c in numeric_cols:
                        if c != col and df_clean[c].notna().any():
                            # Vérifier que la colonne a de la variabilité (écart-type non nul)
                            if df_clean[c].std() > 0:
                                potential_predictors.append(c)

                    if not potential_predictors:
                        # S'il n'y a pas de prédicteurs disponibles, utiliser la médiane
                        if df_clean[col].notna().any():
                            median_value = df_clean[col].median()
                            df_clean[col] = df_clean[col].fillna(median_value)
                        else:
                            df_clean[col] = df_clean[col].fillna(0)
                        continue

                    # Limiter le nombre de prédicteurs pour éviter la haute dimensionnalité
                    # et le surapprentissage - utiliser au maximum 10 prédicteurs
                    if len(potential_predictors) > 10:
                        # Sélectionner les prédicteurs les plus corrélés avec la colonne cible
                        correlations = {}

                        # Vérifier si la colonne cible a de la variabilité
                        if df_clean[col].notna().any() and df_clean[col].std() > 0:
                            for pred in potential_predictors:
                                try:
                                    # Calculer la corrélation uniquement pour les lignes où les deux colonnes ont des valeurs non-NA
                                    mask = df_clean[[col, pred]].notna().all(axis=1)
                                    if mask.sum() >= 3:  # Au moins 3 points pour une corrélation significative
                                        corr = df_clean.loc[mask, col].corr(df_clean.loc[mask, pred])
                                        if not np.isnan(corr):
                                            correlations[pred] = abs(
                                                corr)  # Prendre la valeur absolue de la corrélation
                                except:
                                    # En cas d'erreur, ignorer cette paire
                                    pass

                            # Trier les prédicteurs par corrélation décroissante et prendre les 10 premiers
                            if correlations:
                                sorted_predictors = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
                                predictors = [p[0] for p in sorted_predictors[:10]]
                            else:
                                # Si aucune corrélation valide, prendre les 10 premiers prédicteurs
                                predictors = potential_predictors[:10]
                        else:
                            # Si la colonne cible n'a pas de variabilité, prendre les 10 premiers prédicteurs
                            predictors = potential_predictors[:10]
                    else:
                        predictors = potential_predictors

                    if not predictors:  # Si aucun prédicteur valide
                        if df_clean[col].notna().any():
                            median_value = df_clean[col].median()
                            df_clean[col] = df_clean[col].fillna(median_value)
                        else:
                            df_clean[col] = df_clean[col].fillna(0)
                        continue

                    # Créer un sous-dataframe avec la colonne cible et ses prédicteurs
                    cols_to_use = predictors + [col]
                    sub_df = df_clean[cols_to_use].copy()

                    try:
                        # Standardiser les données
                        scaler = StandardScaler()
                        sub_df_filled = sub_df.fillna(0)  # Remplir les NA par 0 pour la standardisation

                        # Vérifier si chaque colonne a de la variabilité
                        cols_with_variance = [c for c in sub_df_filled.columns if sub_df_filled[c].std() > 0]

                        if not cols_with_variance:
                            # Si aucune colonne n'a de variabilité, utiliser la médiane simple
                            if df_clean[col].notna().any():
                                median_value = df_clean[col].median()
                                df_clean[col] = df_clean[col].fillna(median_value)
                            else:
                                df_clean[col] = df_clean[col].fillna(0)
                            continue

                        # Standardiser uniquement les colonnes avec variance
                        sub_df_scaled = pd.DataFrame(index=sub_df.index)
                        for c in sub_df_filled.columns:
                            if c in cols_with_variance:
                                values = sub_df_filled[c].values.reshape(-1, 1)
                                sub_df_scaled[c] = scaler.fit_transform(values).flatten()
                            else:
                                sub_df_scaled[c] = 0  # Colonnes sans variance sont mises à 0

                        # Configurer et appliquer l'imputation KNN
                        k = min(n_neighbors, len(sub_df) - 1)
                        if k > 0:  # S'assurer que k est au moins 1
                            imputer = KNNImputer(n_neighbors=k, weights='distance')
                            imputed_values = imputer.fit_transform(sub_df_scaled)

                            # Reconvertir à l'échelle d'origine pour les colonnes avec variance
                            imputed_df = pd.DataFrame(imputed_values, index=sub_df.index, columns=sub_df_scaled.columns)
                            for c in cols_with_variance:
                                if c == col:  # Seulement déstandardiser la colonne cible
                                    values = imputed_df[c].values.reshape(-1, 1)
                                    imputed_df[c] = scaler.inverse_transform(values).flatten()

                            # Mettre à jour la colonne cible dans le DataFrame original
                            df_clean[col] = imputed_df[col]
                        else:
                            # Si k est 0 (cas rare), utilisez la médiane simple
                            if df_clean[col].notna().any():
                                median_value = df_clean[col].median()
                                df_clean[col] = df_clean[col].fillna(median_value)
                            else:
                                df_clean[col] = df_clean[col].fillna(0)
                    except Exception as e:
                        # En cas d'erreur, utiliser la méthode simple
                        if df_clean[col].notna().any():
                            median_value = df_clean[col].median()
                            df_clean[col] = df_clean[col].fillna(median_value)
                        else:
                            df_clean[col] = df_clean[col].fillna(0)

        # Pour les colonnes de type objet (texte), utiliser la valeur la plus fréquente ou "Unknown"
        object_cols = df_clean.select_dtypes(include=['object']).columns
        for col in object_cols:
            if df_clean[col].isna().any():  # Vérifier s'il y a des valeurs manquantes
                # Vérifier s'il y a des valeurs non-NA pour trouver la plus fréquente
                if df_clean[col].notna().any():
                    # Utiliser la valeur la plus fréquente, ou "Unknown" s'il n'y en a pas
                    most_common = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else "Unknown"
                    df_clean[col] = df_clean[col].fillna(most_common)
                else:
                    # S'il n'y a que des NA, remplacer par "Unknown"
                    df_clean[col] = df_clean[col].fillna("Unknown")

    return df_clean
