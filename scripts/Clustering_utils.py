import sys
sys.path.append('..')
from scripts.Exploration_utils import load_data, check_missing_values
from scripts.Cleaning_utils import (
    preprocess_dataframe,    
    get_feature_names,
    analyze_correlations,
    analyze_feature_importance
)
import pandas as pd	


def prepare_features_for_clustering(df, max_categories=30, min_unique_ratio=0.05):
    """
    Prépare les features pour la clusterisation en suivant une approche structurée
    
    Args:
        df: DataFrame original
        max_categories: Nombre maximum de catégories autorisées pour les variables catégorielles
        min_unique_ratio: Ratio minimum de valeurs uniques pour les variables numériques
    
    Returns:
        DataFrame préparé pour la clusterisation
    """
    print("1. Prétraitement initial du DataFrame")
    df_clean, preprocess_info = preprocess_dataframe(
        df,
        max_categories=max_categories,
        min_unique_ratio=min_unique_ratio
    )
    
    print("\n2. Résumé des colonnes conservées:")
    print(f"- Numériques: {len(preprocess_info['numerical_columns'])}")
    print(f"- Catégorielles ordinales: {len(preprocess_info['categorical_ordinal'])}")
    print(f"- Catégorielles nominales: {len(preprocess_info['categorical_nominal'])}")
    
    print("\n3. Colonnes numériques optimisées (downcasting):")
    for col, old_type, new_type in preprocess_info['downcasted_columns']:
        print(f"- {col}: {old_type} -> {new_type}")
    
    print("\n4. Colonnes catégorielles filtrées (> {max_categories} catégories supprimées):")
    categorical_dropped = [col for col, reason in preprocess_info['dropped_columns'] 
                         if reason == 'too_many_categories']
    for col in categorical_dropped:
        print(f"- {col}")
    
    # Analyse des corrélations pour les variables numériques
    print("\n5. Analyse des corrélations entre variables numériques")
    correlation_info = analyze_correlations(
        df_clean[preprocess_info['numerical_columns']], 
        threshold=0.7
    )
    
    # Affichage des corrélations fortes
    if correlation_info['strong_correlations']:
        print("\nVariables fortement corrélées à examiner:")
        for corr in correlation_info['strong_correlations']:
            print(f"- {corr['var1']} - {corr['var2']}: {corr['correlation']:.2f}")
    
    return df_clean, preprocess_info

def analyze_and_select_features(df, max_categories=30, min_unique_ratio=0.01):
    """
    Analyse et sélectionne automatiquement les colonnes pertinentes du DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame d'entrée
    max_categories : int
        Nombre maximum de catégories autorisées pour les variables catégorielles
    min_unique_ratio : float
        Ratio minimum de valeurs uniques pour conserver une variable
    
    Returns:
    --------
    dict : Informations sur les features sélectionnées et leur type
    """
    feature_info = {
        'numerical_columns': [],
        'categorical_ordinal': [],
        'categorical_nominal': [],
        'dropped_columns': [],
        'downcasted_columns': {}
    }
    
    # Liste des préfixes/suffixes suggérant une variable ordinale
    ordinal_indicators = ['_level', '_grade', '_score', '_rating', 'grade_', 'level_', 'score_']
    
    for column in df.columns:
        # Ignorer les colonnes avec trop de valeurs manquantes
        missing_ratio = df[column].isna().mean()
        if missing_ratio > 0.5:
            feature_info['dropped_columns'].append((column, 'too_many_missing'))
            continue
            
        # Analyse du type de données
        dtype = df[column].dtype
        n_unique = df[column].nunique()
        unique_ratio = n_unique / len(df)
        
        # Tentative de downcasting pour les variables numériques
        if pd.api.types.is_numeric_dtype(dtype):
            original_size = df[column].memory_usage(deep=True)
            downcasted = pd.to_numeric(df[column], downcast='integer')
            if downcasted.dtype != dtype:
                feature_info['downcasted_columns'][column] = {
                    'original_dtype': dtype,
                    'new_dtype': downcasted.dtype,
                    'memory_saved': original_size - downcasted.memory_usage(deep=True)
                }
            
            # Classification comme numérique si assez de valeurs uniques
            if unique_ratio > min_unique_ratio:
                feature_info['numerical_columns'].append(column)
            else:
                feature_info['dropped_columns'].append((column, 'low_variance'))
                
        # Analyse des variables catégorielles
        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
            if n_unique > max_categories:
                feature_info['dropped_columns'].append((column, 'too_many_categories'))
                continue
                
            # Détection des variables ordinales
            is_ordinal = any(ind in column.lower() for ind in ordinal_indicators)
            if is_ordinal:
                feature_info['categorical_ordinal'].append(column)
            else:
                feature_info['categorical_nominal'].append(column)
    
    return feature_info